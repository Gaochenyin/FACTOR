# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:53:41 2025

@author: chyga
"""
from abc import abstractmethod
import math
from pathlib import Path
import random

from lightning.pytorch import LightningDataModule
import numpy as np
import pandas as pd
import torch
from torch.distributions import (
    AffineTransform,
    Categorical,
    Independent,
    MixtureSameFamily,
    MultivariateNormal,
    Normal,
    TransformedDistribution,
    Uniform,
)
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from preprocessing import preprocess


class StandardScaler:
    def __init__(self, mean=None, scale=None, epsilon=1e-7):
        self.mean_ = mean
        self.scale_ = scale
        self.epsilon = epsilon
        self.transformer = None

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean_ = torch.mean(values, dim=dims)
        self.scale_ = torch.std(values, dim=dims)
        self.transformer = AffineTransform(
            loc=self.mean_, scale=self.scale_ + self.epsilon
        ).inv
        return self

    def transform(self, values):
        return self.transformer(values)

    def inverse_transform(self, values):
        return self.transformer.inv(values)


class ScaledDataset(Dataset):
    def __init__(self, dataset, scaler_x, scaler_y):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def scale(self, v, scaler):
        shape = v.shape
        if len(shape) == 1:
            v = v[None, :]
        scaled = scaler.transform(v)
        return scaled.reshape(shape)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.scale(x, self.scaler_x)
        y = self.scale(y, self.scaler_y)
        return x, y


class BaseDataModule(LightningDataModule):
    def __init__(self, num_workers=0, pin_memory=False, seed=0):
        super().__init__()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.load_datasets()

    @abstractmethod
    def get_data(self):
        pass

    def make_scaled_dataset(self, ds):
        return ScaledDataset(ds, self.scaler_x, self.scaler_y)

    def subsample(self, x, y, max_size):
        N = x.shape[0]
        rng = np.random.RandomState(1)
        train_ratio = self.train_val_calib_test_split_ratio[0]
        sample_idx = rng.choice(
            N, min(N, math.ceil(max_size / train_ratio)), replace=False
        )
        return x[sample_idx], y[sample_idx]

    def load_datasets(self):
        x, y = self.get_data()
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        max_size = 20000
        x, y = self.subsample(x, y, max_size=max_size)
        tensor_data = TensorDataset(x, y)
        self.total_size = len(tensor_data)

        splits_size = (
            np.array(self.train_val_calib_test_split_ratio) * len(tensor_data)
        ).astype(float)
        calib_index = 2
        to_remove_from_calib = max(0.0, splits_size[calib_index] - 2048)
        splits_size[calib_index] -= to_remove_from_calib

        mask = (splits_size != 0) & (np.arange(len(splits_size)) != calib_index)
        splits_size[mask] += to_remove_from_calib / mask.sum()
        splits_size = splits_size.astype(int)
        splits_size[-1] = len(tensor_data) - splits_size[:-1].sum()

        (
            self.data_train,
            self.data_val,
            self.data_calib,
            self.data_test,
        ) = random_split(
            dataset=tensor_data,
            lengths=splits_size.tolist(),
            generator=torch.Generator().manual_seed(1),
        )

        x_train, y_train = self.data_train[:]
        self.scaler_x = StandardScaler().fit(x_train)
        self.scaler_y = StandardScaler().fit(y_train)

        first_x, first_y = self.data_train[0]
        self.input_dim = first_x.shape[0]
        self.output_dim = first_y.shape[0]

    def get_dataloader(
        self, dataset, drop_last=False, shuffle=False, batch_size=None
    ):
        if batch_size is None:
            batch_size = 256
        batch_size = min(len(dataset), batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=False,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.get_dataloader(self.data_train, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val)

    def calib_dataloader(self):
        return self.get_dataloader(self.data_calib)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)


class DatasetGenerator:
    def dist_x(self):
        raise NotImplementedError

    def dist_y(self, x):
        raise NotImplementedError

    def generate(self, n):
        x = self.dist_x().sample((n,))
        y = self.dist_y(x).sample()
        return x, y


class CombinedDistribution(torch.distributions.Distribution):
    def __init__(self, categorical_dist, uniform_dist):
        super().__init__()
        self.categorical = categorical_dist
        self.uniform = uniform_dist

    def sample(self, sample_shape=torch.Size()):
        size = sample_shape[0] if sample_shape else 1
        return torch.stack(
            (self.categorical.sample((size,)), self.uniform.sample((size,))),
            dim=1,
        )

    def log_prob(self, value):
        cat_value, uniform_value = value[:, 0], value[:, 1]
        return self.categorical.log_prob(cat_value) + self.uniform.log_prob(
            uniform_value
        )


class UnimodalHeteroscedastic(DatasetGenerator):
    def __init__(self, scale_power=1.0):
        self.scale_power = scale_power
        self.default_size = 2500

    def dist_x(self):
        return CombinedDistribution(
            Categorical(probs=torch.tensor([1 / 3, 1 / 3, 1 / 3])),
            Uniform(low=torch.tensor(0.5), high=torch.tensor(2.0)),
        )

    def dist_y(self, x):
        x = (x[:, 0] + 1) * x[:, 1]
        loc = torch.full((x.shape[0], 2), 0.0, device=x.device)
        scale = (x**self.scale_power)[:, None].repeat(1, 2)
        return Independent(Normal(loc=loc, scale=scale), 1)


class MVNDependent(DatasetGenerator):
    def __init__(self, d=1, rho=0.0):
        self.d = d
        self.rho = rho
        self.default_size = 2500
        self.cov = self.create_cov()

    def create_cov(self):
        return torch.full((self.d, self.d), self.rho) + torch.eye(self.d) * (
            1.0 - self.rho
        )

    def dist_x(self):
        return CombinedDistribution(
            Categorical(probs=torch.tensor([1 / 3, 1 / 3, 1 / 3])),
            Uniform(low=torch.tensor(0.5), high=torch.tensor(2.0)),
        )

    def dist_y(self, x):
        s = x[:, [0]]
        x_scaled = (x[:, 0] + 1) * x[:, 1]
        loc = torch.full((x.shape[0], self.d), 0.0, device=x.device) + s
        cov = self.cov[None, :, :] * x_scaled[:, None, None]
        return MultivariateNormal(loc=loc, covariance_matrix=cov)


class ToyDataModule(BaseDataModule):
    def __init__(
        self,
        *args,
        data_set="unimodel",
        train_val_calib_test_split_ratio=(0.4, 0.2, 0.4, 0.0),
        size=None,
        **kwargs,
    ):
        self.size = size
        self.data_set = data_set
        self.train_val_calib_test_split_ratio = train_val_calib_test_split_ratio
        self.distribution_generator = None
        super().__init__(*args, **kwargs)

    def get_data(self, seed=0):
        seed_everything(seed)
        if self.data_set == "unimodel":
            self.distribution_generator = UnimodalHeteroscedastic()
        elif self.data_set.startswith("mvnormal_"):
            params = self.data_set.split("_")[-2:]
            self.distribution_generator = MVNDependent(
                d=int(params[0]), rho=float(params[1])
            )

        size = self.size or self.distribution_generator.default_size
        x, y = self.distribution_generator.generate(size)
        return x.numpy(), y.numpy()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class RealDataModule(BaseDataModule):
    def __init__(
        self,
        *args,
        data_dir="data",
        data_name="households",
        train_val_calib_test_split_ratio=(0.4, 0.2, 0.4, 0.0),
        size=None,
        **kwargs,
    ):
        self.size = size
        self.data_dir = data_dir
        self.data_name = data_name
        self.train_val_calib_test_split_ratio = train_val_calib_test_split_ratio
        super().__init__(*args, **kwargs)

    def get_data(self):
        return get_data(self.data_dir, self.data_name)


def get_data(data_dir, data_name):
    if data_name in ["households", "households_subset", "air", "births1", "energy"]:
        return get_data_continuous(data_dir, data_name)
    elif data_name in ["house", "wage", "house_subset"]:
        return get_data_discrete(data_dir, data_name)
    raise ValueError(f"Unknown group/dataset: {data_name}")


def get_data_continuous(data_dir, data_name):
    targets_dict = {
        "households": ["food", "house"],
        "households_subset": ["food", "house"],
        "air": [
            "max_PM2.5",
            "max_NO2",
            "max_O3",
            "max_PM10",
            "max_CO",
            "max_SO2",
        ],
        "births1": ["pregnancy_duration", "birthweight"],
        "energy": ["Y1", "Y2"],
    }
    protected_dict = {
        "households": "region",
        "households_subset": "region",
        "air": "Weekday",
        "births1": "education_mother",
        "energy": "X8",
    }

    path = Path(data_dir) / f"{data_name}.csv"
    df = pd.read_csv(path)

    targets = targets_dict[data_name]
    protects = protected_dict[data_name]
    x = df[df.columns.difference(targets)]
    y = df[targets]

    protected_col = x.pop(protects)
    x.insert(0, protects, protected_col)
    return preprocess(x, y)


def get_data_discrete(data_dir, data_name):
    targets_dict = {
        "house_subset": ["grade", "price"],
        "house": ["grade", "price"],
        "wage": ["male", "log_wage", "age"],
    }
    protected_dict = {
        "house_subset": "floors",
        "house": "floors",
        "wage": "race_white",
    }

    path = Path(data_dir) / f"{data_name}.csv"
    df = pd.read_csv(path, index_col=0 if data_name != "wage" else None)

    targets = targets_dict[data_name]
    protects = protected_dict[data_name]
    x = df[df.columns.difference(targets)]
    y = df[targets]

    protected_col = x.pop(protects)
    x.insert(0, protects, protected_col.astype(int))
    return preprocess(x, y)


class OracleModel:
    def __init__(self):
        self.device = "cpu"
        self.distribution_generator = None
        self.trainer = None

    def fit(self, x, y):
        self.distribution_generator = (
            self.trainer.datamodule.distribution_generator
        )

    def predict(self, x):
        x_inv = self.trainer.datamodule.scaler_x.inverse_transform(x)
        dist = self.distribution_generator.dist_y(x_inv)
        return TransformedDistribution(
            dist, self.trainer.datamodule.scaler_y.transformer
        )

    def to(self, device):
        self.device = device
        return self


class DefaultTrainer:
    def __init__(self):
        self.model = None
        self.datamodule = None

    def fit(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
        model.trainer = self
        x, y = datamodule.data_train[:]
        return model.fit(x, y)