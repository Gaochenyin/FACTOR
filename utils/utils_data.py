# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:53:41 2025

@author: chyga
"""
import torch
import numpy as np
import pandas as pd
import math
import random
from abc import abstractmethod
from lightning.pytorch import LightningDataModule
from torch.distributions import AffineTransform
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from omegaconf import DictConfig
from torch.distributions import TransformedDistribution, Normal, \
    Independent, Uniform, Categorical, MixtureSameFamily, MultivariateNormal
from pathlib import Path
from preprocessing import preprocess
# Source: https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
class StandardScaler:
    def __init__(self, mean=None, scale=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        """
        self.mean_ = mean
        self.scale_ = scale
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean_ = torch.mean(values, dim=dims)
        self.scale_ = torch.std(values, dim=dims)
        self.transformer = AffineTransform(loc=self.mean_, scale=self.scale_ + self.epsilon).inv
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
    def __init__(
        self,
        num_workers=0,
        pin_memory=False,
        seed=0,
    ):
        super().__init__()

        # This line allows to access `__init__` arguments with `self.hparams` attribute
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
        sample_idx = rng.choice(N, min(N, math.ceil(max_size / train_ratio)), replace=False)
        return x[sample_idx], y[sample_idx]

    def load_datasets(self):
        x, y = self.get_data()
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        max_size = 20000
        x, y = self.subsample(x, y, max_size=max_size)
        tensor_data = TensorDataset(x, y)
        self.total_size = len(tensor_data)

        # Convert ratios to number of elements in the dataset
        splits_size = np.array(self.train_val_calib_test_split_ratio) * len(tensor_data)
        # log.debug(f'Total size: {len(tensor_data)}')
        # log.debug(f'Splits size before: {splits_size}')
        calib_index = 2
        to_remove_from_calib = max(0, splits_size[calib_index] - 2048)
        splits_size[calib_index] -= to_remove_from_calib
        # Don't add points in splits that should be empty (e.g., interleaving is often empty)
        mask = (splits_size != 0) & (np.arange(len(splits_size)) != calib_index)
        splits_size[mask] += to_remove_from_calib / mask.sum()
        splits_size = splits_size.astype(int)
        splits_size[-1] = len(tensor_data) - splits_size[:-1].sum()
        # log.debug(f'Splits size after: {splits_size}')

        (self.data_train, self.data_val, self.data_calib, self.data_test,) = random_split(
            dataset=tensor_data,
            lengths=splits_size.tolist(),
            generator=torch.Generator().manual_seed(1),
        )

        x, y = self.data_train[:]
        self.scaler_x = StandardScaler().fit(x)
        self.scaler_y = StandardScaler().fit(y)

        # if self.rc.config.normalize:
        # self.data_train = self.make_scaled_dataset(self.data_train)
        # self.data_val = self.make_scaled_dataset(self.data_val)
        # self.data_calib = self.make_scaled_dataset(self.data_calib)
        # self.data_test = self.make_scaled_dataset(self.data_test)

        # Make the size of the inputs accessible to the models
        first_x, first_y = self.data_train[0]
        self.input_dim = first_x.shape[0]
        self.output_dim = first_y.shape[0]

    def get_dataloader(self, dataset, drop_last=False, shuffle=False, batch_size=None):
        if batch_size is None:
            batch_size = 256 #self.rc.config.default_batch_size
        batch_size = min(len(dataset), batch_size)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory='cpu',
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers = True
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
        # This is a custom container to keep together but does not
        # replicate Independent functionality directly.
        self.categorical = categorical_dist
        self.uniform = uniform_dist

    def sample(self, size):
        # Return samples from both distributions in tuple form
        return torch.stack((self.categorical.sample(size),
                self.uniform.sample(size)), dim = 1)

    def log_prob(self, value):
        # Assume `value` is a tuple (cat_value, uniform_value)
        cat_value, uniform_value = value
        return self.categorical.log_prob(cat_value) + self.uniform.log_prob(uniform_value)


class UnimodalHeteroscedastic(DatasetGenerator):
    def __init__(self, scale_power=1.):
        self.scale_power = scale_power
        self.default_size = 2500

    def dist_x(self):
        categorical_probs = torch.tensor([1/3, 1/3, 1/3])  # probabilities for categories 1, 2, 3
        categorical_dist = Categorical(probs=categorical_probs)
        uniform_dist = Uniform(low=torch.tensor(0.5), high=torch.tensor(2.0))
        return CombinedDistribution(categorical_dist, uniform_dist)
        # return Independent(Uniform(torch.tensor([0.5]), torch.tensor([2.])), 1)

    def dist_y(self, x):
        x = (x[:, 0] + 1) * x[:, 1]
        loc = torch.full((x.shape[0], 2), 0., device=x.device)
        scale = x**self.scale_power
        scale = scale[:, None].repeat(1, 2)
        dist = Independent(Normal(loc=loc, scale=scale), 1)
        return dist
    
class BimodalHeteroscedastic(DatasetGenerator):
    def __init__(self, scale_power=1.):
        self.scale_power = scale_power
        self.default_size = 10000
    
    def dist_x(self):
        categorical_probs = torch.tensor([1/3, 1/3, 1/3])  # probabilities for categories 1, 2, 3
        categorical_dist = Categorical(probs=categorical_probs)
        uniform_dist = Uniform(low=torch.tensor(0.5), high=torch.tensor(2.0))
        return CombinedDistribution(categorical_dist, uniform_dist)

    def dist_y(self, x):
        x = (x[:, 0] + 1) * x[:, 1]
        n = x.shape[0]
        m1 = torch.full((n, 2), 4., device=x.device)
        m2 = torch.full((n, 2), -4., device=x.device)
        loc = torch.stack([m1, m2], dim=1)
        scale = torch.stack([
            x**self.scale_power,
            (1 / x)**self.scale_power
        ], dim=1)
        scale = scale[:, :, None].repeat(1, 1, 2)
        dist = MixtureSameFamily(
            mixture_distribution=Categorical(torch.full((n, 2), 0.5, device=x.device)),
            component_distribution=Independent(Normal(loc=loc, scale=scale), 1)
        )
        return dist
 
class MVNDependent(DatasetGenerator):
    def __init__(self, d = 1, rho = 0):
        self.d = d
        self.rho = rho
        self.default_size = 2500
        self.cov = self.create_cov()
    
    def create_cov(self):
        cov = torch.full((self.d, self.d), self.rho) + \
            torch.eye(self.d) * (1 - self.rho)
        # cov = torch.rand(self.d, self.d) * 2 - 1
        # cov = cov @ cov.t()
        return cov

    def dist_x(self):
        categorical_probs = torch.tensor([1/3, 1/3, 1/3])  # probabilities for categories 1, 2, 3
        categorical_dist = Categorical(probs=categorical_probs)
        uniform_dist = Uniform(low=torch.tensor(0.5), high=torch.tensor(2.0))
        return CombinedDistribution(categorical_dist, uniform_dist)

    def dist_y(self, x):
        # print(x)
        s = x[:, [0]]
        x = (x[:, 0] + 1) * x[:, 1]
        loc = torch.full((x.shape[0], self.d), 0., device=x.device) + s
        cov = self.cov[None, :, :] * x[:, None, None]
        dist = MultivariateNormal(loc=loc, covariance_matrix=cov)
        return dist    

class OneMoonHeteroskedastic(DatasetGenerator):
    def __init__(self, k=100, noise=0.2):
        self.k = k
        self.noise = noise
        self.default_size = 10000

    def dist_x(self):
        categorical_probs = torch.tensor([1/3, 1/3, 1/3])  # probabilities for categories 1, 2, 3
        categorical_dist = Categorical(probs=categorical_probs)
        uniform_dist = Uniform(low=torch.tensor(0.5), high=torch.tensor(2.0))
        return CombinedDistribution(categorical_dist, uniform_dist)

    def dist_y(self, x):
        batch_size = x.shape[0]
        x = (x[:, 0] + 1) * x[:, 1]

        alpha = torch.linspace(0, torch.pi, self.k, device=x.device)
        locs = torch.stack([alpha.cos(), alpha.sin()], dim=-1)
        locs -= torch.tensor([0., 0.5])
        locs[:, 1] *= -1
        locs = locs[None, :, :] * (1.3 - x[:, None, None])
        scale = torch.full((self.k, 2), self.noise, device=x.device)
        scale = scale.tile(batch_size, 1, 1)
        comp_dist = Independent(Normal(locs, scale), 1)
        cat_dist = Categorical(probs=torch.ones(batch_size, self.k, device=x.device))
        dist = MixtureSameFamily(cat_dist, comp_dist)
        return dist
    
class ToyDataModule(BaseDataModule):
    def __init__(self, *args, data_set = 'unimodel',
                 train_val_calib_test_split_ratio = (0.4, 0.2, 0.4, 0.0),
                 size=None, **kwargs):
        self.size = size
        self.data_set = data_set
        self.train_val_calib_test_split_ratio = \
            train_val_calib_test_split_ratio
        super().__init__(*args, **kwargs)

    def get_data(self, seed=0):
        seed_everything(seed)
        # could add different models
        if self.data_set == 'unimodel':
            self.distribution_generator = UnimodalHeteroscedastic()
        elif self.data_set == 'bimodel':
            self.distribution_generator = BimodalHeteroscedastic()
        elif self.data_set.startswith('mvnormal_'):
            params = self.data_set.split('_')[-2:]
            d = int(params[0]); rho = float(params[1])
            self.distribution_generator = MVNDependent(d, rho)
        elif self.data_set == 'moon':
            self.distribution_generator = OneMoonHeteroskedastic()
            
        size = self.size
        if self.distribution_generator is not None:
            if self.size is None:
                size = self.distribution_generator.default_size
            x, y = self.distribution_generator.generate(size)
            x, y = x.numpy(), y.numpy()
            return x, y
        # Ideally, we have access to a Distribution object such that we can also measure 
        # the density of the oracle distribution.
        # However, these datasets do not give access to the density.
        # x, y = generate_data_original(self.size)
        # return x, y
    
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
class RealDataModule(BaseDataModule):
    
    def __init__(self, *args, 
                 data_dir = 'data',
                 data_name = 'households',
                 train_val_calib_test_split_ratio = (0.4, 0.2, 0.4, 0.0),
                 size=None, **kwargs):
        self.size = size
        self.data_dir = data_dir
        self.data_name = data_name
        self.train_val_calib_test_split_ratio = \
            train_val_calib_test_split_ratio
        super().__init__(*args, **kwargs)
        
    def get_data(self):
        return get_data(self.data_dir, self.data_name)

def get_data(data_dir, data_name):
    
    if data_name in ['households', 'households_subset', # feldman
                     'air', 'births1', 'births2',  # cevid
                     'ansur2', 'calcofi', # barrio
                     'energy', # wang
                     'vaccine' # Larry's data
                     ]:
        return get_data_continuous(data_dir, data_name)
    
    
    elif data_name in ['bio', 'blog_data', 
                       'house', 'house_subset', 
                       'wage' # cevid
                       ]:
        return get_data_discrete(data_dir, data_name)
    # elif data_name in ['ansur2']:
    #     return get_data_del_barrio(data_dir, data_name)
    # elif group == 'del_barrio':
    #     return get_data_del_barrio(data_dir, name)
    # elif group == 'feldman':
    #     return get_data_feldman(data_dir, name)
    # elif group == 'mulan':
    #     return get_data_mulan(data_dir, name)
    # elif group == 'wang':
    #     return get_data_wang(data_dir, name)
    else:
        raise ValueError(f'Unknown group: {data_name}')


# dataloader for real-data (continuous and discrete outcomes)    
def get_data_continuous(data_dir, data_name): # Camehl, 2024
    targets_dict = {
        # 'households': ['inc', 'food', 'house', 'utili'],
        'households': ['food', 'house'], # 2D for visulization
        'households_subset': ['food', 'house'], # 2D for visulization
        'bio': ['F7', 'F9'],
        'air': ['max_PM2.5', 'max_NO2',	'max_O3', 'max_PM10', 
                'max_CO', 'max_SO2'],
        'births1': ['pregnancy_duration', 'birthweight'],
        'energy': ['Y1', 'Y2'],
        # 'bio': ['F7', 'F9'],
        'vaccine': ['..', '..'], # 2D for visulization
    }

    protected_dict = {
        # 'households': ['inc', 'food', 'house', 'utili'],
        'households': 'region', # 2D for visulization
        'households_subset': 'region', # 2D for visulization
        'bio': '..', # 2D for visulization
        'air': 'Weekday',
        'births1': 'education_mother',
        'energy': 'X8',
        'vaccine': '..', # 2D for visulization
    }

    
    data_path = Path(data_dir)
    path = data_path / f'{data_name}.csv'
    
    # read data
    df = pd.read_csv(path)
    
    targets = targets_dict[data_name]
    protects = protected_dict[data_name]
    x = df[df.columns.difference(targets)]
    y = df[targets]
    # re-order the protected group to the first column
    protected_col = x.pop(protects)
    x.insert(0, protects, protected_col)
    # x, y = x.to_numpy('float32'), y.to_numpy('float32')
    x, y, categorical_mask = preprocess(x, y) # preprocess the data
    print(f"Sample size of X: {x.shape} with Group Number {len(protected_col.unique())}")
    return x, y

def get_data_discrete(data_dir, data_name):
    targets_dict = {
        'blog_data': [60, 280],
        'house_subset': ['grade', 'price'],
        'house': ['grade', 'price'], # grade is discrete, price is continuous
        'wage': ['male', 'log_wage', 'age'] # male is discrete, log_wage is continuous
    }
    
    protected_dict = {
        # 'households': ['inc', 'food', 'house', 'utili'],
        'blog_data': 'region',
        'house_subset': 'floors',
        'house': 'floors',
        'wage': 'race_white'
    }

    data_path = Path(data_dir)
    path = data_path / f'{data_name}.csv'
    if data_name == 'blog_data':
        df = pd.read_csv(path, header=None)
    elif data_name in ['bio', 'wage']:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, index_col=0)
    targets = targets_dict[data_name]
    protects = protected_dict[data_name]
    x = df[df.columns.difference(targets)]
    y = df[targets]
    # re-order the protected group to the first column
    protected_col = x.pop(protects)
    x.insert(0, protects, protected_col.astype('int'))
    
    # x, y = x.to_numpy('float32'), y.to_numpy('float32')
    x, y, categorical_mask = preprocess(x, y)
    print(f"Sample size of X: {x.shape} with Group Number {len(protected_col.unique())}")
    return x, y

# Dataset from "Nonparametric Multiple-Output Center-Outward Quantile Regression", Equation (1.7)
# There is potentially an error in the paper, as the obtained plot looks different from Fig. 2
def generate_data_original(n):
    def e_function(x):
        v = np.random.normal(0, 1, (2, x.shape[0]))
        e = np.sqrt(1 + (3/2) * np.sin(np.pi * x / 2) ** 2) * v
        return e

    def sample_Y1(x, e1):
        return np.sin((2 * np.pi / 3) * x) + e1

    def sample_Y2(x, e1, e2):
        return np.cos((2 * np.pi / 3) * x) + x**2 + e2 + e1 + 2.65 * x**4
    
    x = np.random.uniform(-1, 1, n)
    e = e_function(x)
    e1, e2 = e
    y1 = sample_Y1(x, e1)
    y2 = sample_Y2(x, e1, e2)
    x = x[:, None]
    y = np.stack([y1, y2], axis=1)
    return x, y


class OracleModel:
    """
    Oracle model that has knowledge of the data generating process.
    """
    def __init__(self):
        self.device = 'cpu'

    def fit(self, x, y):
        self.distribution_generator = self.trainer.datamodule.distribution_generator
        assert self.distribution_generator is not None

    def predict(self, x):
        x = self.trainer.datamodule.scaler_x.inverse_transform(x)
        dist = self.distribution_generator.dist_y(x)
        return TransformedDistribution(dist, self.trainer.datamodule.scaler_y.transformer)

    def to(self, device):
        # We do this to match the interface of other modules
        # but this is not necessary because there are no parameters
        self.device = device
        return self

    @classmethod
    def output_type(cls):
        return 'distribution'
    
class DefaultTrainer:

    def fit(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
        model.trainer = self
        x, y = datamodule.data_train[:]
        return model.fit(x, y)

    def test(self, model, datamodule, **kwargs):
        return model.test(datamodule)