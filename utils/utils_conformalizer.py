# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 12:51:58 2025

@author: chyga
"""
from abc import abstractmethod
import math

import numpy as np
import pandas as pd
import torch

from utils_moc import (
    distance_to_closest_point,
    fast_empirical_cdf,
    get_samples,
    get_samples_and_log_probs,
    latent_distance,
    pairwise_distance_to_closest_point,
)


def hpd(model, x, y, n_samples, cache=None):
    cache = cache or {}
    y_sample_shape = y.shape[:-2]
    dist = model.predict(x)
    samples, log_probs_samples = get_samples_and_log_probs(
        model, x, n_samples, cache, "samples", "log_probs"
    )
    log_probs_y = dist.log_prob(y).detach()
    cdf = fast_empirical_cdf(log_probs_samples, log_probs_y)
    return 1 - cdf


class EmptyCache:
    def __init__(self, dataloader):
        self.len = len(dataloader)

    def __getitem__(self, key):
        return {}

    def __len__(self):
        return self.len


class RegionPredictorBase:
    def __init__(self, dl_calib, model, cache_calib=None, rc=None):
        self.dl_calib = dl_calib
        self.model = model
        self.rc = rc

    def get_q(self, alpha):
        return torch.tensor(torch.nan, device=self.model.device)

    @abstractmethod
    def is_in_region(self, x, y, alpha, cache=None):
        pass


def conformal_quantile(scores, alpha):
    n = scores.shape[0]
    scores = torch.cat(
        [
            scores,
            torch.full(
                (1,) + scores.shape[1:], torch.inf, device=scores.device
            ),
        ],
        dim=0,
    )
    level = torch.tensor((1 - alpha) * (n + 1)) / (n + 1)
    return torch.quantile(
        scores, level.to(scores.device), interpolation="higher", dim=0
    )


class ConformalizerBase(RegionPredictorBase):
    def __init__(self, dl_calib, model, cache_calib=None, **kwargs):
        super().__init__(dl_calib, model, cache_calib, **kwargs)
        self.cache_calib = (
            EmptyCache(dl_calib) if cache_calib is None else cache_calib
        )
        self.scores_dict = {}

    @abstractmethod
    def get_score(self, x, y, alpha, cache=None):
        pass

    def get_calib_scores(self, alpha):
        if alpha not in self.scores_dict:
            scores = [
                self.get_score(
                    x.to(self.model.device),
                    y.to(self.model.device),
                    alpha,
                    cache_cal,
                )
                for (x, y), cache_cal in zip(self.dl_calib, self.cache_calib)
            ]
            self.scores_dict[alpha] = torch.concat(scores, dim=0)
        return self.scores_dict[alpha]

    def get_q(self, alpha):
        return conformal_quantile(self.get_calib_scores(alpha), alpha)

    def is_in_region(self, x, y, alpha, cache=None):
        return self.get_score(x, y, alpha, cache) <= self.get_q(alpha)


class ConformalizerBaseAlphaInvariant(RegionPredictorBase):
    def __init__(self, dl_calib, model, cache_calib=None, **kwargs):
        super().__init__(dl_calib, model, **kwargs)
        cache_calib = (
            EmptyCache(dl_calib) if cache_calib is None else cache_calib
        )
        self.calib_scores = [
            self.get_score(x.to(model.device), y.to(model.device), cache)
            for (x, y), cache in zip(dl_calib, cache_calib)
        ]
        self.calib_scores = torch.concat(self.calib_scores)

    @abstractmethod
    def get_score(self, x, y, cache=None):
        pass

    def get_q(self, alpha):
        return conformal_quantile(self.calib_scores, alpha)

    def is_in_region(self, x, y, alpha, cache=None):
        return self.get_score(x, y, cache) <= self.get_q(alpha)


class C_HDR(ConformalizerBaseAlphaInvariant):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)

    def get_score(self, x, y, cache=None):
        return hpd(self.model, x, y, n_samples=self.n_samples, cache=cache)


class STDQR(ConformalizerBase):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        self.mode_dict = {}
        super().__init__(dl_calib, model, **kwargs)

    def get_latent_regions(self, batch_size, alpha):
        z = self.model.model.latent_dist((batch_size,)).sample((self.n_samples,))
        d = z.shape[-1]
        distances = latent_distance(z)
        k = torch.tensor((1 - alpha) * self.n_samples).int()
        indices = torch.argsort(distances, dim=0)

        z_in = z[indices[:k], torch.arange(batch_size)]
        z_out = z[indices[k:], torch.arange(batch_size)]
        return z_in, z_out

    def get_initial_coverage(self, alpha):
        is_in_region = []
        for (x, y), cache_cal in zip(self.dl_calib, self.cache_calib):
            x, y = x.to(self.model.device), y.to(self.model.device)
            z_in, _ = self.get_latent_regions(x.shape[0], alpha)
            y_in = self.model.model.forward(z_in, x)
            distances = pairwise_distance_to_closest_point(y_in)
            radiuses = torch.quantile(distances, 1 - alpha, dim=0)
            is_in_region_per_batch = (
                distance_to_closest_point(y_in, y) <= radiuses
            )
            is_in_region.append(is_in_region_per_batch)
        return torch.concat(is_in_region, dim=0).float().mean()

    def get_mode(self, alpha):
        if alpha in self.mode_dict:
            return self.mode_dict[alpha]
        coverage = self.get_initial_coverage(alpha)
        self.mode_dict[alpha] = "grow" if coverage <= 1 - alpha else "shrink"
        return self.mode_dict[alpha]

    def get_score(self, x, y, alpha, cache=None):
        z_in, z_out = self.get_latent_regions(x.shape[0], alpha)
        mode = self.get_mode(alpha)
        if mode == "grow":
            y_in = self.model.model.forward(z_in, x)
            return distance_to_closest_point(y_in, y)
        elif mode == "shrink":
            y_out = self.model.model.forward(z_out, x)
            return -distance_to_closest_point(y_out, y)
        raise ValueError(f"Unsupported mode: {mode}")


def to_latent_space(model, x, y):
    return model.model.flow.forward(y, x)


class L_CP(ConformalizerBaseAlphaInvariant):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)

    def get_score(self, x, y, cache=None):
        return latent_distance(to_latent_space(self.model, x, y))


def find_optimal_cutoff_interval(arr, probs, percentage_to_include=0.95):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()

    n = len(arr)
    sorted_idx = np.argsort(arr)
    sorted_arr = arr[sorted_idx]
    sorted_probs = probs[sorted_idx]

    num_to_include = math.ceil(n * percentage_to_include)
    min_width = float("inf")
    best_lower = sorted_arr[0]
    best_upper = sorted_arr[-1]

    for i in range(n - num_to_include + 1):
        current_width = (1 / sorted_probs[i : (i + num_to_include)]).sum()
        if current_width < min_width:
            min_width = current_width
            best_lower = sorted_arr[i]
            best_upper = sorted_arr[i + num_to_include - 1]

    return best_lower, best_upper


class L_CP_DP(L_CP):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        super().__init__(dl_calib, model, n_samples, **kwargs)
        x_cal, y_cal = dl_calib.dataset[:]
        self.probs = torch.exp(
            self.model.predict(x_cal).log_prob(y_cal).detach()
        )

    def get_q(self, alpha):
        return find_optimal_cutoff_interval(
            self.calib_scores, self.probs, 1 - alpha
        )

    def is_in_region(self, x, y, alpha, cache=None):
        lower_q, upper_q = self.get_q(alpha)
        score = self.get_score(x, y, cache)
        return torch.logical_and(score >= lower_q, score <= upper_q)


class CL_CP(ConformalizerBaseAlphaInvariant):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)

    def get_score(self, x, y, cache=None, interpolation="linear", sigma=1e-4):
        z_unfairMat = (
            latent_distance(to_latent_space(self.model, x, y))
            .detach()
            .numpy()
        )

        def synchronized_z(z_unfair):
            df_scores = pd.DataFrame(
                {"pred_col": z_unfair, "protected_col": x[:, 0]}
            )
            calib_sorted = df_scores.sort_values(by=["pred_col"])
            groups = calib_sorted.groupby("protected_col")
            n_calib = df_scores.shape[0]
            taus = np.linspace(0, 1, n_calib)

            Ns = groups.size()
            ps = groups.size() / n_calib
            y_fair = np.zeros(n_calib)

            for group_name, group_df in groups:
                original_subgroup_sorted = (
                    group_df["pred_col"].values
                    + np.random.uniform(-1 * sigma, 1 * sigma)
                )
                fair_subgroup_sorted = np.empty(n_calib)
                for k in range(n_calib):
                    fair_subgroup_sorted[k] = np.quantile(
                        original_subgroup_sorted, taus[k], method=interpolation
                    )
                y_fair += ps[group_name] * fair_subgroup_sorted

            for group_name, group_df in groups:
                index = np.floor(
                    np.linspace(0, n_calib, Ns[group_name], endpoint=False)
                ).astype("int")
                calib_sorted.loc[
                    calib_sorted["protected_col"] == group_name, "pred_col_fair"
                ] = y_fair[index]

            return calib_sorted.sort_index().pred_col_fair.values

        if len(z_unfairMat.shape) >= 2:
            z_fairMat = np.apply_along_axis(synchronized_z, -1, z_unfairMat)
            return torch.from_numpy(z_fairMat)
        return torch.from_numpy(synchronized_z(z_unfairMat))


class CL_CP_DP(CL_CP):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        super().__init__(dl_calib, model, n_samples, **kwargs)
        x_cal, y_cal = dl_calib.dataset[:]
        self.probs = torch.exp(
            self.model.predict(x_cal).log_prob(y_cal).detach()
        )

    def get_q(self, alpha):
        return find_optimal_cutoff_interval(
            self.calib_scores, self.probs, 1 - alpha
        )

    def is_in_region(self, x, y, alpha, cache=None):
        lower_q, upper_q = self.get_q(alpha)
        score = self.get_score(x, y, cache)
        return torch.logical_and(score >= lower_q, score <= upper_q)


class M_CP(ConformalizerBase):
    def __init__(
        self, dl_calib, model, n_samples=100, correction_factor=0, **kwargs
    ):
        self.n_samples = n_samples
        _, first_y = next(iter(dl_calib))
        self.d = first_y.shape[-1]
        self.correction_factor = correction_factor
        super().__init__(dl_calib, model, **kwargs)

    def get_region_bounds(self, x, alpha, cache=None):
        samples = get_samples(self.model, x, self.n_samples, cache, "samples")
        alpha /= 1 + self.correction_factor * (self.d - 1)
        return torch.quantile(
            samples,
            torch.tensor([alpha / 2, 1 - alpha / 2], device=x.device),
            dim=0,
        )

    def get_score(self, x, y, alpha, cache=None):
        ql, qh = self.get_region_bounds(x, alpha, cache)
        cqr_score = torch.maximum(ql - y, y - qh)
        return torch.max(cqr_score, dim=-1).values


class PCP(ConformalizerBaseAlphaInvariant):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)

    def get_score(self, x, y, cache=None):
        samples = get_samples(self.model, x, self.n_samples, cache, "samples2")
        return distance_to_closest_point(samples, y)


class HD_PCP(ConformalizerBase):
    def __init__(self, dl_calib, model, n_samples=100, **kwargs):
        self.n_samples = n_samples
        super().__init__(dl_calib, model, **kwargs)

    def get_score(self, x, y, alpha, cache=None):
        samples, log_probs = get_samples_and_log_probs(
            self.model, x, self.n_samples, cache, "samples2", "log_probs2"
        )
        k = torch.tensor((1 - alpha) * self.n_samples).int()
        indices = torch.argsort(log_probs, dim=0, descending=True)[:k]
        samples = samples[indices, torch.arange(x.shape[0])]
        return distance_to_closest_point(samples, y)