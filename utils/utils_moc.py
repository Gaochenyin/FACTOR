# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 11:25:28 2025

@author: chyga
"""
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
import pandas as pd

def get_samples(model, x, n_samples, cache={}, samples_key=None):
    """
    Helper function to get samples from the model or the cache if available.
    """
    samples = cache.get(samples_key)
    if samples is None:
        dist = model.predict(x)
        samples = dist.sample((n_samples,))
    else:
        samples = samples[:n_samples]
    return samples

def get_samples_and_log_probs(model, x, n_samples, cache={}, samples_key=None, log_probs_key=None):
    """
    Helper function to get samples with their log probabilities from the model or the cache if available.
    """
    samples = get_samples(model, x, n_samples, cache, samples_key)
    log_probs = cache.get(log_probs_key)
    if log_probs is None:
        dist = model.predict(x)
        log_probs = dist.log_prob(samples).detach()
    else:
        log_probs = log_probs[:n_samples]
    return samples, log_probs

def fast_empirical_cdf(a, b):
    """
    Returns the empirical CDF of a at b.
    The first dimension of a represents the samples of the CDF.
    a is a tensor of shape (s, n).
    b is a tensor of shape (..., n).
    This implementation is faster and requires less memory than naive_cdf.
    """
    assert a.dim() == 2 and b.dim() >= 1 and a.shape[-1] == b.shape[-1]
    b_shape = b.shape
    # We move n to the first dimension.
    # This will be useful because searchsorted has to be applied on the last dimension of a.
    a = a.movedim(-1, 0)
    b = b.movedim(-1, 0)

    if b.dim() == 1:
        # The naive implementation is faster for this case.
        cdf = (a <= b[:, None]).float().mean(dim=-1)
    else:
        a = torch.sort(a, dim=1)[0]
        # These operations are needed because the first N - 1 dimensions of a and b
        # have to be the same.
        view = (a.shape[0],) + (1,) * len(b.shape[1:-1]) + (a.shape[1],)
        repeat = (1,) + b.shape[1:-1] + (1,)
        a = a.view(*view).repeat(*repeat)
        cdf = torch.searchsorted(a, b.contiguous(), side='right') / a.shape[-1]
    
    # We move n back to the last dimension.
    cdf = cdf.movedim(0, -1)
    assert cdf.shape == b_shape
    return cdf

def latent_distance(z):
    """
    Returns the distance from the origin in the latent space.
    """
    if isinstance(z, torch.Tensor): # Latent space of MQF2
        return torch.linalg.norm(z, dim=-1)
    elif isinstance(z, list): # Latent space of Glow
        # print('test')
        s = []
        for zi in z:
            si = torch.linalg.norm(zi, dim=(-3, -2, -1))
            s.append(si)
        s = torch.stack(s, dim=-1)
        s = torch.max(s, dim=-1).values
        return s
    raise ValueError(f'Unsupported type: {type(z)}')
    
def distance_to_closest_point(points, y):
    """
    Returns the distance from `y` to the closest points in `points`.
    `points` is a tensor of shape (n_points, b, d).
    `y` is a tensor of shape (..., b, d), where the first dimensions are arbitrary 
    and will be evaluated for the same x.
    """
    n_points, b, d = points.shape
    *y_sample_shape, by, dy = y.shape
    assert (b, d) == (by, dy)
    y = y.unsqueeze(y.dim() - 2)
    norm = torch.linalg.norm(points - y, 2, dim=-1)
    assert norm.shape == tuple(y_sample_shape) + (n_points, b)
    distance = torch.min(norm, dim=-2).values
    assert distance.shape == tuple(y_sample_shape) + (b,)
    return distance

def pairwise_distance_to_closest_point(y):
    """
    Returns the distance from `y` to the closest points in `y`, except the point itself.
    `y` is a tensor of shape (n_points, ..., d).
    """
    n_points, *remaining, d = y.shape
    # Compute the pairwise distances
    norm = torch.linalg.norm(y.unsqueeze(0) - y.unsqueeze(1), 2, dim=-1)
    assert norm.shape == (n_points, n_points, *remaining)
    # Exclude the point itself
    mask = ~torch.eye(y.shape[0], dtype=bool)
    mask = mask[:, :, None].repeat(1, 1, *remaining)
    norm = norm[mask].reshape(n_points, n_points - 1, *remaining)
    # Compute the distance to the closest point
    distance = torch.min(norm, dim=-2).values
    assert distance.shape == (n_points, *remaining)
    return distance



# Plot the data conditional to X
def plot_data_conditional(X, Y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y[:, 0], Y[:, 1], c='b', marker='o')
    ax.set(xlabel='$X$', ylabel='$Y_1$', zlabel='$Y_2$')
    plt.show()
    
# Plot the data ignoring X
def plot_data_unconditional(Y):
    fig, ax = plt.subplots()
    ax.scatter(Y[:, 0], Y[:, 1], c='b', marker='o')
    ax.set(xlabel='$Y_1$', ylabel='$Y_2$')
    plt.show()
    
class CustomLogger(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'].item()
        self.train_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs.item()
        self.val_losses.append(loss)
        
def get_lightning_trainer(max_epochs = 100):
    # Might be better to do this in the future:
    # Rule of thumb:
    # - Validation set of size val_size = min(2500, 0.1 * n)
    #   - The obtained estimation of the loss is unbiased and is already precise with 2500 samples
    # - Test set of size test_size = 0.2 * n because this measures the metrics we are interested in
    #   - Having more samples is useful e.g. to measure conditional coverage
    # - Measure validation loss every 4 * val_size // batch_size steps
    #   - We can afford to measure validation during one fifth of the training
    # - Patience of 15

    ckpt = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=0,  # save k best models (determined by above metric)
        save_last=False,  # save model from last epoch
        verbose=False,
        # dirpath=str(rc.checkpoints_path),
        filename='epoch_{epoch:04d}',
        auto_insert_metric_name=False,
    )

    es = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=15,
        min_delta=0,
    )

    callbacks = [ckpt, es, CustomLogger()]

    accelerator = 'cpu'

    return Trainer(
        accelerator=accelerator,
        devices=1,
        min_epochs=1,
        max_epochs=max_epochs,
        # number of validation steps to execute at the beginning of the training
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        check_val_every_n_epoch=2,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=callbacks,
        logger=False,
    )
def plot_2D_contour_vs_1D_at_coverage(axis, x_value, conformalizer, alpha, 
                                      ylim, zlim, grid_side=50, **kwargs):
    device = conformalizer.model.device
    y1, y2 = torch.linspace(*ylim, grid_side, device=device), torch.linspace(*zlim, grid_side, device=device)
    Y1, Y2 = torch.meshgrid(y1, y2, indexing='ij')
    pos = torch.dstack((Y1, Y2))
    pos = pos[:, :, None, :]
    assert pos.shape == (y1.shape[0], y1.shape[0], 1, 2)
    x_pos = x_value # torch.tensor([[x_value.item()]], device=device)
    mask = conformalizer.is_in_region(x_pos, pos, alpha)
    mask = mask[:, :, 0]

    Y1, Y2, mask = Y1.cpu().numpy(), Y2.cpu().numpy(), mask.float().cpu().numpy()
    assert Y1.shape == Y2.shape == mask.shape

    fig2D, ax2D = plt.subplots()
    contour = ax2D.contour(Y1, Y2, mask, levels=[0], colors='r')
    plt.close(fig2D)
    
    if hasattr(contour, 'collections'):
        contour_paths = contour.collections[0].get_paths()
    else:
        contour_paths = contour.get_paths()
    if len(contour_paths) > 0:
        contour_path = contour_paths[0]
        for contour_points in contour_path.to_polygons():
            axis.plot(
                np.full_like(contour_points[:, 0], x_value[0, 1].item()), 
                contour_points[:, 0],  # y-coordinates
                contour_points[:, 1],  # z-coordinates
                zorder=5,
                **kwargs,
            )
            
def plot_contour_at_coverage_2D(axis, x_mat, conformalizer, method_name, 
                                alpha, ylim, zlim, color, grid_side=50, cache={}, **kwargs):
    device = conformalizer.model.device
    y1, y2 = torch.linspace(*ylim, grid_side, device=device), torch.linspace(*zlim, grid_side, device=device)
    Y1, Y2 = torch.meshgrid(y1, y2, indexing='ij')
    pos = torch.dstack((Y1, Y2))
    pos = pos[:, :, None, :]
    assert pos.shape == (y1.shape[0], y1.shape[0], 1, 2)
        
    def mask_contour(x_value):
        x_pos = x_value[None, :]
        # x_pos = torch.tensor([[x_value]], device=device)
        mask = conformalizer.is_in_region(x_pos, pos, alpha, cache=cache)    
        mask = mask[:, :, [0]]
        return mask.detach().numpy()
    maskMat = np.concatenate([mask_contour(x_value) for x_value in x_mat], axis = 2)
    mask = maskMat.mean(axis = 2)
    Y1, Y2 = Y1.cpu().numpy(), Y2.cpu().numpy()
    assert Y1.shape == Y2.shape == mask.shape

    # axis.contourf(Y1, Y2, np.logical_not(mask), levels=[0, 0.0001], colors=[color], alpha=0)

    fig2D, ax2D = plt.subplots()
    contour = ax2D.contour(Y1, Y2, mask, levels=[0], colors=[color])
    plt.close(fig2D)
    
    if hasattr(contour, 'collections'):
        contour_paths = contour.collections[0].get_paths()
    else:
        contour_paths = contour.get_paths()
    if len(contour_paths) > 0:
        contour_path = contour_paths[0]
        for i, contour_points in enumerate(contour_path.to_polygons()):
            axis.plot(
                contour_points[:, 0],  
                contour_points[:, 1],
                label=method_name if i == 0 else None,
                color=color,
                **kwargs,
            )
    # axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=6, frameon=False)
    # axis.set_xlabel('Y1')
    # axis.set_ylabel('Y2') 


def plot_contour_at_coverage_2D_discrete_groupk(axis, x_mat, conformalizer, method_name, 
                                         alpha, ylim, zlim, 
                                         color, marker, 
                                         k_idx, K,
                                         grid_side=50, cache={}, **kwargs):
    device = conformalizer.model.device
    # take the limit of y2 within y1 level
    y2 = torch.linspace(*zlim, grid_side, device=device)
    Y1, Y2 = torch.meshgrid(ylim, y2, indexing='ij')
    
    pos = torch.dstack((Y1, Y2))
    pos = pos[:, :, None, :]
    
    def mask_contour(x_value):
        x_pos = x_value[None, :]
        # x_pos = torch.tensor([[x_value]], device=device)
        mask = conformalizer.is_in_region(x_pos, pos, alpha, cache=cache) 
        mask = mask[:, :, [0]]
        return mask.detach().numpy()
    maskMat = np.concatenate([mask_contour(x_value) for x_value in x_mat], axis = 2)
    
    mask = maskMat.mean(axis = 2)

    # prepare the df for forestplot
    y2_intervals = [Y2[y1_idx, mask[y1_idx, :] == 1] for y1_idx, y1 in enumerate(ylim)]
    data = []
    for group, iv in zip(ylim, y2_intervals):
        if iv.numel() == 0:
            data.append(dict(
                y1_level=group.item(),
                group = f'Group {k_idx+1}',
                y2_min=None,
                y2_max=None,
                y2_mean=None,
                n_points=iv.numel()
            ))
        else:
            min_y2 = iv.min().item()
            max_y2 = iv.max().item()
            mean_y2 = iv.mean().item()
            data.append(dict(
                y1_level=group.item(),
                group = f'Group {k_idx+1}',
                y2_min=min_y2,
                y2_max=max_y2,
                y2_mean=mean_y2,
                n_points=iv.numel()
            ))
    df_plot = pd.DataFrame(data)
    
    axis.errorbar(
        df_plot['y2_mean'], df_plot['y1_level'] + (k_idx - K//2) * 0.7/K,
        xerr=[df_plot['y2_mean'] - df_plot['y2_min'], df_plot['y2_max'] - df_plot['y2_mean']],
        fmt=marker,
        color=color,
        label=f'Group {k_idx+1}',
        capsize=3,
        markersize=3
    )
    axis.set_ylim(ylim.min()*0.9, ylim.max()*1.1)
    axis.set_yticks(ylim) 
    axis.set_xlabel('Y1')
    axis.set_ylabel('Y2')  
           
def compute_cum_region_size(conformalizer, model, alpha, x, n_samples, cache_region_size={}, cache_test={}):
    samples = cache_region_size.get('samples')
    if samples is None:
        dist = model.predict(x)
        samples = dist.sample((n_samples,))
    log_probs = cache_region_size.get('log_probs')
    if log_probs is None:
        log_probs = dist.log_prob(samples).detach()
        
    # conformalizer_LCP.is_in_region(x, samples, alpha, cache_test)
    # for s in conformalizer_LCP.get_score(x, samples):
    #     print(s)
    # conformalizer.is_in_region(x, samples, alpha, cache_test)
    
    # term1 = conformalizer.is_in_region(x, samples, alpha, cache_test).float() 
    terms = conformalizer.is_in_region(x, samples, alpha, cache_test).float() / np.clip(torch.exp(log_probs),
                                                                                        1e-13, 1)
    
    # terms = conformalizer.is_in_region(x, samples, alpha, cache_test).float() / torch.exp(log_probs)
    cum_region_size = torch.cumsum(terms, dim=0) / torch.arange(1, n_samples + 1, device=x.device)[:, None]
    # terms_LCP = conformalizer_LCP.is_in_region(x, samples, alpha, cache_test).float()  / torch.exp(log_probs)
    
    # l1 = torch.cumsum(terms_LCP, dim=0) / torch.arange(1, n_samples + 1, device=x.device)[:, None]
    # l2 = torch.cumsum(terms, dim=0) / torch.arange(1, n_samples + 1, device=x.device)[:, None]
    return cum_region_size


def compute_coverage_indicator(conformalizer, alpha, x, y, cache={}):
    return conformalizer.is_in_region(x, y, alpha, cache).float()
# def compute_cum_region_size(conformalizer, model, alpha, x, n_samples=100, cache_region_size={}, cache_test={}):
#     # Compute log region size for each instance in the batch and each sample size
#     # between 0 and n_samples
#     samples = cache_region_size.get('samples')
#     if samples is None:
#         dist = model.predict(x)
#         samples = dist.sample((n_samples,))
#     log_probs = cache_region_size.get('log_probs')
#     if log_probs is None:
#         log_probs = dist.log_prob(samples).detach()

#     n_samples = samples.shape[0]
#     assert samples.shape[:-1] == log_probs.shape
    
#     # Use log-sum-exp trick
#     max_logpdf = torch.max(log_probs, dim=0).values
#     log_terms = torch.where(
#         conformalizer.is_in_region(x, samples, alpha, cache_test),
#         -log_probs + max_logpdf, 
#         -torch.inf
#     )
    
#     log_cumsum = torch.logcumsumexp(log_terms, dim=0)
#     log_means = log_cumsum - torch.log(torch.arange(1, n_samples + 1, device=x.device))[:, None]
#     log_region_size = log_means - max_logpdf
#     assert log_region_size.shape == (n_samples, x.shape[0],)
#     return np.exp(log_region_size)

def compute_region_size(*args, **kwargs):
    return compute_cum_region_size(*args, **kwargs)[-1]

def savefig(path, fig=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    fig.savefig(
        path,
        bbox_extra_artists=fig.legends or None,
        bbox_inches='tight',
        **kwargs,
    )
    plt.close(fig)

def scatter_2D_vs_1D(ax, x, y):
    ax.scatter(x[:, 0], y[:, 0], y[:, 1], color='gray', edgecolors='none', alpha=0.1)
    

def nll(model, x, y):
    dist = model.predict(x)
    return -dist.log_prob(y).detach()