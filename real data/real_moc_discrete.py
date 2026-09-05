# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 12:32:37 2025

@author: chyga
"""

import torch
from pathlib import Path
import numpy as np
import pandas as pd
# model module
from mqf2.lightning_module import MQF2LightningModule
# trainer scheme
from utils_moc import (
get_lightning_trainer,
plot_contour_at_coverage_2D_discrete_groupk,
compute_coverage_indicator,
compute_region_size,
savefig
)
from utils_data import (
    RealDataModule)
from utils_conformalizer import (
    M_CP, L_CP,
    CL_CP_DP,
    C_HDR, 
    DR_CP, PCP, HD_PCP,
    CL_CP, L_CP_DP,
    STDQR)
        
import matplotlib.pyplot as plt 
import seaborn as sns
from timeit import default_timer as timer
import matplotlib.gridspec as gridspec

# the name of dataset in the folder data
data_name = 'births1' # 'house', 'wage'
data_name ='house'
data_name = 'air'
for data_name in [# continuous 
                  'births1', 'energy', 'air', 'households', 
                  # discrete 
                  'wage', 'house']:
                  # ,  'house']:
    # load data as data_name.csv
    print(data_name, '\n')
    datamodule = RealDataModule(data_dir = 'data',
     data_name = data_name,
     train_val_calib_test_split_ratio = (0.4, 0.2, 0.4, 0.0))
    # path to save the model
    model_kwargs = dict()
    model_kwargs['input_dim'] = datamodule.input_dim
    model_kwargs['output_dim'] = datamodule.output_dim
    model_path = Path(f'checkpoints/best_MQF2_Real{data_name}.pth')
    # initialize the model
    model = MQF2LightningModule(**model_kwargs)
    # initialize the trainer
    trainer = get_lightning_trainer(max_epochs = 100)
    # fit the model on data use trainer
    trainer.fit(model = model, datamodule = datamodule)
    # save the model at checkpoint
    trainer.save_checkpoint(model_path)
    
    
    
    # load the calibration data
    calib_data = datamodule.calib_dataloader()
    # M-CP (Marginal Conformal Prediction)
    conformalizer_MCP = M_CP(dl_calib = calib_data, model = model,
                               n_samples = 500)
    # C-HDR (Highest Region Conformal Prediction)
    conformalizer_CHDR = C_HDR(dl_calib = calib_data, model = model,
                               n_samples = 500)
    
    # L-CP (Latent Conformal Prediction)
    conformalizer_LCP = L_CP(dl_calib = calib_data, model = model,
                               n_samples = 500)
    
    # CL-CP (Conditional latent Conformal Prediction)
    # the first column of X is the protected features
    # conformalizer_CLCP = CL_CP(dl_calib = calib_data, model = model,
    #                            n_samples = 500)
    
    # proposed
    conformalizer_CLCP = CL_CP(dl_calib = calib_data, model = model,
                               n_samples = 500)
    conformalizer_CLCP_DP = CL_CP_DP(dl_calib = calib_data, model = model,
                               n_samples = 500)
    conformalizer_LCP_DP = L_CP_DP(dl_calib = calib_data, model = model,
                               n_samples = 500)
    ## a quick sanity check
    plt.hist(conformalizer_CLCP_DP.calib_scores,
              bins=30, density=True, alpha=0.6, 
              label='Histogram on calibration set')
    # Generate MC samples for L2 norm of 3D Uniform[0,1] vectors
    n_MC = 5000  # You can increase this for a smoother curve
    _, y_cal = calib_data.dataset[:]
    X = np.random.uniform(0, 1, size=(n_MC, y_cal.shape[1]+1))  # n_MC vectors in 3D
    l2norm = np.linalg.norm(X, axis=1)**2
    
    # Plot the MC density as a smooth curve
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(l2norm)
    r_vals = np.linspace(0, y_cal.shape[1]+1, 500)
    plt.plot(r_vals, kde(r_vals), color='red', label='Theoretical Distribution')
    plt.legend()
    # plt.title(f'{data_set}')
    # plt.show()
    savefig(f'check{data_name}.pdf')
    
    ## additional methods for rebuttal
    # STDQR
    conformalizer_STDQR = STDQR(dl_calib = calib_data, model = model,
                                n_samples = 500)
    
    # DR-CP
    # conformalizer_DRCP = DR_CP(dl_calib = calib_data, model = model)
    
    # PCP
    conformalizer_PCP = PCP(dl_calib = calib_data, model = model,
                                n_samples = 500)
    
    # HD-PCP: extension of PCP
    conformalizer_HDPCP = HD_PCP(dl_calib = calib_data, model = model, 
                                 n_samples = 500)
     
    
    
    # evaluation
    conformalizers = [conformalizer_MCP, 
                      conformalizer_CHDR, conformalizer_LCP,
                      conformalizer_STDQR,
                      conformalizer_PCP, 
                      # conformalizer_DRCP, 
                      conformalizer_HDPCP,
                      conformalizer_CLCP,
                      conformalizer_LCP_DP,
                      conformalizer_CLCP_DP]
    

    methods = ['MCP', 'HDR', 'L-CP', 
                'ST-DQR', 'PCP', 
                # 'DR-CP',
                'HD-CP',
                'FACTOR (w/o OptimCutoff)',
                'FACTOR (w/o Fairness)',
                'FACTOR']
    
    x_cal, y_cal = datamodule.data_calib[:]
    
    # compute the average size of prediction sets and the KS distances
    res = []
    # generate samples for computing the volume size
    dist = model.predict(x_cal)
    samples = dist.sample((100,))
    log_probs = dist.log_prob(samples).detach()
    cache_region_size = {'samples': samples,
                         'log_probs': log_probs}
    
    for method_idx, method in enumerate(methods):
        protected_col = x_cal[:, 0].numpy()
        t = timer()
        coverage = compute_coverage_indicator(conformalizers[method_idx], 
                                              0.05, x_cal, y_cal)
        # prediction the volume on each x_cal
        volume = compute_region_size(conformalizers[method_idx], model, 
                                      0.05, x_cal, 
                                      # cache_region_size = cache_region_size,
                                      n_samples=500)
        elapsed_time = timer() - t
        cp_groups = [(coverage[protected_col == group]).mean().item() 
         for group in np.unique(protected_col)]
        vl_groups = [(volume[protected_col == group]).mean().item() 
         for group in np.unique(protected_col)]
        
        res.append({
            # 'n': n,
            'data_set': data_name,
            'method': method,
            'coverage': cp_groups,
            'volume': vl_groups,
            'elapsed_time': elapsed_time
        })
    
    # organize the table
    methods = [item['method'] for item in res]
    coverages = np.array([item['coverage'] for item in res])
    volumes = np.array([item['volume'] for item in res])
        
    res_df = pd.DataFrame([{'method': r['method'], 
                          'data_name': data_name,
                          'AvgCoverage': np.mean(r['coverage']),
                'KS': np.max(r['coverage']) - np.min(r['coverage']), 
                'p-rule':np.min(r['coverage'])/ np.max(r['coverage']),
                'CovStd': np.std(r['coverage']),
      # 'KS': np.abs(np.array(r['coverage']) - 0.95).sum(), 
                'AvgVol': (np.mean(r['volume'])).item(),
                'AvgTime': r['elapsed_time']} for r in res]) 

    # save the results
    res_df.to_csv(f'{data_name}_rst.csv', index = False)



# summarize the results for the benchmarks
rst_all = pd.concat([pd.read_csv(f'{data_name}_rst.csv') for
           data_name in [
               # continuous
                'births1', 'energy', 'air', 'households',
                # discrete
                'wage',  'house']])

methods = ['MCP', 'HDR', 'L-CP', 
            'ST-DQR', 'PCP', 
            # 'DR-CP',
            'HD-CP',
            'FACTOR']

# organize the results
rst_plot1 = rst_all.query(f'method in {methods}')
palette = sns.color_palette('Paired')
# visual plot for all results
fig, ax_bottom = plt.subplots(1, 3, figsize=(12, 3))
rst_plot1['log_volume'] = np.log(rst_plot1['AvgVol'])
rst_plot1['log_AvgTime'] = np.log(rst_plot1['AvgTime'])
sns.barplot(x='data_name', y='log_volume', data=rst_plot1, 
            hue = 'method', ax=ax_bottom[0], palette=palette)
ax_bottom[0].set_ylabel("log(Average region size)")
ax_bottom[0].set_xlabel(" ")
sns.barplot(x='data_name', y='KS', data=rst_plot1, 
            hue = 'method', ax=ax_bottom[1], palette=palette)
ax_bottom[1].set_ylabel("Empirical KS distance")
ax_bottom[1].set_xlabel(" ")
sns.barplot(x='data_name', y='AvgTime', data=rst_plot1, 
            hue = 'method', ax=ax_bottom[2], palette=palette)
ax_bottom[2].set_ylabel("Elapsed time")
ax_bottom[2].set_xlabel(" ")
# Only add the legend to the first axis, and position it outside
handles, labels = ax_bottom[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.13), ncol=len(labels), fontsize=12, frameon=False)
# Remove legends from other axes to avoid duplicates
for ax in ax_bottom:
    ax.get_legend().remove()
    plt.setp(ax.get_xticklabels(), rotation=30, ha='center')
plt.tight_layout()  # adjust right space for legend
savefig('real_data.pdf')

# Compute mean and std over datasets
rst_plot1.groupby('method').agg(
    mean_log_vol=('log_volume', 'mean'),
    mean_KS=('KS', 'mean'),
    mean_time=('AvgTime', 'mean'),
).reset_index()

# plot the p-rule for the real-data and simulated data
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
import pickle
n = 5000
def load_sim_res(data_set):
    # Load the dictionary
    with open(f'{data_set}_n{n}_res.pkl', 'rb') as f:
        res = pickle.load(f)
    return pd.DataFrame([{'method': r['method'], 
                          'data_set': data_set,
               'CovStd': np.std(r), 
               'KS': np.max(r['coverage']) - np.min(r['coverage']), 
                'AvgVol': (np.mean(r['volume'])).item()} for r in res])
# readin and plot
sim_data = pd.concat([pd.read_csv(f'{data_name}_n{n}_rst.csv') for
           data_name in [
               # continuous
               'mvnormal_2_0', 'mvnormal_2_0.5', 'mvnormal_2_0.8'
               ]])
sim_data['data_set'] = sim_data['data_set'].map({'mvnormal_2_0': 'rho = 0',
                        'mvnormal_2_0.5': 'rho = 0.5',
                        'mvnormal_2_0.8': 'rho = 0.8'})
sim_data = sim_data.query(f'method in {methods}')
real_data = rst_plot1
sns.barplot(x='data_set', y='p-rule', data=sim_data, 
            hue = 'method', ax=axes[0], palette=palette)
axes[0].set_ylabel("P-rule for simulated data")
axes[0].set_xlabel(" ")
axes[0].axhline(y=1, color='red', linestyle='--')
sns.barplot(x='data_name', y='p-rule', data=real_data, 
            hue = 'method', ax=axes[1], palette=palette)
axes[1].set_ylabel("P-rule for real-world data")
axes[1].set_xlabel(" ")
axes[1].axhline(y=1, color='red', linestyle='--')
# Only add the legend to the first axis, and position it outside
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.13), ncol=len(labels), fontsize=12, frameon=False)
# Remove legends from other axes to avoid duplicates
for ax in axes:
    ax.get_legend().remove()
    plt.setp(ax.get_xticklabels(), rotation=30, ha='center')
plt.tight_layout()  # adjust right space for legend
savefig('p_rule_fairness.pdf')

# ablation study
methods = ['L-CP',
'FACTOR (w/o OptimCutoff)',
'FACTOR (w/o Fairness)',
'FACTOR']

# organize the results
rst_plot2 = rst_all.query(f'method in {methods}')
# Rename L‑CP inside the dataframe
rst_plot2['method'] = rst_plot2['method'].replace(
    {'L-CP': 'FACTOR (w/o OptimCutoff & Fairness)'}
)
palette = sns.color_palette('Paired')
# visual plot for all results
fig, ax_bottom = plt.subplots(1, 2, figsize=(8, 3))
rst_plot2['log_volume'] = np.log(rst_plot2['AvgVol'])
rst_plot2['log_AvgTime'] = np.log(rst_plot2['AvgTime'])
sns.barplot(x='data_name', y='log_volume', data=rst_plot2, 
            hue = 'method', ax=ax_bottom[0], palette=palette)
ax_bottom[0].set_ylabel("log(Average region size)")
ax_bottom[0].set_xlabel(" ")
sns.barplot(x='data_name', y='KS', data=rst_plot2, 
            hue = 'method', ax=ax_bottom[1], palette=palette)
ax_bottom[1].set_ylabel("Empirical KS distance")
ax_bottom[1].set_xlabel(" ")
# Only add the legend to the first axis, and position it outside
handles, labels = ax_bottom[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=12, frameon=False)
# Remove legends from other axes to avoid duplicates
for ax in ax_bottom:
    ax.get_legend().remove()
    plt.setp(ax.get_xticklabels(), rotation=30, ha='center')
plt.tight_layout()  # adjust right space for legend
savefig('ablation_study.pdf')


# Compute mean and std over datasets
rst_plot2.groupby('method').agg(
    mean_log_vol=('log_volume', 'mean'),
    std_log_vol=('log_volume', 'std'),
    mean_KS=('KS', 'mean'),
    std_KS=('KS', 'std')
).reset_index()



# example plot
res_df = pd.read_csv(f'{data_name}_rst.csv')
res_df = res_df.query("method in  ['MCP', 'HDR', 'L-CP',  'ST-DQR', 'PCP',  'HD-CP', 'FACTOR']")
# res_df['method'] = res_df['method'].map({'CL-HDR-DP': 'FACTOR',
#                        'MCP': 'MCP',
#                        'HDR': 'HDR',
#                        'L-HDR': 'L-CP',
#                        'CL-HDR': 'CL-HDR'})


# methods = ['MCP', 'HDR', 'L-CP', 'FACTOR']
# create the parent figure
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.3)
## first figure
top_gs = gridspec.GridSpecFromSubplotSpec(1, len(methods), subplot_spec=gs[0],
                                          wspace=0)
axs_top = [fig.add_subplot(top_gs[0, i]) for i in range(len(methods))]

palette = sns.color_palette('Paired')
style_dict = {
    'Oracle': (palette[1], '-'),
    'MCP': (palette[5], '-'),
    'HDR': ('tab:orange', '-'),
    'L-HDR': (palette[3], '-'),
    'CL-HDR': (palette[4], '-'),
    'FACTOR': (palette[2], '-')
}
style_dict_Group = {
    0: ('g', 'D'),
    1: ('r', 'o'),
    2: ('c', 's')
    }    
# obtain the level of protected groups
groups = x_cal[:, 0].unique()
K = len(groups)

# obtain the level of Y1
y1_level = y_cal[:, 0].unique()

# input
for method_idx, method_name in enumerate(methods):
    
    for k_idx, k in enumerate(groups.numpy()):
        
        color, marker = style_dict_Group[k_idx]
        # take the mean of X within each group for evaluation
        x_eval = torch.cat((groups[[k_idx]], 
                        x_cal[x_cal[:, 0]==groups[k_idx], 1:].mean(axis = 0)), dim = 0)[None, :]
        y2_lim = -1e6, y_cal[:, 1].max().item()
        
        # begin the plot
        plot_contour_at_coverage_2D_discrete_groupk(axs_top[method_idx],
                                                    x_eval, conformalizers[method_idx],
                                                    method_name, 
                                                 0.05, y1_level, y2_lim, 
                                                 color, marker, 
                                                 k_idx, K)
        
        avgCoverage = res_df.loc[res_df.method == method_name].AvgCoverage.round(2).item()
        
        axs_top[method_idx].set_title(f'{method_name} (AvgCov: {avgCoverage})', y=0.97)
        axs_top[method_idx].legend(loc='upper left', fontsize=8)
        axs_top[method_idx].set_xlabel('Y1: Price') 
        axs_top[method_idx].set_ylabel('Y2: Grade') 
        axs_top[method_idx].set_xticks([0, 0.5e6, 1e6, 1.5e6])  
        
for axis in axs_top:
    axis.label_outer()    
    
# bottom plot
bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 3, 
                                             subplot_spec=gs[1],
                                             wspace=0.3)
ax_bottom = [fig.add_subplot(bottom_gs[0, i]) for i in range(3)]
# with open(f'{data_set}_n{n}_res.pkl', 'rb') as f:
#      res = pickle.load(f)

res_df['log_volume'] = np.log(res_df.AvgVol)

# fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax_bottom[0].bar(x='method', height='log_volume', data=res_df,
         color=palette)
ax_bottom[0].set_ylabel("log(Average region size)")
ax_bottom[0].set_ylim([17.5, 31])

ax_bottom[1].bar(x='method', height='KS', data=res_df,
          color=palette)
ax_bottom[1].set_ylabel("Empirical KS distance")

ax_bottom[2].bar(x='method', height='AvgTime', data=res_df,
          color=palette)
ax_bottom[2].set_ylabel("Elapsed time")
# fig.suptitle(data_name, fontsize=16)
# save the figure in pdf
savefig(f'{data_name}_2d_byGroup_demo.pdf')




# Paste your data into a dictionary (or load via CSV if you prefer)
res_df = pd.read_csv(f'air_rst.csv')

res_df = res_df.query(f'method in {methods}')
# Compute log(AvgVol) for plotting as 'log_volume'
res_df['log_volume'] = np.log(res_df['AvgVol'])

# Create figure and axes
fig, ax_bottom = plt.subplots(1, 3, figsize=(16, 3), constrained_layout=True)

# Bar plot 1: log_volume
ax_bottom[0].bar(res_df['method'], res_df['log_volume'], color=palette)
ax_bottom[0].set_ylabel("log(Average region size)")

# Bar plot 2: KS
ax_bottom[1].bar(res_df['method'], res_df['KS'], color=palette)
ax_bottom[1].set_ylabel("Empirical KS distance")

# Bar plot 3: AvgTime
ax_bottom[2].bar(res_df['method'], res_df['AvgTime'], color=palette)
ax_bottom[2].set_ylabel("Elapsed time")
# plt.show()
savefig('real_data_example_p=6.pdf')