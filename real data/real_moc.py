# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 12:32:37 2025

@author: chyga
"""
from pathlib import Path
from timeit import default_timer as timer
import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from mqf2.lightning_module import MQF2LightningModule
from utils_conformalizer import C_HDR, CL_CP, L_CP, M_CP
from utils_data import RealDataModule
from utils_moc import (
    compute_coverage_indicator,
    compute_region_size,
    get_lightning_trainer,
    plot_contour_at_coverage_2D,
    savefig,
)

# Configuration
data_name = "households"

# Load real data
datamodule = RealDataModule(
    data_dir="data",
    data_name=data_name,
    train_val_calib_test_split_ratio=(0.4, 0.2, 0.4, 0.0),
)

# Initialize Model
model_kwargs = {
    "input_dim": datamodule.input_dim,
    "output_dim": datamodule.output_dim,
}
model_path = Path(f"checkpoints/best_MQF2_Real{data_name}.pth")
model = MQF2LightningModule(**model_kwargs)

# Train and Save Model
trainer = get_lightning_trainer(max_epochs=500)
trainer.fit(model=model, datamodule=datamodule)
trainer.save_checkpoint(model_path)

# Load Calibration Data
calib_data = datamodule.calib_dataloader()

# Initialize Conformalizers
conformalizer_MCP = M_CP(dl_calib=calib_data, model=model, n_samples=500)
conformalizer_CHDR = C_HDR(dl_calib=calib_data, model=model, n_samples=500)
conformalizer_LCP = L_CP(dl_calib=calib_data, model=model, n_samples=500)
conformalizer_CLCP = CL_CP(dl_calib=calib_data, model=model, n_samples=500)

conformalizers = [
    conformalizer_MCP,
    conformalizer_CHDR,
    conformalizer_LCP,
    conformalizer_CLCP,
]
methods = ["MCP", "HDR", "L-HDR", "CL-HDR"]
x_cal, y_cal = datamodule.data_calib[:]

# Evaluate Metrics
res = []
for method_idx, method in enumerate(methods):
    protected_col = x_cal[:, 0].numpy()
    t = timer()
    coverage = compute_coverage_indicator(
        conformalizers[method_idx], 0.05, x_cal, y_cal
    )
    volume = compute_region_size(
        conformalizers[method_idx], model, 0.05, x_cal, n_samples=100
    )
    elapsed_time = timer() - t

    cp_groups = [
        (coverage[protected_col == group]).mean().item()
        for group in np.unique(protected_col)
    ]
    vl_groups = [
        (volume[protected_col == group]).mean().item()
        for group in np.unique(protected_col)
    ]

    res.append(
        {
            "data_set": data_name,
            "method": method,
            "coverage": cp_groups,
            "volume": vl_groups,
            "elapsed_time": elapsed_time,
        }
    )

# Save Results Pickle
with open(f"{data_name}_res.pkl", "wb") as f:
    pickle.dump(res, f)

# Re-load and Organize Table
with open(f"{data_name}_res.pkl", "rb") as f:
    res = pickle.load(f)

methods = [item["method"] for item in res]
coverages = np.array([item["coverage"] for item in res])
volumes = np.array([item["volume"] for item in res])

res_df = pd.DataFrame(
    [
        {
            "method": r["method"],
            "data_name": data_name,
            "AvgCoverage": np.mean(r["coverage"]),
            "KS": np.max(r["coverage"]) - np.min(r["coverage"]),
            "AvgVol": (np.mean(r["volume"])).item(),
            "AvgTime": r["elapsed_time"],
        }
        for r in res
    ]
)
res_df["log_volume"] = np.log(res_df.AvgVol)

# 2D by Group Demo Plotting
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.3)

top_gs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0)
axs_top = [fig.add_subplot(top_gs[0, i]) for i in range(4)]

palette = sns.color_palette("Paired")
style_dict_Group = {
    0: ("b", "-"),
    1: ("g", "-"),
    2: ("r", "-"),
    3: ("c", "-"),
}
groups = x_cal[:, 0].unique()
K = len(groups)

ylim = -20000, 40000
zlim = -10000, 60000

for k_idx in range(K):
    print(f"Running Group {k_idx + 1} ...")
    for method_idx, method_name in enumerate(methods):
        color, linestyle = style_dict_Group[k_idx]
        x_eval = torch.cat(
            (
                groups[[k_idx]],
                x_cal[x_cal[:, 0] == groups[k_idx], 1:].mean(axis=0),
            ),
            dim=0,
        )[None, :]

        plot_contour_at_coverage_2D(
            axs_top[method_idx],
            x_eval,
            conformalizers[method_idx],
            f"Group {k_idx + 1}",
            0.05,
            ylim,
            zlim,
            color=color,
            linestyle=linestyle,
        )

        avgCoverage = (
            res_df.loc[res_df.method == method_name].AvgCoverage.round(2).item()
        )
        axs_top[method_idx].set_title(
            f"{method_name} (AvgCov: {avgCoverage})", y=0.97
        )
        axs_top[method_idx].legend(loc="upper left", fontsize=8)
        axs_top[method_idx].set_xticks([-5000, 0, 5000, 10000, 15000])
        axs_top[method_idx].set_yticks(np.arange(zlim[0], zlim[1], 18000))

        axs_top[method_idx].set_xlabel("Food")
        axs_top[method_idx].set_ylabel("House")

for axis in axs_top:
    axis.label_outer()

# Bottom Summary Plots
bottom_gs = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=gs[1], wspace=0.3
)
ax_bottom = [fig.add_subplot(bottom_gs[0, i]) for i in range(3)]

ax_bottom[0].bar(x="method", height="log_volume", data=res_df, color=palette)
ax_bottom[0].set_ylabel("log(Average region size)")
ax_bottom[0].set_ylim([19.75, 20.5])

ax_bottom[1].bar(x="method", height="KS", data=res_df, color=palette)
ax_bottom[1].set_ylabel("Empirical KS distance")

ax_bottom[2].bar(x="method", height="AvgTime", data=res_df, color=palette)
ax_bottom[2].set_ylabel("Elapsed time")

fig.suptitle(data_name, fontsize=16)
savefig(f"{data_name}_2d_byGroup_demo.pdf")