# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:54:58 2025

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
from scipy.stats import gaussian_kde
import torch

from mqf2.lightning_module import MQF2LightningModule
from utils_conformalizer import (
    C_HDR,
    CL_CP,
    CL_CP_DP,
    HD_PCP,
    L_CP,
    L_CP_DP,
    M_CP,
    PCP,
    STDQR,
)
from utils_data import DefaultTrainer, OracleModel, ToyDataModule
from utils_moc import (
    compute_coverage_indicator,
    compute_region_size,
    get_lightning_trainer,
    plot_contour_at_coverage_2D,
    plot_2D_contour_vs_1D_at_coverage,
    savefig,
    scatter_2D_vs_1D,
)

# Configuration
n = 5000
data_sets = ["mvnormal_2_0", "mvnormal_2_0.5", "mvnormal_2_0.8"]

for data_set in data_sets:
    print(f"Running {data_set} ...")
    datamodule = ToyDataModule(
        size=n,
        data_set=data_set,
        train_val_calib_test_split_ratio=(0.4, 0.2, 0.4, 0.0),
    )

    # Train the MQF2 (Multivariate Quantile Function Forecaster)
    model_kwargs = {
        "input_dim": datamodule.input_dim,
        "output_dim": datamodule.output_dim,
        "is_energy_score": False,
    }
    model_path = Path(f"checkpoints/best_MQF2_Toy{data_set}{n}.pth")
    model = MQF2LightningModule(**model_kwargs)
    trainer = get_lightning_trainer(max_epochs=500)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.save_checkpoint(model_path)

    # Train Oracle Model
    oracle_model = OracleModel()
    DefaultTrainer().fit(model=oracle_model, datamodule=datamodule)

    # Load calibration data
    calib_data = datamodule.calib_dataloader()
    x_cal, y_cal = datamodule.data_calib[:]
    K = x_cal[:, 0].unique()

    # Initialize Conformalizers
    conformalizer_Oracle = C_HDR(
        dl_calib=calib_data, model=oracle_model, n_samples=500
    )
    conformalizer_MCP = M_CP(
        dl_calib=calib_data, model=model, n_samples=500
    )
    conformalizer_CHDR = C_HDR(
        dl_calib=calib_data, model=model, n_samples=500
    )
    conformalizer_LCP = L_CP(
        dl_calib=calib_data, model=model, n_samples=500
    )
    conformalizer_CLCP = CL_CP(
        dl_calib=calib_data, model=model, n_samples=500
    )
    conformalizer_LCP_DP = L_CP_DP(
        dl_calib=calib_data, model=model, n_samples=500
    )
    conformalizer_CLCP_DP = CL_CP_DP(
        dl_calib=calib_data, model=model, n_samples=5000
    )
    conformalizer_STDQR = STDQR(
        dl_calib=calib_data, model=model, n_samples=500
    )
    conformalizer_PCP = PCP(
        dl_calib=calib_data, model=model, n_samples=500
    )
    conformalizer_HDPCP = HD_PCP(
        dl_calib=calib_data, model=model, n_samples=500
    )

    # Sanity Check Plot
    plt.figure()
    plt.hist(
        conformalizer_CLCP_DP.calib_scores,
        bins=30,
        density=True,
        alpha=0.6,
        label="Histogram on calibration set",
    )

    # MC Simulation
    n_MC = 1000
    X = np.random.uniform(0, 1, size=(n_MC, 3))
    l2norm = np.linalg.norm(X, axis=1) ** 2

    kde = gaussian_kde(l2norm)
    r_vals = np.linspace(0, 3, 100)
    plt.plot(r_vals, kde(r_vals), color="red", label="Theoretical Distribution")
    plt.legend()
    savefig(f"check_{data_set}.pdf")

    # Plot specifications
    eps = 1e-3
    x_test = torch.linspace(x_cal.min() + eps, x_cal.max() - eps, 5)
    x_test = x_test.to("cpu")[:, None]

    def get_lim(x):
        lim = x.min(), x.max()
        diff = lim[1] - lim[0]
        return lim[0] - 0.05 * diff, lim[1] + 0.05 * diff

    ylim = get_lim(y_cal[:, 0])
    zlim = get_lim(y_cal[:, 1])
    levels = [0.2, 0.5, 0.9]
    colors = ("black", "tab:green", "#FFD700", "#d62728")

    methods = [
        "MCP",
        "HDR",
        "L-CP",
        "ST-DQR",
        "PCP",
        "HD-CP",
        "FACTOR (w/o OptimCutoff)",
        "FACTOR (w/o Fairness)",
        "FACTOR",
    ]

    conformalizer_dict = {
        "Oracle": conformalizer_Oracle,
        "MCP": conformalizer_MCP,
        "HDR": conformalizer_CHDR,
        "L-CP": conformalizer_LCP,
        "ST-DQR": conformalizer_STDQR,
        "PCP": conformalizer_PCP,
        "HD-CP": conformalizer_HDPCP,
        "FACTOR (w/o OptimCutoff)": conformalizer_CLCP,
        "FACTOR (w/o Fairness)": conformalizer_LCP_DP,
        "FACTOR": conformalizer_CLCP_DP,
    }

    # Evaluate Coverage & Volume
    res = []
    dist = model.predict(x_cal)
    samples = dist.sample((100,))
    log_probs = dist.log_prob(samples).detach()

    for method_idx, method in enumerate(methods):
        protected_col = x_cal[:, 0].numpy()

        t = timer()
        coverage = compute_coverage_indicator(
            conformalizer_dict[method], 0.05, x_cal, y_cal
        )
        volume = compute_region_size(
            conformalizer_dict[method], model, 0.05, x_cal, n_samples=100
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
                "n": n,
                "data_set": data_set,
                "method": method,
                "coverage": cp_groups,
                "volume": vl_groups,
                "elapsed_time": elapsed_time,
            }
        )

    res_df = pd.DataFrame(
        [
            {
                "method": r["method"],
                "data_set": data_set,
                "AvgCoverage": np.mean(r["coverage"]),
                "KS": np.max(r["coverage"]) - np.min(r["coverage"]),
                "p-rule": np.min(r["coverage"]) / np.max(r["coverage"]),
                "CovStd": np.std(r["coverage"]),
                "AvgVol": (np.mean(r["volume"])).item(),
                "AvgTime": r["elapsed_time"],
            }
            for r in res
        ]
    )
    res_df.to_csv(f"{data_set}_n{n}_rst.csv", index=False)

    # 2D Group Demo Visualization
    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.3)

    top_gs = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=gs[0], wspace=0)
    axs_top = [fig.add_subplot(top_gs[0, i]) for i in range(7)]
    palette = sns.color_palette("Paired")

    style_dict_Group = {
        0: ("g", "-"),
        1: ("r", "-"),
        2: ("c", "-"),
        3: ("c", "-"),
    }

    for k_idx, k in enumerate(K):
        print(f"Running Group {k_idx + 1} ...")
        for method_idx, method_name in enumerate(
            ["MCP", "HDR", "L-CP", "ST-DQR", "PCP", "HD-CP", "FACTOR"]
        ):
            color, linestyle = style_dict_Group[k_idx]
            x_eval = torch.cat((K[[k_idx]], x_test[2]), dim=0)[None, :]

            plot_contour_at_coverage_2D(
                axs_top[method_idx],
                x_eval,
                conformalizer_dict[method_name],
                f"Group {k_idx + 1}",
                0.05,
                ylim,
                zlim,
                color=color,
                linestyle=linestyle,
            )

            avgCoverage = (
                res_df.loc[res_df.method == method_name]
                .AvgCoverage.round(2)
                .item()
            )
            axs_top[method_idx].set_title(
                f"{method_name} (AvgCov: {avgCoverage})", y=0.97
            )
            axs_top[method_idx].legend(loc="upper left", fontsize=8)

    for axis in axs_top:
        axis.label_outer()

    # Bottom metric plots
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1], wspace=0.3
    )
    ax_bottom = [fig.add_subplot(bottom_gs[0, i]) for i in range(3)]

    res_df_filtered = res_df.query(
        "method in ['MCP', 'HDR', 'L-CP', 'ST-DQR', 'PCP', 'HD-CP', 'FACTOR']"
    )

    ax_bottom[0].bar(
        x="method", height="AvgVol", data=res_df_filtered.iloc[1:, :], color=palette
    )
    ax_bottom[0].set_ylabel("Average region size")

    ax_bottom[1].bar(
        x="method", height="KS", data=res_df_filtered.iloc[1:, :], color=palette
    )
    ax_bottom[1].set_ylabel("Empirical KS distance")

    ax_bottom[2].bar(
        x="method", height="AvgTime", data=res_df_filtered.iloc[1:, :], color=palette
    )
    ax_bottom[2].set_ylabel("Elapsed time")

    savefig(f"{data_set}_2d_byGroup_demo.pdf")


# Global Multi-Dataset Summary Plots
def load_sim_res(data_set):
    with open(f"{data_set}_n{n}_res.pkl", "rb") as f:
        res = pickle.load(f)
    return pd.DataFrame(
        [
            {
                "method": r["method"],
                "data_set": data_set,
                "CovStd": np.std(r),
                "KS": np.max(r["coverage"]) - np.min(r["coverage"]),
                "AvgVol": (np.mean(r["volume"])).item(),
            }
            for r in res
        ]
    )


res_df_all = pd.concat(
    [
        pd.read_csv(f"{data_name}_n{n}_rst.csv")
        for data_name in ["mvnormal_2_0", "mvnormal_2_0.5", "mvnormal_2_0.8"]
    ]
)
res_df_all["data_set"] = res_df_all["data_set"].map(
    {
        "mvnormal_2_0": "rho = 0",
        "mvnormal_2_0.5": "rho = 0.5",
        "mvnormal_2_0.8": "rho = 0.8",
    }
)
res_df_all = res_df_all.query(f"method in {methods}")
res_df_all = res_df_all[res_df_all["method"] != "Oracle"]

fig, ax_bottom = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(
    x="data_set",
    y="AvgVol",
    data=res_df_all,
    hue="method",
    ax=ax_bottom[0],
    palette=palette,
)
ax_bottom[0].set_ylabel("Average region size")
ax_bottom[0].set_xlabel(" ")

sns.barplot(
    x="data_set",
    y="KS",
    data=res_df_all,
    hue="method",
    ax=ax_bottom[1],
    palette=palette,
)
ax_bottom[1].set_ylabel("Empirical KS distance")
ax_bottom[1].set_xlabel(" ")

sns.barplot(
    x="data_set",
    y="AvgTime",
    data=res_df_all,
    hue="method",
    ax=ax_bottom[2],
    palette=palette,
)
ax_bottom[2].set_ylabel("Elapsed time")
ax_bottom[2].set_xlabel(" ")

handles, labels = ax_bottom[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.13),
    ncol=len(labels),
    fontsize=12,
    frameon=False,
)

for ax in ax_bottom:
    ax.get_legend().remove()
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

plt.tight_layout()
savefig(f"res_n{n}.pdf")