import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# parameters
# flagged_results = "mixed_effects_flag_results.csv"
# values_results = "mixed_effects_values_results.csv"
# output_folder = "outputs/analysis_comparison"

flagged_results = "mixed_effects_flag_results_standardised_items.csv"
values_results = "mixed_effects_values_results_standardised_items.csv"
output_folder = "outputs/analysis_comparison_standardised_items"

prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_02_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_03_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_0501_2010-08_2025-08_with_flags.nc",
]
y_jitter = 0.15  # amount to jitter y-axis for visibility on combined plots
labels = ['All', 'Cardiovascular', 'Respiratory', 'Antibiotics']
colours = ['black', 'red', 'blue', 'orange']

# load data
Path(output_folder).mkdir(exist_ok=True)
data_flag = []
data_values = []
for path in prescriptions_paths:
    base = Path(path).stem
    out_dir = Path(output_folder) / base
    data_dir = path.replace('data/', 'outputs/').replace(".nc", "")
    df_flag = pd.read_csv(data_dir + "/" + flagged_results)
    df_val = pd.read_csv(data_dir + "/" + values_results)
    df_flag["dataset"] = df_val["dataset"] = base
    data_flag.append(df_flag)
    data_values.append(df_val)
df_flag_all = pd.concat(data_flag, ignore_index=True)
df_val_all = pd.concat(data_values, ignore_index=True)

# =================================================================================================
# all on one figure

for hide_sulfur in [True, False]:
    # flagged
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    for i, (df, label, color) in enumerate(zip(data_flag, labels, colours)):
        if hide_sulfur:
            df = df[~df["name"].str.contains("aqrean_daqi_sulfur_dioxide")]
        y_pos = np.arange(len(df["name"])) + (i - len(labels)/2) * y_jitter  # <-- apply jitter
        axes[0].errorbar(df["coef"], y_pos,
                        xerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
                        fmt='o', color=colours[i], label=labels[i], markersize=3, capsize=2)
    axes[0].set_yticks(np.arange(len(df["name"])))
    axes[0].set_yticklabels(df["name"])

    # values
    for i, (df, label, color) in enumerate(zip(data_values, labels, colours)):
        x = df["name"].str.replace("_values", "")
        y_pos = np.arange(len(df["name"])) + (i - len(labels)/2) * y_jitter
        axes[1].errorbar(df["coef"], y_pos,
                        xerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
                        fmt='o', color=colours[i], label=labels[i], markersize=3, capsize=2)
    axes[1].set_yticks(np.arange(len(df["name"])))
    axes[1].set_yticklabels(x)

    # plot formatting
    for ax in axes:
        ax.axvline(0, color='black', lw=0.8)
        ax.grid(True, linestyle=':', alpha=0.6)
    axes[0].set_title("Flagged models")
    axes[1].set_title("Values models")
    axes[1].set_xlabel("Coefficient estimate")
    plt.tight_layout()
    
    # shared legend
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.09)

    if hide_sulfur:
        plt.savefig(f"{output_folder}/all_no_sulfur.png", dpi=300)
    else:
        plt.savefig(f"{output_folder}/all.png", dpi=300)
    plt.close(fig)


# =================================================================================================
# plot coefficient estimates for flagged variables individually (comparing datasets)

flag_varnames = sorted(df_flag_all["name"].unique())
for var in flag_varnames:
    fig, ax = plt.subplots(figsize=(6, 4))
    matched = False
    for df, label, color in zip(data_flag, labels, colours):
        sub = df[df["name"] == var]
        if not sub.empty:
            ax.errorbar(sub["coef"], label,
                        xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                        fmt='o', color=color, markersize=5, capsize=4)
            matched = True
    if not matched:
        plt.close(fig)
        continue
    ax.axvline(0, color='black', lw=0.8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"{var}")
    ax.set_xlabel("Coefficient estimate")
    fig.tight_layout()
    plt.savefig(f"{output_folder}/flags_{var}.png", dpi=300)
    plt.close(fig)

# =================================================================================================
# plot coefficient estimates for values variables individually (comparing datasets)

val_varnames = sorted(df_val_all["name"].str.replace("_values", "").unique())
for var in val_varnames:
    fig, ax = plt.subplots(figsize=(6, 4))
    matched = False
    for df, label, color in zip(data_values, labels, colours):
        sub = df[df["name"].str.replace("_values", "") == var]
        if not sub.empty:
            ax.errorbar(sub["coef"], label,
                        xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                        fmt='o', color=color, markersize=5, capsize=4)
            matched = True
    if not matched:
        plt.close(fig)
        continue
    ax.axvline(0, color='black', lw=0.8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"{var}")
    ax.set_xlabel("Coefficient estimate")
    fig.tight_layout()
    plt.savefig(f"{output_folder}/values_{var}.png", dpi=300)
    plt.close(fig)
