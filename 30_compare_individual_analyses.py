import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parameters
# standardised_outputs = True  # plot analysis for standardised items
# seasonal_correction_in = True  # plot analysis for seasonally corrected inputs
prescription_codes = ['02_03_0501', '02', '03', '0501']
y_jitter = 0.15  # amount to jitter y-axis for visibility on combined plots
labels = ['All', 'Cardiovascular', 'Respiratory', 'Antibiotics']
colours = ['black', 'red', 'blue', 'orange']

for standardised_outputs in [True, False]:
    for seasonal_correction_in in [True, False]:

        # set input and output paths correctly
        plot_root = f"outputs/comparison/"
        if seasonal_correction_in:
            folders = "deseasonalised_inputs/"
        else:
            folders = "raw_inputs/"
        if standardised_outputs:
            folders += "standardised_outputs/"
        else:
            folders += "raw_outputs/"
        results_root = [
            f"outputs/prescriptions_{c}_2010-08_2025-08/{folders}" for c in prescription_codes
        ]
        plot_root += folders

        # load data
        os.makedirs(plot_root, exist_ok=True)
        data_flag = []
        data_values = []
        datasets = []
        for root in results_root:
            try:
                df_flag = pd.read_csv(f"{root}/mixed_effects_flag_results.csv")
                df_val = pd.read_csv(f"{root}/mixed_effects_values_results.csv")
            except FileNotFoundError:
                print(f"Warning: could not find results in {root}, skipping...")
                continue
            # skip empty results
            if df_flag.empty and df_val.empty:
                print(f"Warning: results in {root} are empty, skipping...")
                continue
            df_flag["dataset"] = root
            df_val["dataset"] = root
            data_flag.append(df_flag)
            data_values.append(df_val)

        # safe concatenation: create empty DataFrames with expected columns if nothing was found
        if data_flag:
            df_flag_all = pd.concat(data_flag, ignore_index=True)
        else:
            df_flag_all = pd.DataFrame(columns=["name", "coef", "ci_low", "ci_high", "dataset"])

        if data_values:
            df_val_all = pd.concat(data_values, ignore_index=True)
        else:
            df_val_all = pd.DataFrame(columns=["name", "coef", "ci_low", "ci_high", "dataset"])

        # =================================================================================================
        # all on one figure

        for hide_sulfur in [True, False]:
            if not data_flag and not data_values:
                print("No data available for plotting 'all' figure; skipping.")
                continue

            # determine how many series we actually have
            n_flag_series = len(data_flag)
            n_val_series = len(data_values)
            labels_flag = labels[:n_flag_series]
            colours_flag = colours[:n_flag_series]
            labels_val = labels[:n_val_series]
            colours_val = colours[:n_val_series]

            # flagged + values in a 2-row figure
            fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

            # flagged
            if n_flag_series > 0:
                last_df = None
                for i, (df, label, color) in enumerate(zip(data_flag, labels_flag, colours_flag)):
                    if hide_sulfur:
                        df = df[~df["name"].str.contains("aqrean_daqi_sulfur_dioxide")]
                    last_df = df
                    y_pos = np.arange(len(df["name"])) + (i - n_flag_series / 2) * y_jitter
                    axes[0].errorbar(df["coef"], y_pos,
                                     xerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
                                     fmt='o', color=color, label=label, markersize=3, capsize=2)
                if last_df is not None:
                    axes[0].set_yticks(np.arange(len(last_df["name"])))
                    axes[0].set_yticklabels(last_df["name"])
            else:
                axes[0].text(0.5, 0.5, 'No flagged results', ha='center', va='center')

            # values
            if n_val_series > 0:
                last_x = None
                for i, (df, label, color) in enumerate(zip(data_values, labels_val, colours_val)):
                    x = df["name"].str.replace("_values", "")
                    last_x = x
                    y_pos = np.arange(len(df["name"])) + (i - n_val_series / 2) * y_jitter
                    axes[1].errorbar(df["coef"], y_pos,
                                     xerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
                                     fmt='o', color=color, label=label, markersize=3, capsize=2)
                if last_x is not None:
                    axes[1].set_yticks(np.arange(len(last_x)))
                    axes[1].set_yticklabels(last_x)
            else:
                axes[1].text(0.5, 0.5, 'No values results', ha='center', va='center')

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
                plt.savefig(f"{plot_root}mixed_no_sulfur.png", dpi=300)
            else:
                plt.savefig(f"{plot_root}mixed.png", dpi=300)
            plt.close(fig)


        # =================================================================================================
        # plot coefficient estimates for flagged variables individually (comparing datasets)

        # if there are no flagged results, skip the per-variable flagged plots
        if df_flag_all.empty:
            print("No per-practice flagged results found; skipping flag variable plots.")
            flag_varnames = []
        else:
            flag_varnames = sorted(df_flag_all["name"].unique())
        os.makedirs(f"{plot_root}flags", exist_ok=True)
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
            plt.savefig(f"{plot_root}flags/{var}.png", dpi=300)
            plt.close(fig)

        # =================================================================================================
        # plot coefficient estimates for values variables individually (comparing datasets)

        if df_val_all.empty:
            print("No values results found; skipping value variable plots.")
            val_varnames = []
        else:
            val_varnames = sorted(df_val_all["name"].str.replace("_values", "").unique())
        os.makedirs(f"{plot_root}values", exist_ok=True)
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
            plt.savefig(f"{plot_root}values/{var}.png", dpi=300)
            plt.close(fig)
