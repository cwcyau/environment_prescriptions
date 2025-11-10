import os
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parameters

# prescription categories and corresponding folder roots
prescription_codes = ['02_03_0501', '02', '03', '0501']
labels = ['All', 'Cardiovascular', 'Respiratory', 'Antibiotics']
colours = ['black', 'red', 'blue', 'orange']
y_jitter = 0.15  # vertical jitter between overlapping categories
seasonal_correction_in = False  # set to True if seasonally corrected inputs were used
standardised_outputs = True  # set to True if standardised output "items" were used
generate_arviz_plots = True  # set to True to generate ArviZ diagnostic plots for each model

# fields in the analysis
value_vars = [
    "hydro_rain",
    "met_rain",
    "met_tmax",
    "aqrean_carbon_monoxide",
    "aqrean_daqi_overall",
    "aqrean_nitrogen_monoxide",
    "aqrean_nitrogen_dioxide",
    "aqrean_daqi_nitrogen_dioxide",
    "aqrean_nox_expressed_as_nitrogen_dioxide",
    "aqrean_ozone",
    "aqrean_daqi_ozone",
    "aqrean_pm2p5",
    "aqrean_pm10",
    "aqrean_daqi_pm10",
    "aqrean_sulfur_dioxide",
    "aqrean_daqi_sulfur_dioxide"
]
value_vars = [ft + "_values" for ft in value_vars]

# set input and output paths correctly
if seasonal_correction_in:
    folders = "deseasonalised_inputs/"
else:
    folders = "raw_inputs/"
if standardised_outputs:
    folders += "standardised_outputs/"
else:
    folders += "raw_outputs/"
results_roots = [
    f"outputs/prescriptions_{c}_2010-08_2025-08/{folders}" for c in prescription_codes
]
plot_root = f"outputs/comparison/{folders}"

# load data
idatas = []
dataframes = []
for root in results_roots:
    csv_path = root + "bayesian_model_summary.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: could not find results at {csv_path}, skipping...")
        continue
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Warning: results in {csv_path} are empty, skipping...")
        continue
    df = df.rename(columns={"Unnamed: 0": "name"})
    dataframes.append(df)

if not dataframes:
    raise FileNotFoundError("No Bayesian results CSVs found — nothing to plot.")

# =================================================================================================
# all in one figure

fig, ax = plt.subplots(figsize=(10, 8))
for i, (df, label, color) in enumerate(zip(dataframes, labels, colours)):
    if df.empty:  # skip missing ones
        continue

    # get data for this var
    df = df[df["name"].isin(value_vars)].copy()
    df = df.set_index("name").reindex(value_vars)

    # set jitter position and plot
    y_pos = np.arange(len(value_vars)) + (i - len(dataframes) / 2) * y_jitter
    ax.errorbar(
        df["mean"], y_pos,
        xerr=[df["mean"] - df["hdi_3%"], df["hdi_97%"] - df["mean"]],
        fmt='o', color=color, label=label, markersize=4, capsize=3
    )

# plot formatting
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(np.arange(len(value_vars)))
ax.set_yticklabels(value_vars)
ax.grid(True, linestyle=':', alpha=0.6)
ax.set_xlabel("Posterior mean (effect size)")
ax.set_title("Bayesian model coefficients comparison")
plt.tight_layout()
  
# shared legend
handles, labels_ = ax.get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=4, frameon=False)
fig.subplots_adjust(bottom=0.09)
plt.savefig(plot_root + "bayesian.png", dpi=300)
plt.close(fig='all')

# =================================================================================================
# individual variable plots

plot_folder = f"{plot_root}values_bayes/"
os.makedirs(plot_folder, exist_ok=True)
for var in value_vars:
    fig, ax = plt.subplots(figsize=(6, 4))
    matched = False
    for df, label, color in zip(dataframes, labels, colours):
        if df.empty:  # skip missing ones
            continue

        # extract var and plot
        sub = df[df["name"] == var]
        if not sub.empty:
            ax.errorbar(
                sub["mean"], label,
                xerr=[sub["mean"] - sub["hdi_3%"], sub["hdi_97%"] - sub["mean"]],
                fmt='o', color=color, markersize=5, capsize=4
            )
            matched = True
    if not matched:
        plt.close(fig='all')
        continue

    # plot formatting and save
    ax.axvline(0, color='black', lw=0.8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"{var}")
    ax.set_xlabel("Posterior mean (effect size)")
    fig.tight_layout()
    plt.savefig(f"{plot_folder}{var}.png", dpi=300)
    plt.close(fig='all')

print(f"Plots saved to {plot_root}")

# =================================================================================================
# generate ArviZ diagnostic plots for each model

if generate_arviz_plots:
    az.rcParams["plot.max_subplots"] = 40
    for root in results_roots:
        # get the inference data
        nc_path = os.path.join(root, "bayesian_model_idata.nc")
        if not os.path.exists(nc_path):
            print(f"Warning: no netCDF file found for at {nc_path}, skipping...")
            continue
        idata = az.from_netcdf(nc_path)
        arviz_out_dir = root + "bayesian_diagnostics/"
        os.makedirs(arviz_out_dir, exist_ok=True)
        posterior_vars = [v for v in idata.posterior.data_vars
                          if "|" not in v and "C(" not in v]

        # convergence plot
        fig = az.plot_trace(idata, var_names=posterior_vars)
        plt.tight_layout()
        plt.savefig(os.path.join(arviz_out_dir, "convergence.png"), dpi=200)
        plt.close(fig='all')

        # posterior plot
        fig = az.plot_posterior(
            idata,
            var_names=posterior_vars,
            hdi_prob=0.94,
            point_estimate="mean",
            kind="hist",
        )
        plt.tight_layout()
        plt.savefig(os.path.join(arviz_out_dir, "posterior.png"), dpi=200)
        plt.close(fig='all')

        # rank plots
        fig = az.plot_rank(idata, var_names=posterior_vars)
        plt.tight_layout()
        plt.savefig(os.path.join(arviz_out_dir, "rank.png"), dpi=200)
        plt.close(fig='all')

        # posterior predictive checks
        if "posterior_predictive" in idata.groups():
            fig = az.plot_ppc(idata, num_pp_samples=100)
            plt.tight_layout()
            plt.savefig(os.path.join(arviz_out_dir, "ppc.png"), dpi=200)
            plt.close(fig='all')
        else:
            print(f"Warning: no posterior predictive data foundin {nc_path}, skipping PPC.")
        
        print(f"Analysis diagnostic plots saved to {arviz_out_dir}")
