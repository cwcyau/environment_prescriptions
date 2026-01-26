import xarray as xr
import numpy as np
import pandas as pd
import os

root = "outputs/bayes_lagged_2/"
folders = ["02_03_0501", "02", "03", "0501"]
results_folders = [os.path.join(root, folder) for folder in folders]

predictors = [
    "flood",
    "imd_centile_values",
    "hydro_rain_values",
    "met_tmax_values",
    "met_tmin_values",
    "aqrean_carbon_monoxide_values",
    "aqrean_nox_expressed_as_nitrogen_dioxide_values",
    "aqrean_ozone_values",
    "aqrean_pm2p5_values",
    "aqrean_pm10_values",
    "aqrean_sulfur_dioxide_values",
]

for folder in results_folders:
    if not os.path.exists(os.path.join(folder, "bayesian_model_summary.csv")):
        continue
    posterior = xr.open_dataset(os.path.join(folder, "bayesian_model_idata.nc"), group="posterior")#

    csv_data = pd.read_csv(os.path.join(folder, "bayesian_model_summary.csv"))
    csv_data["eti_mean_pct"] = np.nan
    csv_data["eti_2.5pc_pct"] = np.nan
    csv_data["eti_97.5pc_pct"] = np.nan
    csv_vars = csv_data["parameter"].values

    for var in posterior.data_vars:
        if var not in csv_vars:
            continue
        samples = posterior[var].values.flatten()
        mean_effect = samples.mean()
        lower, upper = np.quantile(samples, [0.025, 0.975])
        mean_effect = 100 * np.expm1(mean_effect)
        lower = 100 * np.expm1(lower)
        upper = 100 * np.expm1(upper)

        csv_data.loc[csv_data["parameter"] == var, "eti_mean_pct"] = mean_effect
        csv_data.loc[csv_data["parameter"] == var, "eti_2.5pc_pct"] = lower
        csv_data.loc[csv_data["parameter"] == var, "eti_97.5pc_pct"] = upper

    csv_data.to_csv(os.path.join(folder, "bayesian_model_summary.csv"), index=False)