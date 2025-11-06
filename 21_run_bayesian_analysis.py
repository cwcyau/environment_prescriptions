# import os
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"  # for laptop runs

import json
import xarray as xr
import arviz as az
from funcs import run_bayesian_raw_model, status

# parameters
n_practices = None  # limit to n practices with most data points (4000 for testing on laptop, None for full analysis)
seasonal_correction = True  # whether to include a seasonal correction term
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope, 3 = intercept + slope + correlation
standardise_values = True  # whether to standardise values variables (global)
standardise_items = True  # whether to standardise items variable (per practice)
min_obs_per_practice = 20  # practices with fewer points will be excluded
use_pca = False  # whether to use raw values (False) or PCA to reduce dimensionality of factors (True)
draws = 3000  # number of MCMC draws
tune = 3000  # number of tuning steps
chains = 8  # number of MCMC chains
cores = 8  # number of CPU cores to use

# file paths to process
prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_02_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_03_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_0501_2010-08_2025-08_with_flags.nc",
]

# names of values to analyse
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
value_vars = [ft + "_values" for ft in value_vars if ft != "flood"]

# define interactions to include in Bayesian models
# e.g. "hydro_rain* x met_rain*" would create interactions between all fields starting hydro_rain and met_rain and ending in "_values"

# first try without:
interactions = None

# then gradually increase ===================================================================================================================
# interactions = [
#     "hydro_rain* x met_rain*",
#     "hydro_rain* x met_temp*",
#     "met_rain* x met_temp*"
#     ]
# ===========================================================================================================================================

# ideal =====================================================================================================================================
# interactions = [
#     "hydro_rain* x met_rain*",
#     "met_rain* x aqrean*",
#     "hydro_rain* x aqrean*",
#     "aqrean_daqi_overall* x aqrean_daqi_*"
#     ]
# # add additional interactions between all particulate mass and their DAQI equivalents
# for ft in factor_types:
#     if ft.startswith("aqrean") and "daqi" not in ft and "overall" not in ft and ft.replace('aqrean_', 'aqrean_daqi_') in factor_types:
#         interactions.append(f"{ft} x {ft.replace('aqrean_', 'aqrean_daqi_')}")
# ===========================================================================================================================================

if __name__ == "__main__":
    # loop through each file and run the models
    for prescriptions_path in prescriptions_paths:
        # get the data and set save folder
        status(f"Processing file: {prescriptions_path}")
        ds = xr.open_dataset(prescriptions_path)
        save_folder = "outputs/" + prescriptions_path.split("/")[-1].replace(".nc", "/")

        run_bayesian_raw_model(
            ds,
            raw_vars=value_vars,
            results_folder=save_folder,
            seasonal_correction=seasonal_correction,
            practice_correction=practice_correction,
            standardise_values=standardise_values,
            standardise_items=standardise_items,
            n_practices=n_practices,
            min_practice_obs=min_obs_per_practice,
            use_pca=use_pca,
            interactions=interactions,
            poly_terms=None,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
        )

    status("Script complete.")
