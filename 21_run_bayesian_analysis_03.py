# import os
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"  # for laptop runs

import xarray as xr
from funcs import run_bayesian_raw_model, status

# parameters
n_practices = None  # limit to n practices with most data points (4000 for testing on laptop, None for full analysis)
seasonal_correction_in = False  # whether to apply seasonal correction to predictor variables
seasonal_correction_out = True  # whether to include a seasonal correction term for output variable (items)
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
prescription_codes = "03"

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

# ideal ===========================================================================================
# interactions = [
#     "hydro_rain_values*met_rain_values",
#     "met_rain_values*met_tmax_values",
#     "hydro_rain_values*met_tmax_values",
#     "aqrean_nitrogen_dioxide_values*aqrean_nox_expressed_as_nitrogen_dioxide_values",
#     "aqrean_nitrogen_monoxide_values*aqrean_nox_expressed_as_nitrogen_dioxide_values",
#     ]
# interactions += [
#     "aqrean_daqi_overall_values*aqrean_daqi_" + v.split("aqrean_daqi_")[-1]
#     for v in value_vars
#     if "daqi" in v and "overall" not in v
#     ]
# interactions += [
#     f"{v}*{v.replace('aqrean_daqi', 'aqrean')}"
#     for v in value_vars
#     if "daqi" in v and "overall" not in v
# ]
# =================================================================================================

if __name__ == "__main__":
    # set files/folder paths
    prescriptions_path = f"data/prescriptions_{prescription_codes}_2010-08_2025-08_with_flags.nc"
    if seasonal_correction_in:
        results_folder = f"outputs/prescriptions_{prescription_codes}_2010-08_2025-08/deseasonalised_inputs/"
    else:
        results_folder = f"outputs/prescriptions_{prescription_codes}_2010-08_2025-08/raw_inputs/"
    if standardise_items:
        results_folder += "standardised_outputs/"
    else:
        results_folder += "raw_outputs/"

    # get the data and set save folder
    status(f"Processing file: {prescriptions_path}")
    ds = xr.open_dataset(prescriptions_path)

    run_bayesian_raw_model(
        ds,
        raw_vars=value_vars,
        results_folder=results_folder,
        seasonal_correction_in=seasonal_correction_in,
        seasonal_correction_out=seasonal_correction_out,
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
