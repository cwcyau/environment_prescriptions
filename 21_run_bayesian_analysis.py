# for testing on laptop ==============================
# REMEMBER TO COMMENT OUT PARAMETERS BELOW IMPORTS
# import os
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"
# method = "splines"
# prescription_code = "02"
# n_practices = 5
# min_obs_per_practice = 20
# use_pca = False
# practice_correction = 0
# deseasonalise_output = True
# draws = 100
# tune = 100
# chains = 2
# cores = 2
# ====================================================

import xarray as xr
import argparse
from funcs import prepare_ds, run_bayesian_model, status

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=str, required=True,
                    help="Model method: 'standard', 'pca' or 'splines'.")
parser.add_argument("--prescription_code", type=str, required=True,
                    help="BNF prescription code(s) to analyse.")
args = parser.parse_args()
method = args.method
prescription_code = args.prescription_code

# general parameters
n_practices = None  # limit to n practices with most data points (4000 for testing on laptop, None for full analysis)
min_obs_per_practice = 20  # practices with fewer points will be excluded
use_pca = False  # whether to use raw values (False) or PCA to reduce dimensionality of factors (True) (keep False as can run full model within walltime)
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope, 3 = intercept + slope + correlation (keep as 2 as runs within walltime)
deseasonalise_output = True  # whether to include a seasonal correction term for output variable (items) (always True as adding seasonal term is inexpensive)
draws = 3000  # number of MCMC draws
tune = 3000  # number of tuning steps
chains = 8  # number of MCMC chains
cores = 8  # number of CPU cores to use

# model parameters
deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
adjust_predictors = 'c-practice'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
standardise_items = True  # whether to standardise items variable (per practice)

# configure save folder name
seasonal_str = "" if not deseasonalise_predictors else "_deseasonalised"
standardise_str = "outputs_raw" if not standardise_items else "outputs_standardised"
results_root = f"outputs/inputs_{adjust_predictors}{seasonal_str}/{standardise_str}/{method}/{prescription_code}/"

# names of values to analyse
value_vars = [
    "flood",
    "hydro_rain_values",
    "met_rain_values",
    "met_tmax_values",
    "met_tmin_values",
    "aqrean_carbon_monoxide_values",
    "aqrean_daqi_overall_values",
    "aqrean_nitrogen_monoxide_values",
    "aqrean_nitrogen_dioxide_values",
    "aqrean_daqi_nitrogen_dioxide_values",
    "aqrean_nox_expressed_as_nitrogen_dioxide_values",
    "aqrean_ozone_values",
    "aqrean_daqi_ozone_values",
    "aqrean_pm2p5_values",
    "aqrean_daqi_pm2p5_values",
    "aqrean_pm10_values",
    "aqrean_daqi_pm10_values",
    "aqrean_sulfur_dioxide_values",
    "aqrean_daqi_sulfur_dioxide_values"
]

# interaction where combination of variables may have an effect
# e.g. if cold and wet has more of an effect than warm and wet or cold and dry: met_tmax_values*met_rain_values
# not required when using 'splines' method as interactions are modelled automatically
interactions = [
    "met_tmax_values*met_rain_values",
    "met_tmin_values*met_rain_values",
    "met_tmax_values*hydro_rain_values",
    "met_tmin_values*hydro_rain_values",
]


# =================================================================================================

if __name__ == "__main__":
    # set files/folder paths
    prescriptions_path = f"data/prescriptions_{prescription_code}_2010-08_2025-08_with_flags.nc"
    results_folder = f"{results_root}{prescription_code}/"

    # collect and prepare the dataset
    status(f"Processing file: {prescriptions_path}")
    status(f"Method: {method}, prescription code: {prescription_code}")
    ds = xr.open_dataset(prescriptions_path)
    ds = prepare_ds(ds,
                    n_practices=n_practices,
                    standardise_items=standardise_items,
                    adjust_predictors=adjust_predictors,
                    deseasonalise_predictors=deseasonalise_predictors)

    # run the bayesian models
    run_bayesian_model(
        ds,
        raw_vars=value_vars,
        results_folder=results_folder,
        method=method,
        deseasonalise_output=deseasonalise_output,
        practice_correction=practice_correction,
        min_practice_obs=min_obs_per_practice,
        interactions=interactions,
        poly_terms=None,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
    )

    status("Script complete.")
