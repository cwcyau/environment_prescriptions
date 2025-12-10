# for testing on laptop ===========================================================================
# # REMEMBER TO COMMENT OUT PARAMETERS BELOW IMPORTS
# import os
# import xarray as xr
# from funcs import (
#     prepare_ds, run_bayesian_model_lagged, generate_bayesian_diagnostics,
#     compare_bayesian_models_lagged, status
# )
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"
# prescription_code = "02_03_0501"
# n_practices = 100
# min_obs_per_practice = 20
# use_pca = False
# practice_correction = 2
# deseasonalise_output = True
# draws = 100
# tune = 100
# chains = 2
# cores = 2
# deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
# adjust_predictors = 'z-global'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
# standardise_items = False  # KEEP FALSE AS USING LOG ITEMS NOW - whether to standardise items variable (per practice)
# clean_items = True  # whether to clean 'items' by removing low values and practices with low means
# practice_mean_thresh = 2000  # threshold for defining large vs small practices
# results_folder = f"outputs/bayes_lagged_test/{prescription_code}/"
# predictors = [
#     "flood",
#     "hydro_rain_values",
#     "aqrean_carbon_monoxide_values",
#     "aqrean_daqi_overall_values",
# ]
# interactions = None
# =================================================================================================

# for running models on arc =======================================================================
import xarray as xr
import argparse
from funcs import (
    prepare_ds, run_bayesian_model_lagged, generate_bayesian_diagnostics,
    compare_bayesian_models_lagged, status
)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prescription_code", type=str, required=True,
                    help="BNF prescription code(s) to analyse.")
args = parser.parse_args()
prescription_code = args.prescription_code

# general parameters
n_practices = None  # limit to n randomly selected practices (None for all practices)
min_obs_per_practice = 20  # practices with fewer points will be excluded
use_pca = False  # whether to use raw values (False) or PCA to reduce dimensionality of factors (True) (keep False as can run full model within walltime)
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope, 3 = intercept + slope + correlation (keep as 2 as runs within walltime)
deseasonalise_output = True  # whether to include a seasonal correction term for output variable (items) (always True as adding seasonal term is inexpensive)
deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
adjust_predictors = 'z-global'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
standardise_items = False  # KEEP FALSE AS USING LOG ITEMS NOW - whether to standardise items variable (per practice)
clean_items = True  # whether to clean 'items' by removing low values and practices with low means
practice_mean_thresh = 500  # threshold for defining large vs small practices
results_folder = f"outputs/bayes_lagged/{prescription_code}/"
draws = 2000  # number of MCMC draws
tune = 2000  # number of tuning steps
chains = 8  # number of MCMC chains
cores = 8  # number of CPU cores to use

# names of values to analyse
predictors = [
    "flood",
    "imd_centile_values",
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
interactions = None
# interactions = [
#     "met_tmax_values*met_rain_values",
#     "met_tmin_values*met_rain_values",
#     "met_tmax_values*hydro_rain_values",
#     "met_tmin_values*hydro_rain_values",
# ]
# =================================================================================================

if __name__ == "__main__":
    # set files/folder paths
    prescriptions_path = f"data/prescriptions_{prescription_code}_2010-08_2025-08_with_flags.nc"

    # collect and prepare the dataset
    status(f"Processing code {prescription_code} @ {prescriptions_path}")
    ds = xr.open_dataset(prescriptions_path)
    ds = prepare_ds(ds,
                    n_practices=n_practices,
                    standardise_items=standardise_items,
                    clean_items=clean_items,
                    adjust_predictors=adjust_predictors,
                    deseasonalise_predictors=deseasonalise_predictors,
                    practice_mean_thresh=practice_mean_thresh)

    # run the bayesian models
    status("Running lagged Bayesian models...")
    run_bayesian_model_lagged(
        ds,
        raw_vars=predictors,
        results_folder=results_folder,
        deseasonalise_output=deseasonalise_output,
        practice_correction=practice_correction,
        min_practice_obs=min_obs_per_practice,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
    )

    # generate diagnostic plots
    status("Generating model diagnostics...")
    generate_bayesian_diagnostics(results_folder)

    # running model comparisons
    status("Comparing model results...")
    compare_bayesian_models_lagged("/".join(results_folder.split("/")[:-2]) + "/")

    status("Script complete.")
