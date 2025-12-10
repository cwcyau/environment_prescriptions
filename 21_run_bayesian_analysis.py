# for testing on laptop ===========================================================================
# # REMEMBER TO COMMENT OUT PARAMETERS BELOW IMPORTS
# import os, jax
# import xarray as xr
# from funcs import (
#     prepare_ds, run_bayesian_model, compare_bayesian_models, status, run_bayesian_model_gpu
# )
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"
# method = "standard"
# prescription_code = "02_03_0501"
# n_practices = 100
# min_obs_per_practice = 25
# lag = 3  # number of time lags to include for each predictor (0 for no lags)
# almon_order = 2  # order of almon lag polynomial (only used if lag > 0)
# use_pca = False
# practice_correction = 2
# deseasonalise_output = True
# draws = 100
# tune = 100
# chains = 8
# cores = 8
# use_gpu = False
# deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
# adjust_predictors = 'z-global'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
# standardise_items = False  # KEEP FALSE AS USING LOG ITEMS NOW - whether to standardise items variable (per practice)
# clean_items = True  # whether to clean 'items' by removing low values and practices with low means
# practice_mean_thresh = 2000  # threshold for defining large vs small practices
# if lag > 0:
#     results_folder = f"outputs/bayes_lagged_test/{prescription_code}/"
# else:
#     results_folder = f"outputs/bayes_test/{prescription_code}/"
# predictors = [
#     "flood",
#     "hydro_rain_values",
#     "aqrean_carbon_monoxide_values",
#     "aqrean_daqi_overall_values",
# ]
# =================================================================================================

# for running models on arc =======================================================================
# run with qsend_bayes_gpu 02_03_0501 3 3000 3000 8 0
import xarray as xr, jax
import argparse
from funcs import (
    prepare_ds, run_bayesian_model, compare_bayesian_models, status, run_bayesian_model_gpu
)

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lag", type=int, required=True,
                    help="Number of time lags to include for each predictor (0 for no lags).")
parser.add_argument("--prescription_code", type=str, required=True,
                    help="BNF prescription code(s) to analyse.")
parser.add_argument("--tune", type=int, default=2000,
                    help="Number of tuning steps for MCMC (default: 2000).")
parser.add_argument("--draws", type=int, default=2000,
                    help="Number of MCMC draws (default: 2000).")
parser.add_argument("--chains", type=int, default=8,
                    help="Number of MCMC chains (default: 8).")
parser.add_argument("--cores", type=int, default=8,
                    help="Number of CPU cores to use (default: 8).")
parser.add_argument("--n_practices", type=int, default=None,
                    help="Limit to n randomly selected practices (None for all practices).")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU")
parser.add_argument("--no_gpu",  action="store_false", dest="use_gpu", help="Disable GPU")
parser.set_defaults(use_gpu=True)
args = parser.parse_args()
lag = args.lag
prescription_code = args.prescription_code
tune = args.tune
draws = args.draws
n_practices = args.n_practices
if n_practices == 0:
    n_practices = None
chains = args.chains
cores = args.cores
use_gpu = args.use_gpu

# data preparation parameters
# n_practices = None  # limit to n randomly selected practices (None for all practices)
standardise_items = False  # KEEP FALSE AS USING LOG ITEMS NOW - whether to standardise items variable (per practice)
clean_items = True  # whether to clean 'items' by removing low values and practices with low means
adjust_predictors = 'z-global'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
practice_mean_thresh = 500  # threshold for defining large vs small practices

# modelling parameters
min_obs_per_practice = 25  # practices with fewer points will be excluded (after all rows in final df with nans have been removed)
almon_order = 2  # order of almon lag polynomial (only used if lag > 0)
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope, 3 = intercept + slope + correlation (keep as 2 as runs within walltime)
deseasonalise_output = True  # whether to include a seasonal correction term for output variable (items) (always True as adding seasonal term is inexpensive)
# draws = 2000  # number of MCMC draws
# tune = 2000  # number of tuning steps
# chains = 8  # number of MCMC chains
# cores = 1  # number of CPU cores to use

# set folder name for outputs
if lag > 0:
    results_folder = f"outputs/bayes_lagged_{lag}/{prescription_code}/"
else:
    results_folder = f"outputs/bayes_standard/{prescription_code}/"

# names of values to analyse
# predictors = [  # full set
#     "flood",
#     "imd_centile_values",
#     "hydro_rain_values",
#     "met_rain_values",
#     "met_tmax_values",
#     "met_tmin_values",
#     "aqrean_carbon_monoxide_values",
#     "aqrean_daqi_overall_values",
#     "aqrean_nitrogen_monoxide_values",
#     "aqrean_nitrogen_dioxide_values",
#     "aqrean_daqi_nitrogen_dioxide_values",
#     "aqrean_nox_expressed_as_nitrogen_dioxide_values",
#     "aqrean_ozone_values",
#     "aqrean_daqi_ozone_values",
#     "aqrean_pm2p5_values",
#     "aqrean_daqi_pm2p5_values",
#     "aqrean_pm10_values",
#     "aqrean_daqi_pm10_values",
#     "aqrean_sulfur_dioxide_values",
#     "aqrean_daqi_sulfur_dioxide_values"
# ]
predictors = [  # reduced set
    "flood",
    "imd_centile_values",
    "hydro_rain_values",
    # "met_rain_values",
    "met_tmax_values",
    "met_tmin_values",
    "aqrean_carbon_monoxide_values",
    # "aqrean_daqi_overall_values",
    # "aqrean_nitrogen_monoxide_values",
    # "aqrean_nitrogen_dioxide_values",
    # "aqrean_daqi_nitrogen_dioxide_values",
    "aqrean_nox_expressed_as_nitrogen_dioxide_values",
    "aqrean_ozone_values",
    # "aqrean_daqi_ozone_values",
    "aqrean_pm2p5_values",
    # "aqrean_daqi_pm2p5_values",
    "aqrean_pm10_values",
    # "aqrean_daqi_pm10_values",
    "aqrean_sulfur_dioxide_values",
    # "aqrean_daqi_sulfur_dioxide_values"
]
# =================================================================================================

if __name__ == "__main__":
    # set files/folder paths
    prescriptions_path = f"data/prescriptions_{prescription_code}_2010-08_2025-08_with_flags.nc"
    status("Starting Bayesian analysis script...")
    status(f"data path: {prescriptions_path}", level=1)
    status(f"results folder: {results_folder}", level=1)
    status(f"lag: {lag}, almon order: {almon_order}", level=1)
    status(f"tune: {tune}, draws: {draws}, chains: {chains}, cores: {cores}", level=1)
    status(f"n_practices: {n_practices}", level=1)
    status(f"use_gpu: {use_gpu}", level=1)
    status(f"predictors: {predictors}", level=1)

    # collect and prepare the dataset
    status(f"Preparing dataset...")
    ds = xr.open_dataset(prescriptions_path)
    ds = prepare_ds(ds,
                    n_practices=n_practices,
                    standardise_items=standardise_items,
                    clean_items=clean_items,
                    adjust_predictors=adjust_predictors,
                    deseasonalise_predictors=deseasonalise_predictors,
                    practice_mean_thresh=practice_mean_thresh)

    # run the bayesian models
    status("Initialising model run...")
    if use_gpu:
        jax.config.update("jax_platform_name", "gpu")
        status(f"Using device: {jax.devices()[0]}", level=1)
        run_bayesian_model_gpu(
            ds,
            raw_vars=predictors,
            results_folder=results_folder,
            lag=lag,
            almon_order=almon_order,
            deseasonalise_output=deseasonalise_output,
            practice_correction=practice_correction,
            min_practice_obs=min_obs_per_practice,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
        )
    else:
        jax.config.update("jax_platform_name", "cpu")
        status(f"Using device: {jax.devices()[0]}", level=1)
        run_bayesian_model(
            ds,
            raw_vars=predictors,
            results_folder=results_folder,
            lag=lag,
            almon_order=almon_order,
            deseasonalise_output=deseasonalise_output,
            practice_correction=practice_correction,
            min_practice_obs=min_obs_per_practice,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
        )

    # running model comparisons
    status("Comparing model results...")
    compare_bayesian_models("/".join(results_folder.split("/")[:-2]) + "/")

    status("Script complete.")
