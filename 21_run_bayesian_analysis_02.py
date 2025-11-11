# import os
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"  # for laptop runs

import xarray as xr
from funcs import prepare_ds, run_bayesian_model, status

SUFFIX = "interactions/"

# general parameters
prescription_code = "02"
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
results_root = f"outputs/inputs_{adjust_predictors}{seasonal_str}/{standardise_str}/{SUFFIX}"

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

# no interactions =================================================================================
# interactions = None
# file_suffix = ""
# =================================================================================================

# ideal ===========================================================================================
interactions = [
    "hydro_rain_values*met_rain_values",
    "met_rain_values*met_tmax_values",
    "hydro_rain_values*met_tmax_values",
    "aqrean_nitrogen_dioxide_values*aqrean_nox_expressed_as_nitrogen_dioxide_values",
    "aqrean_nitrogen_monoxide_values*aqrean_nox_expressed_as_nitrogen_dioxide_values",
    ]
interactions += [
    "aqrean_daqi_overall_values*aqrean_daqi_" + v.split("aqrean_daqi_")[-1]
    for v in value_vars
    if "daqi" in v and "overall" not in v
    ]
interactions += [
    f"{v}*{v.replace('aqrean_daqi', 'aqrean')}"
    for v in value_vars
    if "daqi" in v and "overall" not in v
]
file_suffix = "_interactions"
# =================================================================================================

if __name__ == "__main__":
    # set files/folder paths
    prescriptions_path = f"data/prescriptions_{prescription_code}_2010-08_2025-08_with_flags.nc"
    results_folder = f"{results_root}{prescription_code}/"

    # get the data and set save folder
    status(f"Processing file: {prescriptions_path}")
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
        file_suffix=file_suffix,
        deseasonalise_output=deseasonalise_output,
        practice_correction=practice_correction,
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
