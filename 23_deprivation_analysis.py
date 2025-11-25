import xarray as xr
from funcs import prepare_ds, run_deprivation_mixed_models, compare_deprivation_analyses, status

# parameters
min_practice_obs = 20  # practices with fewer points will be excluded
n_jobs = 12  # number of parallel jobs to run
n_practices = None  # limit to n practices with most data points (for testing, set None to use all practices)
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope (keep as 2 as runs within walltime)
deseasonalise_output = True  # whether to include a seasonal correction term for output variable (items) (always True as adding seasonal term is inexpensive)
deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
adjust_predictors = 'z-global'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
standardise_items = True  # whether to standardise items variable (per practice)
results_root = "outputs/mixed_effects_deprivation/"

# codes to process
prescription_codes = ["02_03_0501", "02", "03", "0501"]

# name of the variables containing raw values (not flags)
value_vars = [
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

if __name__ == "__main__":
    # for codes in prescription_codes:
        # # set the path to get correct flags (generated from deseasonalised or non-deseasonalised values)
        # if deseasonalise_predictors:
        #     input_path = f"data/prescriptions_{codes}_2010-08_2025-08_with_flags_deseasonalised.nc"
        # else:
        #     input_path = f"data/prescriptions_{codes}_2010-08_2025-08_with_flags.nc"
        # results_folder = f"{results_root}{codes}/"

        # # process the dataset
        # status(f"Processing file: {input_path}")
        # ds = xr.load_dataset(input_path)
        # ds = prepare_ds(ds,
        #                 n_practices=n_practices,
        #                 standardise_items=standardise_items,
        #                 adjust_predictors=adjust_predictors,
        #                 deseasonalise_predictors=deseasonalise_predictors)

        # # run deprivation analysis models
        # run_deprivation_mixed_models(
        #     ds,
        #     value_vars,
        #     results_folder,
        #     deseasonalise_output=deseasonalise_output,
        #     practice_correction=practice_correction,
        #     min_practice_obs=min_practice_obs,
        #     n_jobs=n_jobs,
        # )

    compare_deprivation_analyses(results_root)
    status("Script complete.")
