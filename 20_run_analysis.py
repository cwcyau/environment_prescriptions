import xarray as xr
from funcs import (
    status,
    prepare_ds,
    run_mixed_effects_models,
    compare_mixed_models,
)

def build_predictor_spec(flag_types, value_vars):
    specs = []

    for ft in flag_types:
        # simple binary flood flag
        if ft == "flood":
            specs.append({
                "name": ft,
                "type": "binary_simple",
                "var": ft,
            })
            continue

        # weather flags with high/med/low flags
        elif "aqrean" not in ft:
            med  = f"{ft}_median"
            # high vs median
            high = f"{ft}_high"
            specs.append({
                "name": f"{ft}_high_vs_med",
                "type": "binary_pair",
                "var_anom": high,
                "var_med":  med
            })
            # low vs median
            low  = f"{ft}_low"
            specs.append({
                "name": f"{ft}_low_vs_med",
                "type": "binary_pair",
                "var_anom": low,
                "var_med":  med
            })
        
        # pollutant flags with high flag only
        elif "daqi" not in ft:
            high = f"{ft}_high"
            specs.append({
                "name": f"{ft}_high_vs_not",
                "type": "binary_simple",
                "var": high
            })

        # DAQI flags:
        # pair1: (very high, high) vs (moderate, low)
        # pair2: (very high, high, moderate) vs (low)
        else:
            daqi_vars = {
                "very_high": f"{ft}_very_high",
                "high":      f"{ft}_high",
                "moderate":  f"{ft}_moderate",
                "low":       f"{ft}_low"
            }
            specs.append({
                "name": f"{ft}_daqipair1",
                "type": "daqi_pair1",
                "vars": daqi_vars
            })
            specs.append({
                "name": f"{ft}_daqipair2",
                "type": "daqi_pair2",
                "vars": daqi_vars
            })

    for v in value_vars:
        # continuous value model
        specs.append({
            "name": v,
            "type": "continuous",
            "var": v
        })

    return specs


# Test Parameters =================================================================================
# min_practice_obs = 20  # practices with fewer points will be excluded
# n_jobs = 12  # number of parallel jobs to run
# n_practices = 1000  # limit to n randomly selected practices (None for all practices)
# practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope (keep as 2 as runs within walltime)
# deseasonalise_output = True  # whether to include a seasonal correction term for output variable (items) (always True as adding seasonal term is inexpensive)
# deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
# adjust_predictors = 'z-global'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
# standardise_items = False  # KEEP FALSE AS USING LOG ITEMS NOW - whether to standardise items variable (per practice)
# clean_items = True  # whether to clean 'items' by removing low values and practices with low means
# practice_mean_thresh = 500  # threshold for defining large vs small practices
# results_root = "outputs/mixed_effects_test/"
# prescription_codes = ["02_03_0501"]
# flag_types = [
#     "hydro_rain",
#     "flood",
#     "aqrean_carbon_monoxide",
#     "aqrean_daqi_overall",
# ]
# =================================================================================================

# Runtime parameters ==============================================================================
min_practice_obs = 20  # practices with fewer points will be excluded
n_jobs = 12  # number of parallel jobs to run
n_practices = None  # limit to n randomly selected practices (None for all practices)
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope (keep as 2 as runs within walltime)
deseasonalise_output = True  # whether to include a seasonal correction term for output variable (items) (always True as adding seasonal term is inexpensive)
deseasonalise_predictors = False  # whether to apply seasonal correction to predictor variables
adjust_predictors = 'z-global'  # 'z-global': standardise values globally, 'z-practice': standardise per practice, 'c-global': centre globally, 'c-practice': centre per practice, None: raw values
standardise_items = False  # KEEP FALSE AS USING LOG ITEMS NOW - whether to standardise items variable (per practice)
clean_items = True  # whether to clean 'items' by removing low values and practices with low means
practice_mean_thresh = 500  # threshold for defining large vs small practices
results_root = "outputs/mixed_effects/"
prescription_codes = ["02_03_0501", "02", "03", "0501"]
flag_types = [
    "hydro_rain",
    "met_rain",
    "met_tmax",
    "met_tmin",
    "flood",
    "aqrean_carbon_monoxide",
    "aqrean_daqi_overall",
    "aqrean_nitrogen_monoxide",
    "aqrean_nitrogen_dioxide",
    "aqrean_daqi_nitrogen_dioxide",
    "aqrean_nox_expressed_as_nitrogen_dioxide",
    "aqrean_ozone",
    "aqrean_daqi_ozone",
    "aqrean_pm2p5",
    "aqrean_daqi_pm2p5",
    "aqrean_pm10",
    "aqrean_daqi_pm10",
    "aqrean_sulfur_dioxide",
    "aqrean_daqi_sulfur_dioxide",
]
# =================================================================================================

# build list of continuous predictors and all model specifications
value_vars = [ft + "_values" for ft in flag_types if ft != "flood"]
predictors_spec = build_predictor_spec(flag_types, value_vars)

if __name__ == "__main__":
    for codes in prescription_codes:
        # ensure the correct input file is used
        if deseasonalise_predictors:
            input_path = f"data/prescriptions_{codes}_2010-08_2025-08_with_flags_deseasonalised.nc"
        else:
            input_path = f"data/prescriptions_{codes}_2010-08_2025-08_with_flags.nc"
        results_folder = f"{results_root}{codes}/"

        # prepare data for modelling
        status(f"=====     Processing prescription code: {codes}     =====")
        status(f"Preparing dataset for file: {input_path}")
        ds = xr.load_dataset(input_path)
        ds = prepare_ds(
            ds,
            n_practices=n_practices,
            standardise_items=standardise_items,
            clean_items=clean_items,
            adjust_predictors=adjust_predictors,
            deseasonalise_predictors=deseasonalise_predictors,
            practice_mean_thresh=practice_mean_thresh,
        )

        # run all mixed-effects models for this prescription type
        status("Running unified mixed-effects models...")
        run_mixed_effects_models(
            ds,
            predictors_spec,
            results_folder,
            deseasonalise_output=deseasonalise_output,
            practice_correction=practice_correction,
            min_practice_obs=min_practice_obs,
            n_jobs=n_jobs
        )
    
    # compare and plot results across prescription types
    status("Comparing mixed-effects model results across prescription types...")
    compare_mixed_models(
        results_folder=f"{results_root}",
        save_folder=f"{results_root}",
    )

    status("All analyses complete.")
