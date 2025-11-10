# import os
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"  # for laptop runs

import xarray as xr
from funcs import run_all_flag_mixed_models, run_all_value_mixed_models, status

# parameters
min_practice_obs = 20  # practices with fewer points will be excluded
n_jobs = 12  # number of parallel jobs to run
# seasonal_correction_in = False  # whether to apply seasonal correction to predictor variables
seasonal_correction_out = True  # whether to include a seasonal correction term for output variable (items)
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope
standardise_values = True  # whether to standardise values variables (global)
standardise_items = True  # whether to standardise items variable (per practice)

# codes to process
prescription_codes = ["02_03_0501", "02", "03", "0501"]

# base names of flags and values
# e.g. hydro_rain for hydro_rain_high, hydro_rain_median, hydro_rain_low, hydro_rain_values
flag_types = [
    "hydro_rain",
    "met_rain",
    "met_tmax",
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
    "aqrean_pm10",
    "aqrean_daqi_pm10",
    "aqrean_sulfur_dioxide",
    "aqrean_daqi_sulfur_dioxide"
]

# set names of the variables containing raw values (not flags)
value_vars = [ft + "_values" for ft in flag_types if ft != "flood"]

if __name__ == "__main__":
    # loop through each file and run the models
    for seasonal_correction_in in [False, True]:
        for standardise_items in [False, True]:
            for codes in prescription_codes:
                # set the correct paths
                if seasonal_correction_in:
                    input_path = f"data/prescriptions_{codes}_2010-08_2025-08_with_flags_deseasonalised.nc"
                    results_folder = f"outputs/prescriptions_{codes}_2010-08_2025-08/deseasonalised_inputs/"
                else:
                    input_path = f"data/prescriptions_{codes}_2010-08_2025-08_with_flags.nc"
                    results_folder = f"outputs/prescriptions_{codes}_2010-08_2025-08/raw_inputs/"
                
                if standardise_items:
                    results_folder += "standardised_outputs/"
                else:
                    results_folder += "raw_outputs/"

                # get the data and set save folder
                status(f"Processing file: {input_path} with standardised items = {standardise_items}")
                ds = xr.open_dataset(input_path)
                # run mixed models comparing flags
                # compares:
                #    flooding: flood == 1 to flood == 0
                #    met and hydro: high/low to median periods
                #    particulate mass: high == 1 to high == 0
                #    particulate DAQI: (very high, high) to (moderate, low) and (very high, high, moderate) to (low)
                status(f"Running mixed-effects models for flags...")
                run_all_flag_mixed_models(ds,
                                        flag_types,
                                        results_folder,
                                        seasonal_correction_out=seasonal_correction_out,
                                        practice_correction=practice_correction,
                                        standardise_items=standardise_items,
                                        min_practice_obs=min_practice_obs,
                                        n_jobs=n_jobs,)

                # run mixed models using raw measurements
                # handles all variables except for flooding, as there are no values for this
                status(f"Running mixed-effects models for values...")
                run_all_value_mixed_models(ds,
                                        value_vars,
                                        results_folder,
                                        seasonal_correction_in=seasonal_correction_in,
                                        seasonal_correction_out=seasonal_correction_out,
                                        practice_correction=practice_correction,
                                        standardise_values=standardise_values,
                                        standardise_items=standardise_items,
                                        min_practice_obs=min_practice_obs,
                                        n_jobs=n_jobs)

    status("Script complete.")
