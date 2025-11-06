# import os
# os.environ["PYTENSOR_FLAGS"] = "mode=NUMBA"  # for laptop runs

import xarray as xr
from funcs import run_all_flag_mixed_models, run_all_value_mixed_models, status

# parameters
min_practice_obs = 20  # practices with fewer points will be excluded
n_jobs = 12  # number of parallel jobs to run
seasonal_correction = True  # whether to include a seasonal correction term
practice_correction = 2  # 0 = none, 1 = intercept only, 2 = intercept + slope
standardise_values = True  # whether to standardise values variables (global)
standardise_items = True  # whether to standardise items variable (per practice)

# file paths to process
prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_02_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_03_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_0501_2010-08_2025-08_with_flags.nc",
]

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
    for prescriptions_path in prescriptions_paths:
        # get the data and set save folder
        status(f"Processing file: {prescriptions_path}")
        ds = xr.open_dataset(prescriptions_path)
        save_folder = "outputs/" + prescriptions_path.split("/")[-1].replace(".nc", "/")

        # run mixed models comparing flags
        # compares:
        #    flooding: flood == 1 to flood == 0
        #    met and hydro: high/low to median periods
        #    particulate mass: high == 1 to high == 0
        #    particulate DAQI: (very high, high) to (moderate, low) and (very high, high, moderate) to (low)
        status(f"Running mixed-effects models for flags...")
        run_all_flag_mixed_models(ds,
                                  flag_types,
                                  save_folder,
                                  seasonal_correction=seasonal_correction,
                                  practice_correction=practice_correction,
                                  standardise_items=standardise_items,
                                  min_practice_obs=min_practice_obs,
                                  n_jobs=n_jobs,)

        # run mixed models using raw measurements
        # handles all variables except for flooding, as there are no values for this
        status(f"Running mixed-effects models for values...")
        run_all_value_mixed_models(ds,
                                   value_vars,
                                   save_folder,
                                   seasonal_correction=seasonal_correction,
                                   practice_correction=practice_correction,
                                   standardise_values=standardise_values,
                                   standardise_items=standardise_items,
                                   min_practice_obs=min_practice_obs,
                                   n_jobs=n_jobs)

    status("Script complete.")
