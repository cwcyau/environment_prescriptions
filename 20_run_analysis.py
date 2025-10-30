import xarray as xr
from funcs import plot_practices, practice_analysis, global_analysis, mixed_effects_model

# parameters
prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_02_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_03_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_0501_2010-08_2025-08_with_flags.nc",
]
flag_types = ["hydro_rain", "met_rain", "met_tmax", "flood"]
seed = 42

for prescriptions_path in prescriptions_paths:
    # load dataset
    ds = xr.open_dataset(prescriptions_path)

    # run analyses
    plot_practices(ds, prescriptions_path, flag_types=flag_types, seed=seed)  # flag types currently hard-coded to visualise properly
    # practice_analysis(ds, flag_types, prescriptions_path, seed=seed)
    # global_analysis(ds, flag_types, prescriptions_path, seed=seed)
    # mixed_effects_model(ds, flag_types, prescriptions_path)
