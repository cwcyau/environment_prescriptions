import xarray as xr
from funcs import plot_practices, plot_prior_distributions

# parameters
prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_02_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_03_2010-08_2025-08_with_flags.nc",
    "data/prescriptions_0501_2010-08_2025-08_with_flags.nc",
]
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
seed = 42

priors_done = False
for prescriptions_path in prescriptions_paths:
    ds = xr.open_dataset(prescriptions_path)
    if not priors_done:
        plot_prior_distributions(ds)
        priors_done = True
    plot_practices(ds, prescriptions_path, flag_types=flag_types, seed=seed)
