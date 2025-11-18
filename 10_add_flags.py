import xarray as xr
from funcs import add_hydrology_flags, add_geojson_flood_flags, add_met_flags, load_json, add_particulate_flags

# parameters
append_flags = True  # set to True if appending flags to existing flagged files
seasonal_correction = True  # whether to apply seasonal correction when adding flags
prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08.nc",
    "data/prescriptions_02_2010-08_2025-08.nc",
    "data/prescriptions_03_2010-08_2025-08.nc",
    "data/prescriptions_0501_2010-08_2025-08.nc"
]
hydrology_path = "data/hydrology_rainfall_stations.nc"
geojson_path = "data/Recorded_Flood_Outlines.geojson"
met_path = "data/met_office_stations.nc"
particulates_path = "data/particulates_by_practice_monthly_max.nc"

# load datasets
print("Loading datasets...")
hydrology_ds = xr.open_dataset(hydrology_path)
flood_geojson = load_json(geojson_path)
met_ds = xr.open_dataset(met_path)
particulates_ds = xr.load_dataset(particulates_path)
print("Datasets loaded.")

# set seasonality suffix
if seasonal_correction:
    seasonality = "_deseasonalised"
else:
    seasonality = ""

# set suffixes for loading/saving
if append_flags:
    suffix_old = f"_with_flags{seasonality}.nc"
    suffix_new = f"_with_flags{seasonality}_new.nc"
else:
    suffix_old = ".nc"
    suffix_new = f"_with_flags{seasonality}.nc"

for prescriptions_path in prescriptions_paths:
    # load existing flagged file if appending
    if append_flags:
        prescriptions_path = prescriptions_path.replace(".nc", f"_with_flags{seasonality}.nc")

    # load raw ds
    print("Loading prescriptions dataset:", prescriptions_path)
    prescriptions_ds = xr.load_dataset(prescriptions_path)

    # add the various flags
    # print("  Adding Hydrology rain flags...")
    # prescriptions_ds = add_hydrology_flags(prescriptions_ds, hydrology_ds, seasonal_correction=seasonal_correction)
    print("  Adding Flood GeoJSON flags...")
    prescriptions_ds = add_geojson_flood_flags(prescriptions_ds, flood_geojson)
    print("  Adding MET Office flags...")
    prescriptions_ds = add_met_flags(prescriptions_ds, met_ds, seasonal_correction=seasonal_correction)
    # print("  Adding Particulate flags...")
    # prescriptions_ds = add_particulate_flags(prescriptions_ds, particulates_ds, seasonal_correction=seasonal_correction)

    # save
    save_path = prescriptions_path.replace(suffix_old, suffix_new)
    prescriptions_ds.load()
    prescriptions_ds.to_netcdf(save_path)
    prescriptions_ds.close()
    print("Flags added to file: ", save_path)
