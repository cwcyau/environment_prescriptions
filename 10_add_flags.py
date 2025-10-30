import xarray as xr
from funcs import add_hydrology_flags, add_geojson_flood_flags, add_met_flags, load_json

# parameters
prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08.nc",
    "data/prescriptions_02_2010-08_2025-08.nc",
    "data/prescriptions_03_2010-08_2025-08.nc",
    "data/prescriptions_0501_2010-08_2025-08.nc"
]
hydrology_path = "data/hydrology_rainfall_stations.nc"
geojson_path = "data/Recorded_Flood_Outlines.geojson"
met_path = "data/met_office_stations.nc"

# load datasets
print("Loading datasets...")
hydrology_ds = xr.load_dataset(hydrology_path)
flood_geojson = load_json(geojson_path)
met_ds = xr.load_dataset(met_path)
print("Datasets loaded.")

for prescriptions_path in prescriptions_paths:
    print("Adding flags to file:", prescriptions_path)
    # load raw ds
    prescriptions_ds = xr.load_dataset(prescriptions_path)

    # add the various flags
    print("  Adding Hydrology rain flags...")
    prescriptions_ds = add_hydrology_flags(prescriptions_ds, hydrology_ds)
    print("  Adding Flood GeoJSON flags...")
    prescriptions_ds = add_geojson_flood_flags(prescriptions_ds, flood_geojson)
    print("  Adding MET Office flags...")
    prescriptions_ds = add_met_flags(prescriptions_ds, met_ds)

    # save
    save_path = prescriptions_path.replace(".nc", "_with_flags.nc")
    prescriptions_ds.to_netcdf(save_path)
    print("Flags added to file: ", save_path)
