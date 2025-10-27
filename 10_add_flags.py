import xarray as xr
from funcs import add_hydrology_flags, add_geojson_flood_flags, load_json

# parameters
prescriptions_paths = [
    "data/prescriptions_02_03_0501_2020-09_2025-08.nc",
    "data/prescriptions_02_2020-09_2025-08.nc",
    "data/prescriptions_03_2020-09_2025-08.nc",
    "data/prescriptions_0501_2020-09_2025-08.nc"
]
hydrology_path = "data/hydrology_rainfall_stations.nc"
geojson_path = "data/Recorded_Flood_Outlines.geojson"

# load datasets
print("Loading datasets...")
hydrology_ds = xr.load_dataset(hydrology_path)
flood_geojson = load_json(geojson_path)


for prescriptions_path in prescriptions_paths:
    print("Adding flags to file:", prescriptions_path)
    # load raw ds
    prescriptions_ds = xr.load_dataset(prescriptions_path)

    # add hydrology-based flood and drought flags and geojson-based flood flags
    prescriptions_ds = add_hydrology_flags(prescriptions_ds, hydrology_ds)
    prescriptions_ds = add_geojson_flood_flags(prescriptions_ds, flood_geojson)
    # ADD MET RAIN/TEMP FLAGS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # save
    save_path = prescriptions_path.replace(".nc", "_with_flags.nc")
    prescriptions_ds.to_netcdf(save_path)
    print("Flags added to file: ", save_path)
