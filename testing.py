import xarray as xr
import numpy as np
from pathlib import Path

# Folder containing the updated NetCDF files
orig_nc_files = ["data/prescriptions_02_03_0501_2020-09_2025-08.nc",
                 "data/prescriptions_02_2020-09_2025-08.nc",
                 "data/prescriptions_03_2020-09_2025-08.nc",
                 "data/prescriptions_0501_2020-09_2025-08.nc"]
new_nc_files = ["data/prescriptions_02_03_0501_2010-08_2025-08.nc",
                "data/prescriptions_02_2010-08_2025-08.nc",
                "data/prescriptions_03_2010-08_2025-08.nc",
                "data/prescriptions_0501_2010-08_2025-08.nc"]

for i in range(2, 4):
    orig_ds = xr.load_dataset(orig_nc_files[i])
    new_ds = xr.load_dataset(new_nc_files[i])

    # fix the lat/lon dimensions in the new dataset
    if new_ds['latitude'].dims != ('practice_id',):
        lat_1d = np.array([new_ds["latitude"].sel(practice_id=pid).values[~np.isnan(new_ds["latitude"].sel(practice_id=pid).values)][0]
                        for pid in new_ds["practice_id"].values])
    else:
        lat_1d = new_ds['latitude'].values
    if new_ds['longitude'].dims != ('practice_id',):
        lon_1d = np.array([new_ds["longitude"].sel(practice_id=pid).values[~np.isnan(new_ds["longitude"].sel(practice_id=pid).values)][0]
                           for pid in new_ds["practice_id"].values])
    else:
        lon_1d = new_ds['longitude'].values
    new_ds = new_ds.assign_coords({
        "latitude": ("practice_id", lat_1d),
        "longitude": ("practice_id", lon_1d)
    })
    new_ds.to_netcdf(new_nc_files[i])
    print(f"Updated coordinates for {new_nc_files[i]} and saved.")
    print(new_ds['latitude'])
    print(new_ds['longitude'])
    
    print(f"Comparing {orig_nc_files[i]} and {new_nc_files[i]}")
    print("  Original dataset variables:", list(orig_ds.data_vars))
    print("  New dataset variables:", list(new_ds.data_vars))
    print("  Original dataset time range:", str(orig_ds['date'].values[0]), "to", str(orig_ds['date'].values[-1]))
    print("  New dataset time range:", str(new_ds['date'].values[0]), "to", str(new_ds['date'].values[-1]))
    print("  Original practice count:", orig_ds['practice_id'].size)
    print("  New practice count:", new_ds['practice_id'].size)
    print("  Number of NaN lats/lons in original dataset:", np.isnan(orig_ds['latitude']).sum().item(), "/", orig_ds['latitude'].size)
    print("  Number of NaN lats/lons in new dataset:", np.isnan(new_ds['latitude']).sum().item(), "/", new_ds['latitude'].size)
