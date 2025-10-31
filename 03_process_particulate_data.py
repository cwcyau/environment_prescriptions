import numpy as np
import xarray as xr
from tqdm import tqdm

# parameters
monthly_agg = 'max'

particulate_paths = [
    'data/aqrean/*_co_*.nc',
    'data/aqrean/*level_daqi_2*.nc',
    'data/aqrean/*_no_2*.nc',
    'data/aqrean/*_no2_2*.nc',
    'data/aqrean/*_no2_daqi_*.nc',
    'data/aqrean/*_nox_2*.nc',
    'data/aqrean/*_o3_2*.nc',
    'data/aqrean/*_o3_daqi*.nc',
    'data/aqrean/*_pm2p5_2*.nc',
    'data/aqrean/*_pm2p5_daqi_*.nc',
    'data/aqrean/*_pm10_2*.nc',
    'data/aqrean/*_pm10_daqi_*.nc',
    'data/aqrean/*_so2_2*.nc',
    'data/aqrean/*_so2_daqi_*.nc'
]

prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08.nc",
    "data/prescriptions_02_2010-08_2025-08.nc",
    "data/prescriptions_03_2010-08_2025-08.nc",
    "data/prescriptions_0501_2010-08_2025-08.nc"
]

# extract practice locations
practice_ids, practice_lats, practice_lons = np.array([]), np.array([]), np.array([])
print("Extracting all practice locations from prescriptions data...")
for path in prescriptions_paths:
    ds = xr.open_dataset(path)
    practice_ids = np.append(practice_ids, ds['practice_id'].values)
    practice_lats = np.append(practice_lats, ds['latitude'].values)
    practice_lons = np.append(practice_lons, ds['longitude'].values)
    ds.close()

practice_ids, inds = np.unique(practice_ids, return_index=True)
practice_lats = practice_lats[inds]
practice_lons = practice_lons[inds] + 360  # stored as 360 +/- actual longitude

# extract all monthly dates from particulate data files
print("Extracting all monthly dates from particulate data files...")
all_months = []
for path in particulate_paths:
    ds_tmp = xr.open_mfdataset(path, chunks='auto')
    # take first variable as representative
    var_name = list(ds_tmp.data_vars)[0]
    ds_mon_tmp = ds_tmp[var_name].resample(time='MS').max()  # monthly start
    all_months.append(ds_mon_tmp.time.values)
    ds_tmp.close()

import pandas as pd
master_time = np.unique(np.concatenate(all_months))  # union of all monthly timestamps
master_time = pd.to_datetime(master_time)  # ensure datetime type
# get monthly maximums at practice locations
ds_out = xr.Dataset(coords={
    "date": ("date", master_time),
    "practice_id": ("practice_id", practice_ids),
    "latitude": ("practice_id", practice_lats),
    "longitude": ("practice_id", practice_lons),
})
for path in tqdm(particulate_paths,
                 desc="Processing particulate data files",
                 total=len(particulate_paths)):
    ds = xr.open_mfdataset(path, chunks='auto')

    # find the relevant variable
    if any('daily_air_quality_index' in v for v in ds.data_vars):
        var_name = [v for v in ds.data_vars if 'daily_air_quality_index' in v][0]
        suffix = '_daqi'
    else:
        var_name = [v for v in ds.data_vars if 'mass_concentration' in v][0]
        suffix = ''

    # assign simplified name
    if var_name == 'daily_air_quality_index':
        v_out = 'daqi'
    else:
        v_out = var_name.split('of_')[-1].replace('_in_air', '').replace('_dry_aerosol', '') + suffix

    # resample to monthly maxima and align with master time index
    ds_mon = getattr(ds[var_name].resample(time='MS'), monthly_agg)()
    ds_mon = ds_mon.reindex(time=master_time)

    # select nearest grid point for all practices
    data_sel = ds_mon.sel(latitude=xr.DataArray(practice_lats, dims="practice_id"),
                          longitude=xr.DataArray(practice_lons, dims="practice_id"),
                          method="nearest")

    # transpose to (date, practice_id)
    data_sel = data_sel.transpose("time", "practice_id")

    # add to output dataset
    ds_out[v_out] = ("date", "practice_id"), data_sel.data.compute()  # compute to store as concrete array

    ds.close()

# save output dataset
output_path = f"data/particulates_by_practice_monthly_{monthly_agg}.nc"
print(f"\nSaving dataset to {output_path}")
ds_out.to_netcdf(output_path)
print("Done.")
