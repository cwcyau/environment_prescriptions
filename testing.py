import numpy as np
import pandas as pd
import xarray as xr
from funcs import load_json, plot_regions_map

ds = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")

print(np.unique(ds["region"].values))
print(len(ds['practice_id'].values))

ds = ds.isel(practice_id=ds["region"] != "Wales")

print(np.unique(ds["region"].values))
print(len(ds['practice_id'].values))