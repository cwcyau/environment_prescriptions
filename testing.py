import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")

vals = ds['aqrean_daqi_nitrogen_dioxide_values']

print(vals.min().values, vals.max().values, vals.mean().values, vals.std().values)

print(ds.data_vars)