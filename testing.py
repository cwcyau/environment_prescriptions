import xarray as xr
import matplotlib.pyplot as plt
from funcs import remove_seasonal_effects

prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08",
]
suffix = "_with_flags.nc"

for path in prescriptions_paths:
    ds = xr.load_dataset(path + suffix)
    print(ds.data_vars)


