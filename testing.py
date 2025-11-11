import xarray as xr
from time import perf_counter

def method1(ds, var):
    mean = ds[var].groupby("practice_id").mean(dim="date")
    ds[var] = ds[var] - mean
    std = ds[var].groupby("practice_id").std(dim="date") + 1e-8  # prevent div by zero
    ds[var] = ds[var] / std
    return ds

def method2(ds, var):
    mean = ds[var].mean(dim="date")
    ds[var] = ds[var] - mean
    std = ds[var].std(dim="date") + 1e-8  # prevent div by zero
    ds[var] = ds[var] / std
    return ds

n = 5
z = 0.0
for _ in range(n):
    ds = xr.load_dataset("data/prescriptions_02_2010-08_2025-08_with_flags.nc")
    var = "met_rain_values"

    z0 = perf_counter()
    ds2 = method2(ds, var)
    z1 = perf_counter()
    z += z1 - z0
print(f"Standardisation took {z / n:.2f} seconds.")