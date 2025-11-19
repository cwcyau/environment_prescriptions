import xarray as xr

ds = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")
ds2 = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags_new.nc")

print(ds)
print()
print('---------------')
print()
print(ds2)