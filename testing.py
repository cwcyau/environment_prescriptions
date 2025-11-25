
import xarray as xr

codes = ["02", "03", "0501", "02_03_0501"]
for code in codes:
    print('---------------', code)
    ds = xr.open_dataset(f"data/prescriptions_{code}_2010-08_2025-08_with_flags.nc")
    print([d for d in ds.data_vars if "imd_centile" in d])
    ds = ds.rename({"imd_centile": "imd_centile_values"})
    print([d for d in ds.data_vars if "imd_centile" in d])
    ds.to_netcdf(f"data/prescriptions_{code}_2010-08_2025-08_with_flags_new.nc")
    ds.close()