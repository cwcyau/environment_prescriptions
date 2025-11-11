import xarray as xr

prescriptions_paths = [
    "data/prescriptions_02_03_0501_2010-08_2025-08",
    "data/prescriptions_02_2010-08_2025-08",
    "data/prescriptions_03_2010-08_2025-08",
    "data/prescriptions_0501_2010-08_2025-08"
]

for path in prescriptions_paths:
    ds1 = xr.load_dataset(path + "_with_flags.nc")
    ds2 = xr.load_dataset(path + "_with_flags_deseasonalised.nc")
    print(f"Comparing datasets for {path}:")
    print(ds1)
    print(ds2)



# finish download
# rm -r pres/outputs/
# qsend_mixed
# qsend_bayes 02
# qsend_bayes 03
# qsend_bayes 0501
# qsend_bayes 02_03_0501