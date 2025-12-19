import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from funcs import prepare_ds

# load data
data = xr.load_dataset("./data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")

standardise_items = False
clean_items = True
adjust_predictors = 'z-global'
deseasonalise_predictors = False
practice_mean_thresh = 500

data = prepare_ds(
    data,
    standardise_items=standardise_items,
    clean_items=clean_items,
    adjust_predictors=adjust_predictors,
    deseasonalise_predictors=deseasonalise_predictors,
    practice_mean_thresh=practice_mean_thresh
)

data['items'] = xr.where(data['items'] <= 0, np.nan, data['items'])

# drop all dates where any variable is NaN, per practice
def clean_practice(ds, practice_id):
    practice_data = ds.sel(practice_id=practice_id)
    # drop dates where any variable is NaN
    practice_data = practice_data.dropna(dim='date', how='any')
    return practice_data

cleaned_practices = []
for pid in tqdm(data['practice_id'].values):
    cleaned_practices.append(clean_practice(data, pid))

# combine back into single dataset
data_cleaned = xr.concat(cleaned_practices, dim='practice_id')
data_cleaned['practice_id'] = data['practice_id']  # restore practice_ids

# number of observations per practice
obs_per_practice = data_cleaned['items'].notnull().sum(dim='date')

# plot histogram
plt.figure(figsize=(8, 5))
plt.hist(obs_per_practice.values, bins=30, edgecolor='black')
plt.xlabel('Number of observations per practice')
plt.ylabel('Number of practices')
plt.title('Histogram of observations per practice after cleaning')
plt.savefig('testing.png', bbox_inches='tight')
plt.show()
