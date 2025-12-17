import numpy as np
import pandas as pd
import xarray as xr
import pymc as pm
from tqdm import tqdm
from funcs import prepare_ds

# load data
print("loading data...")
data = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")
data = prepare_ds(data,
                n_practices=None,
                standardise_items=False,
                clean_items=True,
                practice_mean_thresh=500)

# log transform "items" variable
data["items_log"] = np.log1p(data["items"])

# calculate practice_id level means of log items
print('calculating practice means...')
practice_means = data["items_log"].groupby("practice_id").mean(dim="date")

# estimate slopes of log items over date dimension for each practice
print('estimating slopes...')
slopes = []
for practice_id in tqdm(practice_means["practice_id"].values):
    practice_data = data.sel(practice_id=practice_id)
    x = np.arange(practice_data.sizes["date"])
    y = practice_data["items_log"].values
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        slopes.append((practice_id, coeffs[0]))
    else:
        slopes.append((practice_id, np.nan))

# plot histograms of means and slopes
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(practice_means.values.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Mean Log Items per Practice")

plt.subplot(1, 2, 2)
slope_values = np.array([s[1] for s in slopes if not np.isnan(s[1])])
plt.hist(slope_values, bins=50, color='green', alpha=0.7)
plt.title("Histogram of Slopes of Log Items per Practice")

plt.tight_layout()
plt.savefig("outputs/intercepts_and_slopes.png", bbox_inches='tight')

# print statistics
mean_of_means = np.nanmean(practice_means.values)
mean_of_slopes = np.nanmean(slope_values)
sd_of_means = np.nanstd(practice_means.values)
sd_of_slopes = np.nanstd(slope_values)
print(f"Mean of Mean Log Items per Practice: {mean_of_means}")
print(f"SD of Mean Log Items per Practice: {sd_of_means}")
print(f"Mean of Slopes of Log Items per Practice: {mean_of_slopes}")
print(f"SD of Slopes of Log Items per Practice: {sd_of_slopes}")
percentiles_means = np.percentile(practice_means.values, [5, 25, 50, 75, 95])
percentiles_slopes = np.percentile(slope_values, [5, 25, 50, 75, 95])
print("Percentiles of Mean Log Items per Practice (5th, 25th, 50th, 75th, 95th):", percentiles_means)
print("Percentiles of Slopes of Log Items per Practice (5th, 25th, 50th, 75th, 95th):", percentiles_slopes)