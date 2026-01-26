import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from funcs import prepare_ds

# load data
print("loading data...")
data = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")
data = prepare_ds(data,
                n_practices=None,
                standardise_items=False,
                clean_items=True)

# log transform "items" variable
data["items_log"] = np.log1p(data["items"])

# exclude practices with <20 months of data
print('filtering practices with <20 months of data...')
valid_practices = []
for practice_id in data["practice_id"].values:
    practice_data = data.sel(practice_id=practice_id)
    n_months = np.sum(~np.isnan(practice_data["items_log"].values))
    if n_months >= 20:
        valid_practices.append(practice_id)
data = data.sel(practice_id=valid_practices)

# calculate practice_id level means of log items
print('calculating practice means...')
practice_means = data["items_log"].groupby("practice_id").mean(dim="date")
practice_means_raw = data["items"].groupby("practice_id").mean(dim="date")

# estimate slopes of log items over date dimension for each practice
slopes = []
for practice_id in tqdm(practice_means["practice_id"].values,
                        desc="estimating slopes of log items"):
    practice_data = data.sel(practice_id=practice_id)
    x = np.arange(practice_data.sizes["date"])
    y = practice_data["items_log"].values
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        slopes.append((practice_id, coeffs[0]))
    else:
        slopes.append((practice_id, np.nan))

# estimate slopes of raw items over date dimension for each practice
slopes_raw = []
for practice_id in tqdm(practice_means["practice_id"].values,
                        desc="estimating slopes of raw items"):
    practice_data = data.sel(practice_id=practice_id)
    x = np.arange(practice_data.sizes["date"])
    y = practice_data["items"].values
    mask = ~np.isnan(y)
    if np.sum(mask) > 1:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        slopes_raw.append((practice_id, coeffs[0]))
    else:
        slopes_raw.append((practice_id, np.nan))

# plot histograms of practice means and slopes
plt.figure(figsize=(12, 5))
plt.subplot(2, 2, 1)
plt.hist(practice_means.values.flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Histogram of Mean Log Items per Practice")

plt.subplot(2, 2, 2)
slope_values = np.array([s[1] for s in slopes if not np.isnan(s[1])])
low = np.percentile(slope_values, 1)
high = np.percentile(slope_values, 99)
bins = np.linspace(low, high, 50)
plt.hist(slope_values, bins=bins, color='green', alpha=0.7)
plt.xlim(low, high)
plt.title("Histogram of Slopes of Log Items per Practice")

plt.subplot(2, 2, 3)
plt.hist(practice_means_raw.values.flatten(), bins=50, color='orange', alpha=0.7)
plt.title("Histogram of Mean Raw Items per Practice")

plt.subplot(2, 2, 4)
slope_values_raw = np.array([s[1] for s in slopes_raw if not np.isnan(s[1])])
low_raw = np.percentile(slope_values_raw, 1)
high_raw = np.percentile(slope_values_raw, 99)
bins_raw = np.linspace(low_raw, high_raw, 50)
plt.hist(slope_values_raw, bins=bins_raw, color='red', alpha=0.7)
plt.xlim(low_raw, high_raw)
plt.title("Histogram of Slopes of Raw Items per Practice")

plt.tight_layout()
plt.savefig("outputs/intercepts_and_slopes.png", bbox_inches='tight')

# print statistics
mean_of_means = np.nanmean(practice_means.values)
mean_of_slopes = np.nanmean(slope_values)
sd_of_means = np.nanstd(practice_means.values)
sd_of_slopes = np.nanstd(slope_values)
mean_of_means_raw = np.nanmean(practice_means_raw.values)
mean_of_slopes_raw = np.nanmean(slope_values_raw)
sd_of_means_raw = np.nanstd(practice_means_raw.values)
sd_of_slopes_raw = np.nanstd(slope_values_raw)
print(f"Mean of Mean Log Items per Practice: {mean_of_means}")
print(f"SD of Mean Log Items per Practice: {sd_of_means}")
print(f"Mean of Slopes of Log Items per Practice: {mean_of_slopes}")
print(f"SD of Slopes of Log Items per Practice: {sd_of_slopes}")
print(f"Mean of Mean Raw Items per Practice: {mean_of_means_raw}")
print(f"SD of Mean Raw Items per Practice: {sd_of_means_raw}")
print(f"Mean of Slopes of Raw Items per Practice: {mean_of_slopes_raw}")
print(f"SD of Slopes of Raw Items per Practice: {sd_of_slopes_raw}")
percentiles_means = np.percentile(practice_means.values, [5, 25, 50, 75, 95])
percentiles_slopes = np.percentile(slope_values, [5, 25, 50, 75, 95])
percentiles_means_raw = np.percentile(practice_means_raw.values, [5, 25, 50, 75, 95])
percentiles_slopes_raw = np.percentile(slope_values_raw, [5, 25, 50, 75, 95])
print("Percentiles of Mean Log Items per Practice (5th, 25th, 50th, 75th, 95th):", percentiles_means)
print("Percentiles of Slopes of Log Items per Practice (5th, 25th, 50th, 75th, 95th):", percentiles_slopes)
print("Percentiles of Mean Raw Items per Practice (5th, 25th, 50th, 75th, 95th):", percentiles_means_raw)
print("Percentiles of Slopes of Raw Items per Practice (5th, 25th, 50th, 75th, 95th):", percentiles_slopes_raw)