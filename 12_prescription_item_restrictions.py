import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from funcs import clean_prescription_items

ds = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")

# plot example time series of items with values removed
ds_clean = clean_prescription_items(ds)
fig, axes = plt.subplots(5, 5, figsize=(15, 10))
for i, ax in enumerate(axes.flatten()):
    if i < ds.dims['practice_id']:
        id = np.random.choice(ds.coords['practice_id'])
        vals = ds['items'].sel(practice_id=id).values
        mn = np.nanmean(vals)
        ax.plot(vals, 'bo')
        if id in ds_clean['practice_id'].values:
            vals = ds_clean['items'].sel(practice_id=id).values
            ax.plot(vals, 'rx', alpha=0.7)
        else:
            ax.text(0.9, 0.9, "Dropped", transform=ax.transAxes, color='red', fontsize=12, ha='right')
        ax.axhline(mn, color='blue', linestyle='--', label='Mean')
        ax.axhline(np.max([mn * 0.1, 1]), color='green', linestyle='--', label='10% of mean')
        ax.text(0.1, 0.9, f"{id}", transform=ax.transAxes)
fig.tight_layout()
fig.savefig("outputs/item_restriction_examples.png")

# plot histograms of original and cleaned values
fig2 = plt.figure(figsize=(10, 6))
orig_vals = ds['items'].values.flatten()
clean_vals = ds_clean['items'].values.flatten()
bins = np.linspace(0.1, np.nanmax(np.log1p(orig_vals)), 50)
plt.hist(np.log1p(orig_vals), bins=bins, alpha=0.5, color='blue', label=id)
plt.hist(np.log1p(clean_vals), bins=bins, alpha=0.5, color='red', label=id)
plt.xticks(np.arange(0, 10), np.expm1(np.arange(0, 10)).astype(int))
fig2.tight_layout()
fig2.savefig("outputs/item_restriction_before_and_after.png")

# plot histograms of values split by practice mean
split_thresh = 500
practice_means = ds_clean['items'].mean(dim='date')
low_vals = ds_clean['items'].isel(practice_id=practice_means < split_thresh).values.flatten()
high_vals = ds_clean['items'].isel(practice_id=practice_means >= split_thresh).values.flatten()
fig3 = plt.figure(figsize=(10, 6))
bins = np.linspace(0.1, np.nanmax(np.log1p(high_vals)), 50)
plt.hist(np.log1p(low_vals), bins=bins, alpha=0.5, color='blue', label='Low mean practices')
plt.hist(np.log1p(high_vals), bins=bins, alpha=0.5, color='red', label='High mean practices')
plt.xticks(np.arange(0, 10), np.expm1(np.arange(0, 10)).astype(int))
fig3.tight_layout()
fig3.savefig("outputs/item_restriction_groups.png")

# combined (original distribution compared to cleaned groups)
fig4 = plt.figure(figsize=(8, 5))
orig_vals = ds['items'].values.flatten()
clean_vals = ds_clean['items'].values.flatten()
bins = np.linspace(0.1, np.nanmax(np.log1p(orig_vals)), 50)
plt.hist(np.log1p(orig_vals), bins=bins, alpha=0.5, color='black', label='Raw data')
plt.hist(np.log1p(low_vals), bins=bins, alpha=0.5, color='blue', label='"small" practices')
plt.hist(np.log1p(high_vals), bins=bins, alpha=0.5, color='red', label='"large" practices')
plt.xticks(np.arange(0, 12), np.expm1(np.arange(0, 12)).astype(int))
plt.xlabel("Monthly Prescription items")
plt.ylabel("Practice-Month Count")
plt.legend()
fig4.tight_layout()
fig4.savefig("outputs/item_restriction_before_and_grouped.png", bbox_inches='tight', dpi=600)


# # F85037, F81168 (flat with sharp drop at end going to 0)
# # G82710 (flat with sharp drop at end going to 0 and low spike in middle)
# # Y02327, Y04584 (all values less than 100 but not poor quality data)
# # Y06762 (first point cut off unnecessarily by threshold of 100)
# # G82690 (need to cut off strange low outliers at end)
# # Y00840 (low values but good data)
# # Y00605 (drop at end but not a huge one, maybe should keep all for this borderline case)
# # Y00100 (good data but low values)
# # Y00148 (bad very low data)
# # Y02489 (mostly very low values but rises at end, maybe chop off the low ones)
# # C81090 (descends over time, then sudden falloff to 0)
# # Y07329 (starts very low but then normal values)
# # Y04032 (moderate for most of time then sudden rise at end)
# test_practices = ['F85037', 'F81168', 'G82710', 'Y02327', 'Y04584',
#                   'Y06762', 'G82690', 'Y00840', 'Y00605', 'Y00100',
#                   'Y00148', 'Y02489', 'C81090', 'Y07329', 'Y04032']

# # plot time series of items with values removed
# ds = ds.sel(practice_id=test_practices)
# fig, axes = plt.subplots(3, 5, figsize=(15, 10))
# for i, ax in enumerate(axes.flatten()):
#     vals = ds['items'].sel(practice_id=test_practices[i]).values
#     ax.plot(vals, 'k')
#     mn = np.nanmean(vals)
#     sd = np.nanstd(vals)
#     vals[(vals < mn - 3*sd) | (vals > mn + 3*sd)] = np.nan
#     ax.plot(vals, 'r', alpha=0.7)
#     ax.axhline(mn, color='blue', linestyle='--', label='Mean')
#     ax.axhline(mn + 3*sd, color='green', linestyle='--', label='Mean ± 3 SD')
#     ax.axhline(mn - 3*sd, color='green', linestyle='--')
#     ax.text(0.1, 0.9, f"{test_practices[i]}", transform=ax.transAxes)
# plt.tight_layout()
# plt.savefig("test.png")