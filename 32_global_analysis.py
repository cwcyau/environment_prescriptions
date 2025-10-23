import json
import os
import numpy as np
import xarray as xr
from scipy import stats

nc_file_path = "prescriptions_0501_02_03_2020-09_2025-08.nc"
min_group_n = 3
jitter_eps = 1e-9  # tiny noise to break exact ties

ds = xr.load_dataset(nc_file_path)

# month-normalization per practice
month_numbers = ds["date"].dt.month
def month_center_per_practice(x):
    return x.groupby(month_numbers).map(lambda m: m - m.mean(dim="date"))
items_norm = month_center_per_practice(ds["items"])

# scale each practice to avoid domination by high-prescription practices
items_scaled = items_norm / (items_norm.std(dim="date") + 1e-9)

# flatten arrays for global testing
items_flat = items_scaled.values.flatten()
masks = {k: ds[k].values.flatten() == 1 for k in ["flood","flood_geo","drought"]}

global_results = {}
for kind, mask in masks.items():
    flagged = items_flat[mask]
    nonflagged = items_flat[~mask]

    if len(flagged) < min_group_n or len(nonflagged) < min_group_n:
        continue

    # add tiny jitter to avoid ties
    flagged_j = flagged + np.random.normal(0, jitter_eps, flagged.shape)
    nonflagged_j = nonflagged + np.random.normal(0, jitter_eps, nonflagged.shape)

    mean_flag = np.nanmean(flagged_j)
    mean_non = np.nanmean(nonflagged_j)

    # pooled standard deviation for standardized effect size
    pooled_sd = np.nanstd(np.concatenate([flagged_j, nonflagged_j])) + 1e-9
    std_effect = (mean_flag - mean_non) / pooled_sd

    # statistical testing
    try:
        _, pval = stats.mannwhitneyu(flagged_j, nonflagged_j, alternative="two-sided", method="asymptotic")
    except:
        pval = np.nan

    global_results[kind] = {
        "mean_flagged": float(mean_flag),
        "mean_nonflagged": float(mean_non),
        "std_effect_size": float(std_effect),
        "n_flagged": len(flagged),
        "n_nonflagged": len(nonflagged),
        "p_value": float(pval)
    }

# print results
print("Global analysis results:")
for k, v in global_results.items():
    print(f"  {k}: {v}")

# save output
save_path = "outputs/" + nc_file_path.replace(".nc", "") + "/global_analysis_results.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    json.dump(global_results, f, indent=2)
