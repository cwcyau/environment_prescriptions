import json
import os
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from tqdm import tqdm

nc_file_path = "prescriptions_0501_02_03_2020-09_2025-08.nc"
min_group_n = 3

# load dataset
ds = xr.load_dataset(nc_file_path)

# month-normalization per practice
month_numbers = ds["date"].dt.month
def month_center_per_practice(x):
    return x.groupby(month_numbers).apply(lambda m: m - m.mean(dim="date"))
items_norm = month_center_per_practice(ds["items"])

# loop through each practice and analyse flagged vs non-flagged prescription counts
results = []
deltas = {"flood":[], "flood_geo":[], "drought":[]}
signif_counts = {k:{"up":0,"down":0} for k in deltas}
for practice in tqdm(ds['row_id'].values, desc="Per-practice analysis"):
    practice_data = ds.sel(row_id=practice)
    if practice_data["items"].count() == 0:
        continue

    items = items_norm.sel(row_id=practice).values
    masks = {k: practice_data[k].values==1 if k in practice_data else np.zeros_like(items, bool) for k in deltas}

    for kind, mask in masks.items():
        flag_vals = items[mask]
        nonflag_vals = items[~mask]

        if len(flag_vals)<min_group_n or len(nonflag_vals)<min_group_n:
            continue

        mean_flag = np.nanmean(flag_vals)
        mean_non = np.nanmean(nonflag_vals)
        pooled_sd = np.nanstd(np.concatenate([flag_vals, nonflag_vals])) + 1e-9
        std_effect = (mean_flag - mean_non) / pooled_sd

        try:
            _, pval = stats.mannwhitneyu(flag_vals, nonflag_vals, alternative="two-sided")
        except:
            pval = np.nan

        results.append({
            "practice_id": practice,
            "kind": kind,
            "mean_flagged": float(mean_flag),
            "mean_nonflagged": float(mean_non),
            "std_effect_size": float(std_effect),
            "n_flagged": int(len(flag_vals)),
            "n_nonflagged": int(len(nonflag_vals)),
            "p_value": float(pval)
        })

        deltas[kind].append(std_effect)
        alpha = 0.05
        if not np.isnan(pval) and pval<alpha:
            if std_effect > 0:
                signif_counts[kind]["up"] += 1
            elif std_effect < 0:
                signif_counts[kind]["down"] += 1

# print and save outputs
results_df = pd.DataFrame(results)
save_filename = nc_file_path.replace(".nc", "_per_practice_analysis.csv")
results_df.to_csv(save_filename, index=False)
summary = {
    "median_sd_effect": {k: float(np.nanmedian(v)) for k, v in deltas.items()},
    "signif_counts": signif_counts
}
print(summary)
save_path = "outputs/" + nc_file_path.replace(".nc", "") + "/per_practice_summary.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    json.dump(summary, f, indent=2)
