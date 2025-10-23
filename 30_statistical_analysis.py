import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
flagged_filename = "prescriptions_with_flood_geo.nc"

ds = xr.load_dataset(flagged_filename)

practices = ds['row_id'].values

# plot some example practice prescriptions and mark with drought/flood flags
sample_practices = practices[np.random.choice(len(practices), size=10, replace=False)]
for practice in tqdm(sample_practices, desc="Generating plots for practices", total=len(sample_practices)):
    practice_data = ds.sel(row_id=practice)
    if practice_data["items"].count() == 0:
        continue

    plt.figure(figsize=(12, 6))
    plt.plot(practice_data["date"].values, practice_data["items"].values, "k-", label="Prescriptions")

    drought_months = practice_data["date"].where(practice_data["drought"] == 1, drop=True).values
    flood_months = practice_data["date"].where(practice_data["flood"] == 1, drop=True).values
    flood_geo_months = practice_data["date"].where(practice_data["flood_geo"] == 1, drop=True).values

    drought_values = practice_data["items"].sel(date=drought_months).values
    flood_values = practice_data["items"].sel(date=flood_months).values
    flood_geo_values = practice_data["items"].sel(date=flood_geo_months).values

    plt.scatter(drought_months, drought_values, color='red', label='Drought Flag', zorder=5)
    plt.scatter(flood_months, flood_values, color='blue', label='Flood Flag', zorder=5)
    plt.scatter(flood_geo_months, flood_geo_values, color='purple', label='Flood Geo Flag', zorder=5)

    plt.title(f"Prescriptions Time Series for Practice {practice}")
    plt.xlabel("Date")
    plt.ylabel("Number of Prescriptions")
    plt.legend()
    plt.grid()
    plt.savefig(f"plots/practice_{practice}_prescriptions.png")

# explore whether prescriptions rise/fall during flagged periods
results, deltas_flood, deltas_drought, deltas_flood_geo = [], [], [], []
flood_significant_increase = 0
flood_significant_decrease = 0
flood_geo_significant_increase = 0
flood_geo_significant_decrease = 0
drought_significant_increase = 0
drought_significant_decrease = 0
min_group_n = 3  # require at least this many months in each group

for practice in tqdm(practices, desc="Statistical analysis for practices", total=len(practices)):
    station_data = ds.sel(row_id=practice)
    if station_data["items"].count() == 0:
        continue

    # arrays aligned by date
    items = station_data["items"].values
    drought_mask = station_data["drought"].values == 1
    flood_mask = station_data["flood"].values == 1
    flood_geo_mask = station_data["flood_geo"].values == 1 if "flood_geo" in station_data else np.zeros_like(items, dtype=bool)

    # test all three separately
    for kind, mask in (("drought", drought_mask), ("flood", flood_mask), ("flood_geo", flood_geo_mask)):
        flag_vals = items[mask]
        nonflag_vals = items[~mask]

        # need enough samples in both groups
        if len(flag_vals) < min_group_n or len(nonflag_vals) < min_group_n:
            continue

        mean_flag = np.nanmean(flag_vals)
        mean_non = np.nanmean(nonflag_vals)
        rel_change = (mean_flag - mean_non) / (mean_non + 1e-9)

        # nonparametric comparison (two-sided)
        try:
            stat, pval = stats.mannwhitneyu(flag_vals, nonflag_vals, alternative="two-sided")
        except Exception:
            pval = np.nan

        results.append({
            "station_id": practice,
            "kind": kind,
            "mean_flagged": float(mean_flag),
            "mean_nonflagged": float(mean_non),
            "rel_change": float(rel_change),
            "n_flagged": int(len(flag_vals)),
            "n_nonflagged": int(len(nonflag_vals)),
            "p_value": float(pval) if not np.isnan(pval) else np.nan
        })

        if kind == "flood":
            deltas_flood.append(rel_change)
        elif kind == "flood_geo":
            deltas_flood_geo.append(rel_change)
        else:
            deltas_drought.append(rel_change)

        # count significant changes
        alpha = 0.05
        if not np.isnan(pval) and pval < alpha:
            if kind == "flood":
                if rel_change > 0:
                    flood_significant_increase += 1
                elif rel_change < 0:
                    flood_significant_decrease += 1
            elif kind == "flood_geo":
                if rel_change > 0:
                    flood_geo_significant_increase += 1
                elif rel_change < 0:
                    flood_geo_significant_decrease += 1
            elif kind == "drought":
                if rel_change > 0:
                    drought_significant_increase += 1
                elif rel_change < 0:
                    drought_significant_decrease += 1

results_df = pd.DataFrame(results)
print(results_df)

summary = {
    "stations_tested_total_rows": len(results_df),
    "stations_tested_flood": int((results_df["kind"] == "flood").sum()) if not results_df.empty else 0,
    "stations_tested_drought": int((results_df["kind"] == "drought").sum()) if not results_df.empty else 0,
    "stations_tested_flood_geo": int((results_df["kind"] == "flood_geo").sum()) if not results_df.empty else 0,
    "flood_significant_increase": flood_significant_increase,
    "flood_significant_decrease": flood_significant_decrease,
    "flood_geo_significant_increase": flood_geo_significant_increase,
    "flood_geo_significant_decrease": flood_geo_significant_decrease,
    "drought_significant_increase": drought_significant_increase,
    "drought_significant_decrease": drought_significant_decrease,
    "flood_median_rel_change": float(np.nanmedian(deltas_flood)) if deltas_flood else np.nan,
    "flood_geo_median_rel_change": float(np.nanmedian(deltas_flood_geo)) if deltas_flood_geo else np.nan,
    "drought_median_rel_change": float(np.nanmedian(deltas_drought)) if deltas_drought else np.nan
}
print("Summary:", summary)

results_df.to_csv("prescription_statistical_analysis_with_flood_geo.csv", index=False)
