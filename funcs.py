import os, json, requests
import numpy as np
import xarray as xr
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pyproj import Transformer
from tqdm import tqdm
from scipy import stats
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

MIN_GROUP_N = 3  # minimum number of flags against practice for analysis

# DATA FUNCTIONS ==================================================================================

def add_hydrology_flags(prescriptions_ds, hydrology_ds,
                        observed_property="rain", agg="sum",
                        flag_types=["high", "low", "median"]):
    """
    Add flags to the prescriptions dataset based on hydrology station data.

    prescriptions_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    hydrology_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    observed_property: str, the property to observe (e.g. "rain")
    agg: str, the aggregation method to use (e.g. "sum")
    flag_types: list of str, the types of flags to create (e.g. ["high", "low", "median"])
    """
    # get practice and station locations
    lat_p = prescriptions_ds.latitude.values
    lon_p = prescriptions_ds.longitude.values
    lat_s = hydrology_ds.latitude.values
    lon_s = hydrology_ds.longitude.values
    sid_s = hydrology_ds.station_id.values

    # find nearest station for each practice
    tree = cKDTree(np.column_stack([lat_s, lon_s]))
    _, nearest_idx = tree.query(np.column_stack([lat_p, lon_p]))
    nearest_stations = sid_s[nearest_idx]
    unique_stations = np.unique(nearest_stations)

    # prepare arrays for flags
    pres_datetimes = pd.to_datetime(prescriptions_ds.date.values)
    pres_months = pres_datetimes.to_period("M")
    daily_rain_datetimes = pd.to_datetime(hydrology_ds.date.values)
    daily_rain_readings = hydrology_ds['rainfall']
    outputs = {}
    for flag_type in flag_types:
        outputs[flag_type] = np.full((len(pres_datetimes), len(lat_p)), np.nan, dtype=np.float32)
    outputs["values"] = np.full((len(pres_datetimes), len(lat_p)), np.nan, dtype=np.float32)

    # get flags for each unique station
    for station_id in tqdm(unique_stations,
                           desc="      Fetching station flags",
                           total=len(unique_stations)):
        # get the rainfall data for this station
        daily_station_readings = daily_rain_readings.sel(station_id=station_id).values

        # aggregate the data to monthly totals
        monthly_rain_datetimes, monthly_rain_readings = aggregate_monthly(daily_rain_datetimes,
                                                                          daily_station_readings,
                                                                          agg)

        # remove seasonal effects from readings
        monthly_z_values = remove_seasonal_effects(monthly_rain_datetimes,
                                                   monthly_rain_readings)

        # generate flags
        mask = nearest_stations == station_id
        for flag_type in flag_types:
            flags_temp = generate_flags(monthly_rain_datetimes,
                                        monthly_z_values,
                                        flag_type,
                                        pres_months)
            outputs[flag_type][:, mask] = flags_temp[:, None]
        
        # save aggregated values for nc_months
        agg_months = pd.to_datetime(monthly_rain_datetimes).to_period("M")
        values_series = pd.Series(monthly_rain_readings, index=agg_months)
        aligned_values = values_series.reindex(pres_months, fill_value=np.nan).values
        outputs["values"][:, mask] = aligned_values[:, None]
        
    # create arrays for new flags
    for flag_type in flag_types:
        prescriptions_ds[f"hydro_{observed_property}_{flag_type}"] = (("date", "practice_id"),
                                                                      outputs[flag_type])
    prescriptions_ds[f"hydro_{observed_property}_values"] = (("date", "practice_id"),
                                                             outputs["values"])

    return prescriptions_ds

def add_geojson_flood_flags(prescriptions_ds, geojson_features,
                            search_radius_m=5000, simplify_tol=50):
    """
    Add flood flags to the dataset based on geojson polygons.
    """
    practice_ids = prescriptions_ds['practice_id'].values
    lat_vec = prescriptions_ds.coords['latitude'].values
    lon_vec = prescriptions_ds.coords['longitude'].values

    # convert lat/lon to projected coordinates (EPSG:27700, British National Grid)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    xs, ys = transformer.transform(lon_vec, lat_vec)

    # preprocess flood polygons
    geoms = []
    months = []
    for f in tqdm(geojson_features,
                  desc="Processing flood polygons",
                  total=len(geojson_features)):
        start = pd.to_datetime(f["properties"].get("start_date"),
                               errors="coerce")
        end = pd.to_datetime(f["properties"].get("end_date"),
                             errors="coerce")
        if pd.isna(start) or start.year < 2020:
            continue
        geom = shape(f["geometry"])
        geom = geom.simplify(simplify_tol, preserve_topology=True)
        geom = geom.buffer(search_radius_m)
        end = end or start
        # assign polygon to each month it spans
        for m in pd.period_range(start.to_period("M"),
                                 end.to_period("M"), freq="M"):
            geoms.append(geom)
            months.append(m)

    geom_months = np.array(months)

    # generate flooding flags
    flood_flags = np.full(prescriptions_ds['items'].shape, np.nan, dtype=np.float32)
    for t_idx, date in enumerate(tqdm(prescriptions_ds['date'].values,
                                      desc="      Flagging flood months",
                                      total=len(prescriptions_ds['date'].values))):
        date_period = pd.Period(pd.to_datetime(date), freq="M")
        idxs = np.where(geom_months == date_period)[0]
        if len(idxs) == 0:
            flood_flags[t_idx, :] = 0.0
            continue
        month_geoms = [geoms[i] for i in idxs]
        month_tree = STRtree(month_geoms)

        for i, (x, y) in enumerate(zip(xs, ys)):
            pt = Point(x, y)
            nearby_idxs = month_tree.query(pt)
            if any(month_geoms[j].intersects(pt) for j in nearby_idxs):
                flood_flags[t_idx, i] = 1.0
            else:
                flood_flags[t_idx, i] = 0.0

    prescriptions_ds["flood"] = (("date", "practice_id"), flood_flags)
    return prescriptions_ds

def add_met_flags(prescriptions_ds, met_ds,
                  observed_properties=["tmax", "rain"],
                  flag_types=["high", "low", "median"]):
    """
    Add flags to the prescriptions dataset based on MET Office station data.

    prescriptions_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    met_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    observed_property: str, the property to observe (e.g. "rain")
    agg: str, the aggregation method to use (e.g. "sum")
    flag_types: list of str, the types of flags to create (e.g. ["high", "low", "median"])
    """
    # get practice and station locations
    lat_p = prescriptions_ds.latitude.values
    lon_p = prescriptions_ds.longitude.values
    lat_s = met_ds.latitude.values
    lon_s = met_ds.longitude.values
    sid_s = met_ds.station_id.values

    # find nearest station for each practice
    tree = cKDTree(np.column_stack([lat_s, lon_s]))
    _, nearest_idx = tree.query(np.column_stack([lat_p, lon_p]))
    nearest_stations = sid_s[nearest_idx]
    unique_stations = np.unique(nearest_stations)

    # prepare arrays for flags
    pres_datetimes = pd.to_datetime(prescriptions_ds.date.values)
    pres_months = pres_datetimes.to_period("M")
    for observed_property in observed_properties:
        print("    Adding MET flags for", observed_property)
        outputs = {}
        for flag_type in flag_types:
            outputs[flag_type] = np.full((len(pres_datetimes), len(lat_p)), np.nan, dtype=np.float32)
        outputs["values"] = np.full((len(pres_datetimes), len(lat_p)), np.nan, dtype=np.float32)

        # get flags for each unique station
        for station_id in tqdm(unique_stations, desc="      Fetching station flags"):
            # get observed property for this station
            values = met_ds.sel(station_id=station_id)[observed_property].values
            met_datetimes = pd.to_datetime(met_ds.sel(station_id=station_id).date.values)

            # remove seasonal effects from readings
            z_values = remove_seasonal_effects(met_datetimes, values)

            # generate flags
            nc_months = pres_datetimes.to_period("M")
            mask = nearest_stations == station_id
            for flag_type in flag_types:
                flags_temp = generate_flags(met_datetimes, z_values, flag_type, nc_months)
                outputs[flag_type][:, mask] = flags_temp[:, None]
                    
            # save values for nc file time period
            values_series = pd.Series(values, index=met_datetimes)
            aligned_values = values_series.reindex(pres_months, fill_value=np.nan).values
            outputs["values"][:, mask] = aligned_values[:, None]

        # create arrays for new flags
        for flag_type in flag_types:
            prescriptions_ds[f"met_{observed_property}_{flag_type}"] = (("date", "practice_id"),
                                                                        outputs[flag_type])
        prescriptions_ds[f"met_{observed_property}_values"] = (("date", "practice_id"),
                                                                outputs["values"])
    return prescriptions_ds


# INSPECTION FUNCTIONS ============================================================================
def plot_practices(ds, nc_file_path, sample_size=30, seed=None):
    """
    Plot prescriptions for a sample of practices.
    """
    practices = ds['practice_id'].values
    if seed is not None:
        np.random.seed(seed)
    sample_practices = practices[np.random.choice(len(practices),
                                                size=sample_size,
                                                replace=False)]

    # plot prescriptions time series with flags for a sample of practices
    for practice in tqdm(sample_practices, desc="Generating plots"):
        practice_data = ds.sel(practice_id=practice)
        if practice_data["items"].count() == 0:
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(practice_data["date"].values,
                practice_data["items"].values,
                "ko-", markersize=10, alpha=0.2,
                label="Prescriptions")

        plot_args = (("drought","red","v"),
                    ("flood","blue","v"),
                    ("flood_geo","green","^"))
        for kind, color, marker in plot_args:
            if kind not in practice_data:
                continue
            flagged_dates = practice_data["date"].where(practice_data[kind] == 1,
                                                        drop=True).values
            flagged_values = practice_data["items"].sel(date=flagged_dates).values
            plt.scatter(flagged_dates, flagged_values,
                        c=color, s=60, marker=marker, alpha=0.7,
                        label=f"{kind.title()} Flag", zorder=5)

        plt.title(f"Prescriptions Time Series for Practice {practice}")
        plt.xlabel("Date")
        plt.ylabel("Number of Prescriptions")
        plt.legend()
        plt.grid()

        # save
        save_path = "outputs/" + nc_file_path.replace(".nc", "") + "/" + practice + ".png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# ANALYSIS FUNCTIONS ==============================================================================
def mixed_effects_model(ds, flag_types, nc_file_path):
    """
    Fit a mixed-effects model to the prescription data.
    """
    # prepare dataframe
    df_list = []
    for practice in tqdm(ds['practice_id'].values, desc="Preparing data"):
        practice_data = ds.sel(practice_id=practice).to_dataframe().reset_index()
        practice_data['practice_id'] = practice
        for flag in flag_types:
            practice_data[flag] = ds[flag].sel(practice_id=practice).values
        df_list.append(practice_data)

    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=['items'])

    # convert flags to integers
    df[flag_types] = df[flag_types].astype(int)

    # remove month-of-year effects globally
    df['items_adj'] = df['items'] - df.groupby(df['date'].dt.month)['items'].transform('mean')

    # optionally scale and clip extreme values
    sd_global = df['items_adj'].std()
    df['items_adj'] = df['items_adj'] / (sd_global + 1e-9)
    df['items_adj'] = df['items_adj'].clip(-5, 5)

    # fit mixed-effects model
    model_formula = 'items_adj ~ ' + ' + '.join(flag_types) + ' - 1'
    md = smf.mixedlm(model_formula, df, groups=df['practice_id'])
    mdf = md.fit(method='lbfgs', reml=True)

    # print and save results
    summary = mdf.summary()
    print(summary)

    save_path = os.path.join("outputs", nc_file_path.replace(".nc", ""), "mixed_effects_result.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(str(summary))

def global_analysis(ds, flag_types, nc_file_path,
                    min_group_n=MIN_GROUP_N, jitter_eps=1e-6,
                    n_boot=1000, n_perm=1000, seed=None):
    """
    Perform global analysis on the prescription data.
    """
    # month-normalization per practice
    month_numbers = ds["date"].dt.month
    def month_center_per_practice(x):
        return x.groupby(month_numbers).map(lambda m: m - m.mean(dim="date"))
    items_norm = month_center_per_practice(ds["items"])

    # scale each practice to avoid domination by high-prescription practices
    items_scaled = items_norm / (items_norm.std(dim="date") + 1e-9)

    # flatten arrays for global testing
    items_flat = items_scaled.values.flatten()

    global_results = {}
    for kind in tqdm(flag_types, desc="Global analysis", total=len(flag_types)):
        if kind not in ds:
            continue
        flag_arr = ds[kind].values.flatten().astype(float)
        is_nan = np.isnan(flag_arr)
        is_one = flag_arr == 1
        is_zero = flag_arr == 0

        # flagged: explicit ones; nonflagged: explicit zeros only (exclude NaNs)
        flagged = items_flat[is_one]
        nonflagged = items_flat[is_zero & ~is_nan]

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
            _, pval = stats.mannwhitneyu(flagged_j, nonflagged_j,
                                        alternative="two-sided",
                                        method="asymptotic")
        except:
            pval = np.nan

        # bootstrap using helper (for both mean diff and std effect)
        boot_res = bootstrap_effects(flagged_j, nonflagged_j,
                                     n_boot=n_boot, jitter_eps=jitter_eps,
                                     min_group_n=min_group_n, seed=seed)
        if boot_res is not None:
            se_lo, se_hi = boot_res.get("std_effect_ci", [None, None])
            md_lo, md_hi = boot_res.get("mean_diff_ci", [None, None])
            boot_pval = boot_res.get("boot_pval")
            mean_diff_boot_p = boot_res.get("mean_diff_boot_p")
        else:
            se_lo = se_hi = md_lo = md_hi = None
            boot_pval = None
            mean_diff_boot_p = None

        # permutation p-value for mean difference (and optionally std effect)
        try:
            mean_diff_perm_p = permutation_pvalue(lambda a, b: float(np.nanmean(a) - np.nanmean(b)),
                                                  flagged_j, nonflagged_j, n_perm=n_perm, seed=seed)
        except Exception:
            mean_diff_perm_p = None
        try:
            std_effect_perm_p = permutation_pvalue(lambda a, b: float((np.nanmean(a) - np.nanmean(b)) /
                                                                      (np.nanstd(np.concatenate([a, b])) + 1e-9)),
                                                   flagged_j, nonflagged_j, n_perm=n_perm, seed=seed)
        except Exception:
            std_effect_perm_p = None

        global_results[kind] = {
            "mean_flagged": float(mean_flag),
            "mean_nonflagged": float(mean_non),
            "mean_diff_size": float(mean_flag - mean_non),
            "mean_diff_ci": [None if md_lo is None else float(md_lo),
                             None if md_hi is None else float(md_hi)],
            "mean_diff_boot_p": mean_diff_boot_p,
            "mean_diff_perm_p": mean_diff_perm_p,
            "mannwhitney_p": float(pval),
            "std_effect_size": float(std_effect),
            "std_effect_ci": [None if se_lo is None else float(se_lo),
                              None if se_hi is None else float(se_hi)],
            "std_effect_boot_p": boot_pval,
            "std_effect_perm_p": std_effect_perm_p,
            "n_flagged": len(flagged),
            "n_nonflagged": len(nonflagged),
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

def practice_analysis(ds, flag_types, nc_file_path,
                      min_group_n=MIN_GROUP_N, n_boot=500,
                      n_perm=1000, seed=None):
    # month-normalization per practice
    month_numbers = ds["date"].dt.month
    def month_center_per_practice(x):
        return x.groupby(month_numbers).apply(lambda m: m - m.mean(dim="date"))
    items_norm = month_center_per_practice(ds["items"])

    # loop through each practice and analyse flagged vs non-flagged prescription counts
    results = []
    deltas = {k:[] for k in flag_types}
    signif_counts = {k:{"up":0,"down":0} for k in deltas}
    practices = ds['practice_id'].values
    for practice in tqdm(practices,
                         desc="Per-practice analysis",
                         total=len(practices)):
        practice_data = ds.sel(practice_id=practice)
        if practice_data["items"].count() == 0:
            continue

        items = items_norm.sel(practice_id=practice).values

        for kind in deltas.keys():
            # if the flag exists for this practice, respect NaNs (unknown) by excluding them
            if kind in practice_data:
                arr = practice_data[kind].values.astype(float)
                is_nan = np.isnan(arr)
                is_one = arr == 1
                is_zero = arr == 0

                # flagged: explicit ones; nonflagged: explicit zeros only (exclude NaNs)
                flag_vals = items[is_one]
                nonflag_vals = items[is_zero & ~is_nan]
            else:
                # flag variable absent -> no flagged observations, all are valid non-flagged
                flag_vals = items[np.zeros_like(items, dtype=bool)]
                nonflag_vals = items

            if len(flag_vals)<min_group_n or len(nonflag_vals)<min_group_n:
                continue

            mean_flag = np.nanmean(flag_vals)
            mean_non = np.nanmean(nonflag_vals)
            pooled_sd = np.nanstd(np.concatenate([flag_vals, nonflag_vals])) + 1e-9
            std_effect = (mean_flag - mean_non) / pooled_sd

            try:
                _, pval = stats.mannwhitneyu(flag_vals, nonflag_vals,
                                             alternative="two-sided")
            except:
                pval = np.nan

            # bootstrap per-practice using helper (CI and bootstrap p-values)
            boot_res = bootstrap_effects(flag_vals, nonflag_vals, n_boot=n_boot,
                                         jitter_eps=1e-6, min_group_n=min_group_n,
                                         seed=seed)
            if boot_res is not None:
                se_lo, se_hi = boot_res.get("std_effect_ci", [None, None])
                md_lo, md_hi = boot_res.get("mean_diff_ci", [None, None])
                boot_pval = boot_res.get("boot_pval")
                mean_diff_boot_p = boot_res.get("mean_diff_boot_p")
            else:
                se_lo = se_hi = md_lo = md_hi = None
                boot_pval = None
                mean_diff_boot_p = None

            # permutation p-values for mean diff and standardized effect
            try:
                mean_diff_perm_p = permutation_pvalue(lambda a, b: float(np.nanmean(a) - np.nanmean(b)), 
                                                      flag_vals, nonflag_vals, n_perm=n_perm, seed=seed)
            except Exception:
                mean_diff_perm_p = None
            try:
                std_effect_perm_p = permutation_pvalue(lambda a, b: float((np.nanmean(a) - np.nanmean(b)) / (np.nanstd(np.concatenate([a, b])) + 1e-9)),
                                                       flag_vals, nonflag_vals, n_perm=n_perm, seed=seed)
            except Exception:
                std_effect_perm_p = None

            results.append({
                "practice_id": practice,
                "kind": kind,
                "mean_flagged": float(mean_flag),
                "mean_nonflagged": float(mean_non),
                "mean_diff_size": float(mean_flag - mean_non),
                "mean_diff_ci": [None if md_lo is None else float(md_lo),
                                 None if md_hi is None else float(md_hi)],
                "mannwhitney_p": float(pval),
                "mean_diff_boot_p": mean_diff_boot_p,
                "mean_diff_perm_p": mean_diff_perm_p,
                "std_effect_size": float(std_effect),
                "std_effect_ci": [None if se_lo is None else float(se_lo),
                                  None if se_hi is None else float(se_hi)],
                "std_effect_boot_p": boot_pval,
                "std_effect_perm_p": std_effect_perm_p,
                "n_flagged": int(len(flag_vals)),
                "n_nonflagged": int(len(nonflag_vals)),
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

    # perform FDR (Benjamini-Hochberg) corrections on per-practice p-values where present
    if not results_df.empty:
        # ensure columns exist
        for col in ["mannwhitney_p", "mean_diff_perm_p", "std_effect_perm_p"]:
            if col not in results_df.columns:
                results_df[col] = np.nan

        mw_p = results_df["mannwhitney_p"].fillna(1.0).values
        md_p = results_df["mean_diff_perm_p"].fillna(1.0).values
        se_p = results_df["std_effect_perm_p"].fillna(1.0).values

        try:
            _, mw_q, _, _ = multipletests(mw_p, alpha=0.05, method="fdr_bh")
            _, md_q, _, _ = multipletests(md_p, alpha=0.05, method="fdr_bh")
            _, se_q, _, _ = multipletests(se_p, alpha=0.05, method="fdr_bh")
        except Exception:
            mw_q = np.ones_like(mw_p)
            md_q = np.ones_like(md_p)
            se_q = np.ones_like(se_p)

        results_df["mannwhitney_q"] = mw_q
        results_df["mean_diff_perm_q"] = md_q
        results_df["std_effect_perm_q"] = se_q
        results_df["mannwhitney_fdr_sig"] = results_df["mannwhitney_q"] <= 0.05
        results_df["mean_diff_perm_fdr_sig"] = results_df["mean_diff_perm_q"] <= 0.05
        results_df["std_effect_perm_fdr_sig"] = results_df["std_effect_perm_q"] <= 0.05

    save_filename = nc_file_path.replace(".nc", "_per_practice_analysis.csv")
    results_df.to_csv(save_filename, index=False)

    # summary including FDR counts
    signif_counts_fdr = {k: {"mannwhitney_fdr": 0, "mean_diff_perm_fdr": 0, "std_effect_perm_fdr": 0} for k in deltas}
    if not results_df.empty:
        for kind in deltas.keys():
            sub = results_df[results_df["kind"] == kind]
            if not sub.empty:
                signif_counts_fdr[kind]["mannwhitney_fdr"] = int(sub["mannwhitney_fdr_sig"].sum())
                signif_counts_fdr[kind]["mean_diff_perm_fdr"] = int(sub["mean_diff_perm_fdr_sig"].sum())
                signif_counts_fdr[kind]["std_effect_perm_fdr"] = int(sub["std_effect_perm_fdr_sig"].sum())

    summary = {
        "median_sd_effect": {k: float(np.nanmedian(v)) for k, v in deltas.items()},
        "signif_counts": signif_counts,
        "signif_counts_fdr": signif_counts_fdr
    }
    print(summary)
    save_path = "outputs/" + nc_file_path.replace(".nc", "") + "/per_practice_summary.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)


# HELPER FUNCTIONS ================================================================================
def bootstrap_effects(flagged, nonflagged, n_boot=1000, jitter_eps=1e-6,
                      min_group_n=MIN_GROUP_N, seed=None):
    """
    Compute bootstrap estimates for mean difference and standardized effect between two samples.

    Returns a dict with keys:
      - 'std_effect_ci': [lo, hi]
      - 'mean_diff_ci': [lo, hi]
      - 'boot_pval': two-sided bootstrap p-value for standardized effect

    If inputs are too small for bootstrapping, returns None.
    """
    f = np.asarray(flagged)
    nf = np.asarray(nonflagged)
    # remove nans
    f = f[~np.isnan(f)]
    nf = nf[~np.isnan(nf)]

    if len(f) < min_group_n or len(nf) < min_group_n or not n_boot or n_boot <= 0:
        return None

    rng = np.random.default_rng(seed)
    boot_std = np.empty(n_boot)
    boot_md = np.empty(n_boot)
    for i in range(n_boot):
        f_samp = rng.choice(f, size=len(f), replace=True)
        nf_samp = rng.choice(nf, size=len(nf), replace=True)
        # add tiny jitter to avoid degenerate ties
        f_samp_j = f_samp + rng.normal(0, jitter_eps, f_samp.shape)
        nf_samp_j = nf_samp + rng.normal(0, jitter_eps, nf_samp.shape)
        mean_f = np.nanmean(f_samp_j)
        mean_nf = np.nanmean(nf_samp_j)
        pooled = np.nanstd(np.concatenate([f_samp_j, nf_samp_j])) + 1e-9
        boot_md[i] = mean_f - mean_nf
        boot_std[i] = (mean_f - mean_nf) / pooled

    se_lo, se_hi = np.percentile(boot_std, [2.5, 97.5])
    md_lo, md_hi = np.percentile(boot_md, [2.5, 97.5])
    prop_pos = np.mean(boot_std > 0)
    boot_pval = float(2.0 * min(prop_pos, 1.0 - prop_pos))

    return {
        "std_effect_ci": [float(se_lo), float(se_hi)],
        "mean_diff_ci": [float(md_lo), float(md_hi)],
        "boot_pval": boot_pval,
        "mean_diff_boot_p": float(2.0 * min(np.mean(boot_md > 0), 1.0 - np.mean(boot_md > 0)))
    }

def permutation_pvalue(stat_fn, a, b, n_perm=1000, seed=None):
    """
    Compute a two-sided permutation p-value for statistic stat_fn(a, b).
    stat_fn should accept two 1D arrays and return a scalar.
    Returns float p-value.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    # remove nans
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return None

    rng = np.random.default_rng(seed)
    obs = stat_fn(a, b)
    pooled = np.concatenate([a, b])
    n = len(a)
    perms = 0
    ge = 0
    for i in range(n_perm):
        idx = rng.choice(len(pooled), size=len(pooled), replace=False)
        # permuted groups by shuffling and splitting
        perm = pooled[idx]
        pa = perm[:n]
        pb = perm[n:]
        pstat = stat_fn(pa, pb)
        if abs(pstat) >= abs(obs):
            ge += 1
        perms += 1

    pval = float((ge + 1) / (perms + 1))
    return pval

def load_json(json_path):
    """Load JSON file and return features list."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data.get("features", [])

def generate_flags(z_datetimes, z_values, flag_type, target_months, z_thresh=2.0):
    """
    Create simple anomaly flags (0/1) for each monthly sum of readings.
    'flag_type' can be 'high', 'low', or 'median'.
    """
    # compute raw flags, preserving NaNs
    if flag_type == "high":
        flagged = np.where(np.isnan(z_values), np.nan, z_values >= z_thresh)
    elif flag_type == "low":
        flagged = np.where(np.isnan(z_values), np.nan, z_values <= -z_thresh)
    elif flag_type == "median":
        flagged = np.where(np.isnan(z_values), np.nan, np.abs(z_values) <= z_thresh)
    else:
        raise ValueError("flag_type must be 'high', 'low', or 'median'")

    # convert dates to monthly periods
    z_months = pd.to_datetime(z_datetimes).to_period("M")
    target_months = pd.to_datetime(target_months).to_period("M")

    # create output array
    flags_out = np.full(len(target_months), np.nan, dtype=np.float32)

    # vectorized mapping
    for month in np.unique(z_months):
        mask_target = target_months == month
        mask_z = z_months == month
        if np.any(mask_target):
            flags_out[mask_target] = np.where(flagged[mask_z], 1.0, 0.0).astype(np.float32)[0]

    return flags_out

def remove_seasonal_effects(datetimes, values):
    datetimes = pd.to_datetime(datetimes)
    month_nums = datetimes.month
    medians = np.full(12, np.nan)
    mads = np.full(12, np.nan)

    for m in range(1, 13):
        mask = (month_nums == m)
        if not np.any(mask):
            continue
        v = values[mask]
        med = np.nanmedian(v)
        medians[m - 1] = med
        mads[m - 1] = np.nanmedian(np.abs(v - med)) * 1.4826

    # Vectorized z-score calculation (no list comprehension)
    m = month_nums - 1
    monthly_anomalies = (values - medians[m]) / (mads[m] + 1e-9)
    return monthly_anomalies

def aggregate_monthly(datetimes, values, method):
    datetimes = pd.to_datetime(datetimes)
    df = pd.DataFrame({'date': datetimes, 'value': values})
    df.set_index('date', inplace=True)
    if method == "sum":
        # ensure months with <15 days are NaN
        monthly = df.resample('MS').sum(min_count=15)
    else:
        monthly = df.resample('MS').agg(method)
    return monthly.index.values, monthly['value'].values


# DEPRECATED ==
HYDROLOGY_API_BASE = "https://environment.data.gov.uk/hydrology"
def fetch_hydro_measures(station_guid):
    """Fetch all measures (timeseries) for a station."""
    url = f"{HYDROLOGY_API_BASE}/id/stations/{station_guid}/measures"
    params = {"_view": "default"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    items = r.json().get("items", [])
    # return a list of measure IDs
    return [m["@id"].split("/")[-1] for m in items]

def fetch_hydro_readings_for_period(measure_id, start_date, end_date):
    """Fetch readings for a measure and return (timestamps, values) arrays."""
    url = f"{HYDROLOGY_API_BASE}/id/measures/{measure_id}/readings"
    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    params = {"mineq-date": start_date,
              "max-date": end_date,
              "_limit": 2000000}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return np.array([]), np.array([])
    datetimes = np.array([pd.to_datetime(x["dateTime"]) for x in items])
    values = np.array([
        float(x["value"]) if "value" in x and x["value"] not in [None, ""] else np.nan
        for x in items
    ])
    return datetimes, values

def fetch_hydro_readings(measure_id):
    """Fetch readings for a measure and return (timestamps, values) arrays."""
    url = f"{HYDROLOGY_API_BASE}/id/measures/{measure_id}/readings"
    params = {"_limit": 2000000}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return np.array([]), np.array([])
    datetimes = np.array([pd.to_datetime(x["dateTime"]) for x in items])
    values = np.array([
        float(x["value"]) if "value" in x and x["value"] not in [None, ""] else np.nan
        for x in items
    ])
    return datetimes, values

def add_hydrology_flags_old(prescriptions_ds, hydrology_ds,
                        observed_property="rain", agg="sum",
                        flag_types=["high", "low", "median"]):
    """
    Add flags to the prescriptions dataset based on hydrology station data.

    prescriptions_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    hydrology_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    observed_property: str, the property to observe (e.g. "rain")
    agg: str, the aggregation method to use (e.g. "sum")
    flag_types: list of str, the types of flags to create (e.g. ["high", "low", "median"])
    """
    # get practice and station locations
    lat_p = prescriptions_ds.latitude.values
    lon_p = prescriptions_ds.longitude.values
    lat_s = hydrology_ds.latitude.values
    lon_s = hydrology_ds.longitude.values
    sid_s = hydrology_ds.station_id.values

    # find nearest station for each practice
    tree = cKDTree(np.column_stack([lat_s, lon_s]))
    _, nearest_idx = tree.query(np.column_stack([lat_p, lon_p]))
    nearest_stations = sid_s[nearest_idx]
    unique_stations = np.unique(nearest_stations)

    # prepare arrays for flags
    nc_datetimes = pd.to_datetime(prescriptions_ds.date.values)
    nc_months = nc_datetimes.to_period("M")
    outputs = {}
    for flag_type in flag_types:
        outputs[flag_type] = np.full((len(nc_datetimes), len(lat_p)), np.nan, dtype=np.float32)
    outputs["values"] = np.full((len(nc_datetimes), len(lat_p)), np.nan, dtype=np.float32)

    # get flags for each unique station
    for station_id in tqdm(unique_stations,
                           desc="      Fetching station flags",
                           total=len(unique_stations)):
        # get relevant measures for this station
        measures = fetch_hydro_measures(station_id)
        if observed_property == "rain":
            measure_id = [m for m in measures if "rainfall-t-86400" in m]
        else:
            raise ValueError(f"Unsupported observed_property: {observed_property}")
        
        # check measure found
        if len(measure_id) == 0:
            raise ValueError(f"No {observed_property} measure found for station {station_id}")
        elif len(measure_id) > 1:
            print(f"Warning: multiple {observed_property} measures found for station {station_id}," +
                   " using the first one.")
        measure_id = measure_id[0]
        
        # get readings for the measure
        # start_date = nc_datetimes.min().strftime("%Y-%m-%d")
        # end_date = nc_datetimes.max().strftime("%Y-%m-%d")
        # datetimes, values = fetch_hydro_readings_for_period(measure_id, start_date, end_date)
        datetimes, values = fetch_hydro_readings(measure_id)

        if len(datetimes) == 0:
            print(f"No readings found for station {station_id}, measure {measure_id}.",
                  " Setting flags to NaN.")
            for flag_type in flag_types:
                outputs[flag_type][:, nearest_stations == station_id] = np.nan
            continue

        # aggregate the data to monthly totals
        datetimes_agg, values_agg = aggregate_monthly(datetimes, values, agg)

        # remove seasonal effects from readings
        z_values = remove_seasonal_effects(datetimes_agg, values_agg)

        # generate flags
        mask = nearest_stations == station_id
        for flag_type in flag_types:
            flags_temp = generate_flags(datetimes_agg, z_values, flag_type, nc_months)
            outputs[flag_type][:, mask] = flags_temp[:, None]
        
        # save aggregated values for nc_months
        agg_months = pd.to_datetime(datetimes_agg).to_period("M")
        values_series = pd.Series(values_agg, index=agg_months)
        aligned_values = values_series.reindex(nc_months, fill_value=np.nan).values
        outputs["values"][:, mask] = aligned_values[:, None]
        
    # create arrays for new flags
    for flag_type in flag_types:
        prescriptions_ds[f"hydro_{observed_property}_{flag_type}"] = (("date", "practice_id"),
                                                                      outputs[flag_type])
    prescriptions_ds[f"hydro_{observed_property}_values"] = (("date", "practice_id"),
                                                             outputs["values"])

    return prescriptions_ds

def remove_seasonal_effects_old(datetimes, values):
    # collect monthly medians and mean absolute deviations (MADs)
    # (median and MAD are more robust to outliers than mean and SD)
    month_nums = np.array([m.month for m in datetimes])
    medians = np.zeros(12)
    mads = np.zeros(12)
    for m in range(1, 13):
        mask = month_nums == m
        if not np.any(mask):
            medians[m-1] = np.nan
            mads[m-1] = np.nan
            continue
        v = values[mask]
        medians[m-1] = np.nanmedian(v)
        mads[m-1] = np.nanmedian(np.abs(v - np.nanmedian(v)))
    
    # convert MADs to standard deviation equivalents
    # 1.4826 approximates SD for normal distribution
    mads = mads * 1.4826

    # compute monthly anomaly "z-scores"
    monthly_anomalies = np.array([
        (val - medians[m - 1]) / (mads[m - 1] + 1e-9)
        for val, m in zip(values, [m.month for m in datetimes])
    ])

    return monthly_anomalies

def aggregate_monthly_old(datetimes, values, method):
    month_periods = np.array([pd.Period(t, freq="M") for t in datetimes])
    unique_months = np.unique(month_periods)

    # compute monthly totals
    if method == "sum":
        monthly_vals = np.array([
            np.nansum(values[month_periods == m]) for m in unique_months
        ])
    elif method == "mean":
        monthly_vals = np.array([
            np.nanmean(values[month_periods == m]) for m in unique_months
        ])
    elif method == "median":
        monthly_vals = np.array([
            np.nanmedian(values[month_periods == m]) for m in unique_months
        ])
    elif method == "max":
        monthly_vals = np.array([
            np.nanmax(values[month_periods == m]) for m in unique_months
        ])
    elif method == "min":
        monthly_vals = np.array([
            np.nanmin(values[month_periods == m]) for m in unique_months
        ])
    else:
        raise ValueError("Unhandled aggregation method: " + method +
                        ". Use 'sum', 'mean', 'median', 'max' or 'min'.")
    
    # convert months back to datetimes
    unique_months = np.array([m.to_timestamp() for m in unique_months])
    return unique_months, monthly_vals

def aggregate_monthly_old2(datetimes, values, method):
    # convert to YYYYMM integer for grouping
    datetimes = pd.to_datetime(datetimes)
    keys = datetimes.year * 12 + datetimes.month
    sorter = np.argsort(keys)
    keys, values = keys[sorter], values[sorter]

    unique_keys, idx_start = np.unique(keys, return_index=True)
    idx_end = np.r_[idx_start[1:], len(values)]

    agg_funcs = {
        "sum": np.nansum,
        "mean": np.nanmean,
        "median": np.nanmedian,
        "max": np.nanmax,
        "min": np.nanmin
    }
    func = agg_funcs[method]

    monthly_vals = np.array([func(values[i0:i1]) for i0, i1 in zip(idx_start, idx_end)])
    months = [pd.Timestamp(year=int(k // 12), month=int(k % 12 or 12), day=1) for k in unique_keys]
    return np.array(months), monthly_vals
