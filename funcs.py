import os, json, requests, re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pyproj import Transformer
from tqdm import tqdm
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from joblib import Parallel, delayed


MIN_GROUP_N = 5  # minimum number of flags against practice for analysis

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
        outputs[flag_type] = np.full((len(pres_datetimes), len(lat_p)),
                                     np.nan, dtype=np.float32)
    outputs["values"] = np.full((len(pres_datetimes), len(lat_p)),
                                np.nan, dtype=np.float32)

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
        monthly_rain_months = pd.to_datetime(monthly_rain_datetimes).to_period("M")

        # remove seasonal effects from readings
        monthly_z_values = remove_seasonal_effects(monthly_rain_datetimes,
                                                   monthly_rain_readings)

        # generate flags
        mask = nearest_stations == station_id
        for flag_type in flag_types:
            flags_temp = generate_flags(monthly_rain_months,
                                        monthly_z_values,
                                        flag_type,
                                        pres_months)
            outputs[flag_type][:, mask] = flags_temp[:, None]
        
        # save aggregated values for nc_months
        values_series = pd.Series(monthly_rain_readings, index=monthly_rain_months)
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
    lat_vec = prescriptions_ds.coords['latitude'].values
    lon_vec = prescriptions_ds.coords['longitude'].values

    # convert lat/lon to projected coordinates (EPSG:27700, British National Grid)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    xs, ys = transformer.transform(lon_vec, lat_vec)

    # preprocess flood polygons
    geoms = []
    months = []
    for f in tqdm(geojson_features,
                  desc="      Processing flood polygons",
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
            met_months = met_datetimes.to_period("M")

            # remove seasonal effects from readings
            z_values = remove_seasonal_effects(met_datetimes, values)

            # generate flags
            nc_months = pres_datetimes.to_period("M")
            mask = nearest_stations == station_id
            for flag_type in flag_types:
                flags_temp = generate_flags(met_months, z_values, flag_type, nc_months)
                outputs[flag_type][:, mask] = flags_temp[:, None]
                    
            # save values for nc file time period
            values_series = pd.Series(values, index=met_months)
            aligned_values = values_series.reindex(pres_months, fill_value=np.nan).values
            outputs["values"][:, mask] = aligned_values[:, None]

        # create arrays for new flags
        for flag_type in flag_types:
            prescriptions_ds[f"met_{observed_property}_{flag_type}"] = (("date", "practice_id"),
                                                                        outputs[flag_type])
        prescriptions_ds[f"met_{observed_property}_values"] = (("date", "practice_id"),
                                                                outputs["values"])
    return prescriptions_ds

def add_particulate_flags(prescriptions_ds, particulates_ds, mass_z_thresh=1.5):
    # align particulates ds to prescriptions ds
    particulates_ds = particulates_ds.reindex(date=prescriptions_ds.date,
                                              practice_id=prescriptions_ds.practice_id,
                                              method=None)
    
    # get flags for each particulates variable
    part_vars = particulates_ds.data_vars
    for var in tqdm(part_vars,
                    desc="      Fetching particulate flags",
                    total=len(part_vars)):

        is_daqi = "daqi" in var
        values = particulates_ds[var].values

        # convert mass concentration to z-scores and configure flags
        if not is_daqi:
            medians = np.nanmedian(values, axis=0, keepdims=True)
            mads = np.nanmedian(np.abs(values - medians), axis=0, keepdims=True) * 1.4826
            vals_to_flag = (values - medians) / mads
            thresholds = {"high": (mass_z_thresh, None)}
            flag_types = ["high"]
        # otherwise use unprocessed DAQI monthly maximums and standard DAQI thresholds
        else:
            vals_to_flag = values
            thresholds = {
                "very_high": (10, None),
                "high": (7, 10),
                "moderate": (4, 7),
                "low": (1, 4)
            }
            flag_types = list(thresholds.keys())

        # generate flags
        outputs = {}
        for flag_type in flag_types:
            low, high = thresholds[flag_type]
            if low is None:
                mask = vals_to_flag < high
            elif high is None:
                mask = vals_to_flag >= low
            else:
                mask = (vals_to_flag >= low) & (vals_to_flag < high)
            # ensure mask is NaN where values are NaN
            mask = mask.astype(np.float32)
            mask[np.isnan(values)] = np.nan
            outputs[flag_type] = mask

        # store raw values
        outputs["values"] = values.astype(np.float32)

        # re-order name of daqi variables for new dataset
        if is_daqi and "_daqi" in var:
            var = "daqi_" + var.replace("_daqi", "")

        # add to prescriptions dataset
        for flag_type in flag_types:
            prescriptions_ds[f"aqrean_{var}_{flag_type}"] = (("date", "practice_id"), outputs[flag_type])
        prescriptions_ds[f"aqrean_{var}_values"] = (("date", "practice_id"), outputs["values"])

    return prescriptions_ds


# INSPECTION FUNCTIONS ============================================================================
def plot_practices(ds, nc_file_path, flag_types, sample_size=30, seed=None):
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

        n = len(flag_types)
        fig, axes = plt.subplots(n, 3, figsize=(18, 6*n))
        for f, flag_type in enumerate(flag_types):
            # get vars for this flag type
            flag_vars = [ft for ft in practice_data.data_vars
                         if ft.startswith(flag_type)]
            if len(flag_vars) == 0:
                continue

            # plot prescriptions with flags
            plot_prescriptions(axes[f, 0], practice_data, flag_vars)

            # plot readings for flagged variable
            plot_readings(axes[f, 1], practice_data, flag_vars)

            # plot readings with seasonal correction
            plot_readings(axes[f, 2], practice_data, flag_vars, seasonal_correction=True)

        # save
        folder_name = nc_file_path.split('/')[-1].replace('.nc', '')
        save_path = f"outputs/{folder_name}/{practice}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.tight_layout()
        fig.suptitle(f"Practice ID: {practice}", y=1.02, fontsize=16)
        plt.savefig(save_path)
        plt.close()

def plot_prescriptions(ax, practice_data, flag_vars):
    datetimes = practice_data['date'].values
    items = practice_data["items"].values
    ax.plot(datetimes, items, "ko-", markersize=10, alpha=0.2,
            label="Prescriptions")

    # plot flagged points
    if flag_vars is not None:
        if len(flag_vars) == 1:
            # mark flagged months
            flag_var = flag_vars[0]
            flags = practice_data[flag_var].values
            flagged_dates = datetimes[flags == 1]
            flagged_values = items[flags == 1]
            ax.scatter(flagged_dates, flagged_values,
                    c='b', s=60, marker='v', alpha=0.7,
                    label=" ".join(flag_var.split('_')).title(),
                    zorder=5)
        elif len(flag_vars) == 2:
            # mark high months
            flag_var = [fv for fv in flag_vars if fv.endswith("high")][0]
            flags = practice_data[flag_var].values
            flagged_dates = datetimes[flags == 1]
            flagged_values = items[flags == 1]
            ax.scatter(flagged_dates, flagged_values,
                    c='r', s=60, marker='^', alpha=0.7,
                    label=" ".join(flag_var.split('_')).title(),
                    zorder=5)
        elif len(flag_vars) == 4:
            # mark high/low months
            params = (("high", "r", "^"), ("low", "b", "v"))
            for suffix, color, marker in params:
                flag_var = [fv for fv in flag_vars if fv.endswith(suffix)][0]
                flags = practice_data[flag_var].values
                flagged_dates = datetimes[flags == 1]
                flagged_values = items[flags == 1]
                ax.scatter(flagged_dates, flagged_values,
                        c=color, s=60, marker=marker, alpha=0.7,
                        label=suffix.title() + " " + " ".join(flag_var.split('_')[:-1]).title(),
                        zorder=5)
        elif len(flag_vars) == 5:
            # mark very_high/high/low/months
            params = (("very_high", "m", "D"), ("high", "r", "^"),
                      ("moderate", "y", "o"), ("low", "g", "v"))
            for suffix, color, marker in params:
                if suffix == "high":
                    flag_var = [fv for fv in flag_vars if fv.endswith("high")
                                and "very" not in fv][0]
                else:
                    flag_var = [fv for fv in flag_vars if fv.endswith(suffix)][0]
                flags = practice_data[flag_var].values
                flagged_dates = datetimes[flags == 1]
                flagged_values = items[flags == 1]
                ax.scatter(flagged_dates, flagged_values,
                           c=color, s=60, marker=marker, alpha=0.7,
                           label=suffix.title() + " " + " ".join(flag_var.split('_')[:-1]).title(),
                           zorder=5)
    
    years = pd.date_range(datetimes.min(), datetimes.max(), freq="YS")
    ax.set_xticks(years, years.year)
    ax.set_ylabel("Number of Prescriptions")
    ax.legend()
    ax.grid()
    return ax

def plot_readings(ax, practice_data, flag_vars, seasonal_correction=False):
    # find the key for the readings
    readings_key = [v for v in flag_vars if v.endswith("values")]
    if len(readings_key) == 0 or len(flag_vars) <= 1:
        # no readings variable or just a simple flag (e.g. flood)
        return ax

    # get readings
    datetimes = practice_data['date'].values
    readings = practice_data[readings_key[0]].values

    # correct for seasonal effects
    if seasonal_correction:
        readings = remove_seasonal_effects(datetimes, readings)

    # plot readings baseline
    ax.plot(datetimes, readings, "ko-", markersize=10, alpha=0.2,
            label="Readings")

    # plot flagged months depending on how many flag vars exist
    if len(flag_vars) == 2:
        # high + values
        flag_var = [fv for fv in flag_vars if fv.endswith("high")][0]
        flags = practice_data[flag_var].values
        flagged_dates = datetimes[flags == 1]
        flagged_values = readings[flags == 1]
        ax.scatter(flagged_dates, flagged_values,
                   c='r', s=60, marker='^', alpha=0.7,
                   label=" ".join(flag_var.split('_')).title(),
                   zorder=5)

    elif len(flag_vars) == 4:
        # high/low + values
        params = (("high", "r", "^"), ("low", "b", "v"))
        for suffix, color, marker in params:
            flag_var = [fv for fv in flag_vars if fv.endswith(suffix)][0]
            flags = practice_data[flag_var].values
            flagged_dates = datetimes[flags == 1]
            flagged_values = readings[flags == 1]
            ax.scatter(flagged_dates, flagged_values,
                       c=color, s=60, marker=marker, alpha=0.7,
                       label=suffix.title() + " " + " ".join(flag_var.split('_')[:-1]).title(),
                       zorder=5)

    elif len(flag_vars) == 5:
        # DAQI-style very_high/high/moderate/low + values
        params = (("very_high", "m", "D"), ("high", "r", "^"),
                  ("moderate", "y", "o"), ("low", "g", "v"))
        for suffix, color, marker in params:
            if suffix == "high":
                flag_var = [fv for fv in flag_vars if fv.endswith("high") and "very" not in fv][0]
            else:
                flag_var = [fv for fv in flag_vars if fv.endswith(suffix)][0]
            flags = practice_data[flag_var].values
            flagged_dates = datetimes[flags == 1]
            flagged_values = readings[flags == 1]
            ax.scatter(flagged_dates, flagged_values,
                       c=color, s=60, marker=marker, alpha=0.7,
                       label=suffix.title() + " " + " ".join(flag_var.split('_')[:-1]).title(),
                       zorder=5)

    years = pd.date_range(datetimes.min(), datetimes.max(), freq="YS")
    ax.set_xticks(years, years.year)
    ax.set_xlim(datetimes.min(), datetimes.max())
    ax.set_ylabel(('Corrected\n' if seasonal_correction else '') +
                  readings_key[0].replace('_', ' ').title())
    ax.legend()
    ax.grid()
    return ax


# ANALYSIS FUNCTIONS ==============================================================================

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

def download_file(url, session, out_dir='', timeout=20, overwrite=False):
    """Download file streaming to disk. Skip if already exists."""
    fname = url.split("/")[-1]
    out_path = out_dir / fname

    if out_path.exists() and not overwrite:
        # optionally check file size to avoid partial downloads
        if out_path.stat().st_size > 1_000_000:  # >1MB sanity check
            print(f"Skipping already downloaded ZIP: {fname}")
            return out_path
        else:
            print(f"Re-downloading incomplete ZIP: {fname}")

    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    fh.write(chunk)
    return out_path

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

def generate_flags(z_months, z_values, flag_type, target_months, z_thresh=1.0):
    """
    Create simple anomaly flags (0/1) for each monthly sum of readings.
    'flag_type' can be 'high', 'low', or 'median'.
    """
    # compute raw flags
    if flag_type == "high":
        flagged = z_values >= z_thresh
    elif flag_type == "low":
        flagged = z_values <= -z_thresh
    elif flag_type == "median":
        flagged = np.abs(z_values) < z_thresh
    else:
        raise ValueError("flag_type must be 'high', 'low', or 'median'")

    # map flags to target months
    flags_out = np.full(len(target_months), np.nan, dtype=bool)
    for month in np.unique(z_months):
        mask_target = target_months == month
        if np.any(mask_target):
            mask_z = z_months == month
            flags_out[mask_target] = flagged[mask_z]

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

    # Vectorized z-score calculation
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

















# DEPRECATED ======================================================================================
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
