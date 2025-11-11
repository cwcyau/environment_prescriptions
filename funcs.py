import os, json, requests, re
import numpy as np
import pandas as pd
import bambi as bmb
import arviz as az
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from datetime import datetime
from pyproj import Transformer
from tqdm import tqdm
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from typing import List, Dict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# DATA FUNCTIONS ==================================================================================
def add_hydrology_flags(prescriptions_ds, hydrology_ds,
                        observed_property="rain", agg="sum",
                        flag_types=["high", "low", "median"],
                        seasonal_correction=False):
    """
    Add flags to the prescriptions dataset based on hydrology station data.

    prescriptions_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    hydrology_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    observed_property: str, the property to observe (e.g. "rain")
    agg: str, the aggregation method to use (e.g. "sum")
    flag_types: list of str, the types of flags to create (e.g. ["high", "low", "median"])
    seasonal_correction: bool, whether to apply seasonal correction
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
        if seasonal_correction:
            monthly_z_values = remove_seasonal_effects(monthly_rain_datetimes,
                                                       monthly_rain_readings)
        else:
            monthly_z_values = standardise_mad(monthly_rain_readings)

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
                  flag_types=["high", "low", "median"],
                  seasonal_correction=False):
    """
    Add flags to the prescriptions dataset based on MET Office station data.

    prescriptions_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    met_ds: xarray Dataset with 'latitude', 'longitude', 'date' coords
    observed_property: str, the property to observe (e.g. "rain")
    agg: str, the aggregation method to use (e.g. "sum")
    flag_types: list of str, the types of flags to create (e.g. ["high", "low", "median"])
    seasonal_correction: bool, whether to apply seasonal correction
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
        status("    Adding MET flags for", observed_property)
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
            if seasonal_correction:
                z_values = remove_seasonal_effects(met_datetimes, values)
            else:
                z_values = standardise_mad(values)

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

def add_particulate_flags(prescriptions_ds, particulates_ds, mass_z_thresh=1.5, seasonal_correction=False):
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
            if seasonal_correction:
                vals_to_flag = remove_seasonal_effects(particulates_ds.date.values, values)
            else:
                vals_to_flag = standardise_mad(values)
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
        if is_daqi:
            if var == "daqi":
                var = "daqi_overall"
            else:
                var = "daqi_" + var.replace("_daqi", "")

        # add to prescriptions dataset
        for flag_type in flag_types:
            prescriptions_ds[f"aqrean_{var}_{flag_type}"] = (("date", "practice_id"), outputs[flag_type])
        prescriptions_ds[f"aqrean_{var}_values"] = (("date", "practice_id"), outputs["values"])

    return prescriptions_ds

# DATA HELPERS ------------------------------------------------------------------------------------
def download_file(url, session, out_dir='', timeout=20, overwrite=False):
    """Download file streaming to disk. Skip if already exists."""
    fname = url.split("/")[-1]
    out_path = out_dir / fname

    if out_path.exists() and not overwrite:
        # optionally check file size to avoid partial downloads
        if out_path.stat().st_size > 1_000_000:  # >1MB sanity check
            status(f"Skipping already downloaded ZIP: {fname}")
            return out_path
        else:
            status(f"Re-downloading incomplete ZIP: {fname}")

    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    fh.write(chunk)
    return out_path

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
    m = month_nums - 1  # 0-based months (0..11)

    # ensure 2D shape for uniform processing
    values_2d = np.atleast_2d(values)
    if values_2d.shape[0] == 1 and len(datetimes) > 1:
        # transpose if accidentally shaped (station_id, date)
        values_2d = values_2d.T

    n_date, n_stations = values_2d.shape
    medians = np.full((12, n_stations), np.nan)
    mads = np.full((12, n_stations), np.nan)

    # compute monthly median and median absolute difference per column
    for month in range(1, 13):
        mask = month_nums == month
        if not np.any(mask):
            continue
        v_month = values_2d[mask, :]
        med = np.nanmedian(v_month, axis=0)
        mad = np.nanmedian(np.abs(v_month - med), axis=0) * 1.4826
        medians[month - 1, :] = med
        mads[month - 1, :] = mad

    # calculate monthly anomalies
    anomalies = (values_2d - medians[m, :]) / (mads[m, :] + 1e-9)

    # revert shape
    if values.ndim == 1:
        return anomalies[:, 0]
    return anomalies

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

# INSPECTION FUNCTIONS ----------------------------------------------------------------------------
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



# MODELING FUNCTIONS ==============================================================================
def run_all_flag_mixed_models(
    ds,
    flag_types: List[str],
    results_folder: str,
    deseasonalise_output: bool = False,
    practice_correction: int = 0,
    min_practice_obs: int = 20,
    n_jobs: int = 1,
):
    """
    Runs mixed-effects models comparing prescription 'items' across requested flag groups.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'items' and flag variables (indexed by date, practice_id).
    flag_types : list of str
        Base flag names to test, e.g. ["hydro_rain","met_tmax","flood","aqrean_carbon_monoxide","aqrean_daqi", ...]
        The function will look for derived variable names (e.g. hydro_rain_high, hydro_rain_median, ...).
    results_folder : str
        Folder path for saving results.
    seasonal_correction_out : bool
        Whether to add month-of-year fixed effects: " + C(month)".
    practice_correction : int
        Whether to add practice-level fixed effects:
        0 = none,
        1 = random intercepts,
        2 = random intercepts + slopes.
    standardise_items : bool
        Whether to standardise 'items' (per practice) before modelling.
    min_practice_obs : int
        Practices excluded if they have fewer than this number of observations.
    n_jobs : int
        Number of parallel jobs (joblib).
    """

    def fit_mixed_effects(df: pd.DataFrame, formula: str, re_formula: str, group_var="practice_id") -> Dict:
        out = dict(coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan, error=None)
        try:
            md = smf.mixedlm(formula, df, groups=df[group_var], re_formula=re_formula)
            mdf = md.fit(method="lbfgs", reml=True, disp=False)
            # expected coefficient name is 'flag_binary' in formula
            if "flag_binary" in mdf.params:
                coef = float(mdf.params["flag_binary"])
                pval = float(mdf.pvalues.get("flag_binary", np.nan))
                ci_low, ci_high = map(float, mdf.conf_int().loc["flag_binary"].values)
            else:
                coef = np.nan
                pval = np.nan
                ci_low, ci_high = [np.nan, np.nan]
            out.update(dict(coef=coef, pval=pval, ci_low=ci_low, ci_high=ci_high))
        except Exception as e:
            out.update(dict(error=str(e)))
        return out

    # build tasks from requested flag_types =======================================================
    tasks = []  # each task is tuple (family, task_name, task_args)
    seen_task_names = set()

    def add_task(family, name, *args):
        if name in seen_task_names:
            return
        seen_task_names.add(name)
        tasks.append((family, name, *args))

    # helper to check existence of variable(s) in ds and warn if missing
    def vars_exist(*vars_to_check) -> bool:
        missing = [v for v in vars_to_check if v not in ds]
        if missing:
            status(f"Missing variables in ds (skipping comparison): {missing}")
            return False
        return True

    for base in flag_types:
        # flood (single var: flood)
        # compare flood == 1 to flood == 0
        if base == "flood":
            if vars_exist("flood"):
                add_task("flood", "flood_vs_not", "flood")
            else:
                status("Requested 'flood' but variable 'flood' not found in dataset.")

        # hydro_/met_ (high/low/median/values)
        # compare high vs median, low vs median
        elif base.startswith("hydro_") or base.startswith("met_"):
            high = f"{base}_high"
            median = f"{base}_median"
            low = f"{base}_low"
            # high vs median
            if vars_exist(high, median):
                add_task("hydro_met", f"{base}_high_vs_median", high, median)
            # low vs median
            if vars_exist(low, median):
                add_task("hydro_met", f"{base}_low_vs_median", low, median)

        # aqrean_daqi (very_high/high/moderate/low/values)
        # compare (high + very_high) vs (low + moderate), (high + very_high + moderate) vs low
        elif base.startswith("aqrean_daqi"):
            levels = [f"{base}_{lvl}" for lvl in ("very_high", "high", "moderate", "low")]
            if vars_exist(*levels):
                # pair1: (high + very_high) vs (low + moderate)
                add_task("daqi", f"{base}_high+vhigh_vs_low+mod", tuple(levels), "pair1")
                # pair2: (high + very_high + moderate) vs low
                add_task("daqi", f"{base}_high+vhigh+mod_vs_low", tuple(levels), "pair2")
            else:
                status(f"Requested {base} but one or more DAQI level variables are missing: {levels}")

        # aqrean mass pollutants (high/values)
        # compare high == 1 vs high == 0
        elif base.startswith("aqrean_"):
            high = f"{base}_high"
            if vars_exist(high):
                add_task("aqrean_mass", f"{base}_high_vs_not", high)
            else:
                status(f"Requested {base} but variable {high} not found.")

        else:
            status(f"Unknown flag_type '{base}' requested — skipping.")

    if not tasks:
        status("No valid comparison tasks built — nothing to run.")
        return pd.DataFrame([])

    # prepare a base dataframe (items + month + index) ============================================
    status("Preparing base dataframe from dataset...")
    df_items = ds["items"].to_dataframe().reset_index()
    df_items["date"] = pd.to_datetime(df_items["date"])
    df_items["month"] = df_items["date"].dt.month
    df_items = df_items.set_index(["date", "practice_id"])
    df_index = df_items.index

    # run tasks ===================================================================================
    def run_task(task):
        family = task[0]
        name = task[1]
        args = task[2:]
        df_model = df_items[["items", "month"]].copy()
        df_model["date_code"] = df_model.index.get_level_values("date").map(ds["date_code"].to_series())
        df_model["practice_id"] = df_model.index.get_level_values("practice_id")

        # generate correct binary field for flag/family type
        if family == "hydro_met":
            var_flag = args[0]  # high/low variable (depending on task)
            var_med = args[1]   # median variable
            if var_flag not in ds or var_med not in ds:
                return dict(name=name, coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan,
                            error=f"at least one var missing from: {args}")
            # get series aligned to same index
            s_flag = ds[var_flag].to_dataframe()[var_flag].reindex(df_index)
            s_med = ds[var_med].to_dataframe()[var_med].reindex(df_index)
            # flag_binary: 1 if flag==1, 0 if med==1, else NaN
            flag_binary = np.where(s_flag == 1, 1,
                                   np.where(s_med == 1, 0, np.nan))
            df_model["flag_binary"] = flag_binary

        elif family == "flood" or family == "aqrean_mass":
            var = args[0]
            if var not in ds:
                return dict(name=name, coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan,
                            error=f"at least one var missing from: {args}")
            s_f = ds[var].to_dataframe()[var].reindex(df_index)
            # 1 if flood==1, 0 if flood==0, else NaN
            flag_binary = np.where(s_f == 1, 1, np.where(s_f == 0, 0, np.nan))
            df_model["flag_binary"] = flag_binary

        elif family == "daqi":
            levels_tuple = args[0]  # tuple of the four level var names
            pair_kind = args[1]     # "pair1" or "pair2"
            # unpack levels expected order: (very_high, high, moderate, low)
            very_high, high, moderate, low = levels_tuple
            if not all(v in ds for v in (very_high, high, moderate, low)):
                return dict(name=name, coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan,
                            error="missing daqi levels")
            df_daqi = pd.DataFrame(index=df_index)
            df_daqi["vh"] = ds[very_high].to_dataframe()[very_high].reindex(df_index)
            df_daqi["h"]  = ds[high].to_dataframe()[high].reindex(df_index)
            df_daqi["m"]  = ds[moderate].to_dataframe()[moderate].reindex(df_index)
            df_daqi["l"]  = ds[low].to_dataframe()[low].reindex(df_index)
            # define aggregated flags
            flag_high = np.where((df_daqi["vh"] == 1) | (df_daqi["h"] == 1), 1, 0)
            flag_mod  = np.where(df_daqi["m"] == 1, 1, 0)
            flag_low  = np.where(df_daqi["l"] == 1, 1, 0)
            # create mask where any value is present to preserve nans
            any_known = (~df_daqi[["vh", "h", "m", "l"]].isna()).any(axis=1)
            # compute binary per pair
            if pair_kind == "pair1":
                # (high+vhigh) vs (low+moderate)
                cond1 = (flag_high == 1)
                cond0 = (flag_low == 1) | (flag_mod == 1)
                flag_binary = np.where(cond1, 1, np.where(cond0, 0, np.nan))
                flag_binary = np.where(any_known, flag_binary, np.nan)
            else:
                # pair2: (high+vhigh+moderate) vs low
                cond1 = (flag_high == 1) | (flag_mod == 1)
                cond0 = (flag_low == 1)
                flag_binary = np.where(cond1, 1, np.where(cond0, 0, np.nan))
                flag_binary = np.where(any_known, flag_binary, np.nan)

            df_model["flag_binary"] = flag_binary

        else:
            return dict(name=name, coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan, error="unknown family")

        # collapse to rows where both items and flag_binary are present
        df_model_clean = df_model.dropna(subset=["items", "flag_binary"]).copy()

        # filter practices with at least min_practice_obs
        df_model_clean = df_model_clean.reset_index(drop=True)
        practice_counts = df_model_clean.groupby("practice_id").size()
        valid_practices = practice_counts[practice_counts >= min_practice_obs].index
        df_model_clean = df_model_clean[df_model_clean["practice_id"].isin(valid_practices)]

        if df_model_clean.empty:
            return dict(name=name, coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan,
                        error=f"No practices with >= {min_practice_obs} observations")

        # add month if seasonal correction requested
        if deseasonalise_output:
            formula = "items ~ flag_binary + C(month)"
        else:
            formula = "items ~ flag_binary"
        
        # define random effects formula
        if practice_correction == 0:
            re_formula = None
        elif practice_correction == 1:
            re_formula = "~1"
        elif practice_correction == 2:
            re_formula = "~1 + date_code"

        # fit model
        fit_res = fit_mixed_effects(df_model_clean, formula, re_formula=re_formula)
        fit_res["name"] = name
        return fit_res

    # run tasks in parallel (if requested) ========================================================
    status(f"Running mixed-effects models for {len(tasks)} tasks (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_task)(task) for task in tqdm(tasks)
    )

    # save results ================================================================================
    results_df = pd.DataFrame(results)
    os.makedirs(results_folder, exist_ok=True)
    csv_path = os.path.join(results_folder, "mixed_effects_flag_results.csv")
    txt_path = os.path.join(results_folder, "mixed_effects_flag_results.txt")
    # save csv results
    results_df.to_csv(csv_path, index=False)
    status(f"Results saved to {csv_path}")
    # save pretty results text
    max_name_len = results_df["name"].str.len().max()
    max_coef_len = results_df["coef"].apply(lambda x: len(f"{x:.2f}")).max()
    max_ci_len = results_df[["ci_low", "ci_high"]].map(lambda x: len(f"{x:.2f}") if pd.notna(x) else 4).max().max()
    max_pval_len = results_df["pval"].apply(lambda x: len(f"{x:.3g}") if pd.notna(x) else 4).max()
    with open(txt_path, "w") as f:
        for row in results_df.itertuples():
            name_str = row.name.ljust(max_name_len)
            coef_str = f"{row.coef:.2f}".rjust(max_coef_len) if pd.notna(row.coef) else "NaN".rjust(max_coef_len)
            ci_str = f"(CI: {row.ci_low:.2f}, {row.ci_high:.2f})".ljust(max_ci_len+10) if pd.notna(row.ci_low) and pd.notna(row.ci_high) else "(CI: NaN, NaN)".ljust(max_ci_len+16)
            pval_str = f"p = {row.pval:.3g}".rjust(max_pval_len+5) if pd.notna(row.pval) else "p = NaN".rjust(max_pval_len+5)
            error_str = f"** error: {row.error}" if pd.notna(row.error) and row.error != "" else ""
            f.write(f"{name_str}  {coef_str}  {ci_str}  {pval_str}  {error_str}\n")
    status(f"Pretty results saved to {txt_path}")

    return results_df

def run_all_value_mixed_models(
    ds,
    value_vars: List[str],
    results_folder: str,
    deseasonalise_output: bool = False,
    practice_correction: int = 0,
    min_practice_obs: int = 20,
    n_jobs: int = 1,
):
    """
    Runs mixed-effects models comparing prescription 'items' using raw continuous measurements.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'items' and continuous variables (indexed by date, practice_id).
    value_vars : list of str
        Names of variables to test as predictors, e.g. ["hydro_rain", "met_tmax", "aqrean_pm10"].
    results_folder : str
        Folder path for saving results.
    seasonal_correction_out : bool
        Whether to add month-of-year fixed effects to the predicted variable.
    seasonal_correction_in : bool
        Whether to preform month of year correction on predictor variables.
    practice_correction : int
        Level of practice-level random effects correction:
        0 = no practice correction,
        1 = random intercept per practice,
        2 = random intercept + slope on date_code per practice.
    standardise_values : bool
        Whether to standardise predictor values (globally) before modelling.
    standardise_items : bool
        Whether to standardise 'items' (per practice) before modelling.
    min_practice_obs : int
        Practices excluded if they have fewer than this number of observations.
    n_jobs : int
        Number of parallel jobs.
    """

    def fit_mixed_effects(df: pd.DataFrame, formula: str, re_formula: str, group_var="practice_id") -> Dict:
        out = dict(coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan, error=None)
        try:
            md = smf.mixedlm(formula, df, groups=df[group_var], re_formula=re_formula)
            mdf = md.fit(method="lbfgs", reml=True, disp=False)
            predictor = formula.split("~")[1].split("+")[0].strip()
            if predictor in mdf.params:
                coef = float(mdf.params[predictor])
                pval = float(mdf.pvalues.get(predictor, np.nan))
                ci_low, ci_high = map(float, mdf.conf_int().loc[predictor].values)
                out.update(dict(coef=coef, pval=pval, ci_low=ci_low, ci_high=ci_high))
        except Exception as e:
            out.update(dict(error=str(e)))
        return out

    # prepare base dataframe
    df_items = ds[["items"]].to_dataframe().reset_index()
    df_items["date"] = pd.to_datetime(df_items["date"])
    df_items["month"] = df_items["date"].dt.month
    df_items = df_items.set_index(["date", "practice_id"])
    df_index = df_items.index

    tasks = []
    for var in value_vars:
        if var in ds:
            tasks.append(var)
        else:
            status(f"Variable '{var}' not found in dataset — skipping.")

    if not tasks:
        status("No valid variables to run.")
        return pd.DataFrame([])
    
    if practice_correction == 0:
        re_formula = None
    elif practice_correction == 1:
        re_formula = "~1"
    elif practice_correction == 2:
        re_formula = "~1 + date_code"

    def run_task(var):
        # prepare the dataframe for this variable
        df_model = df_items[["items", "month"]].copy()
        df_model["date_code"] = df_model.index.get_level_values("date").map(ds["date_code"].to_series())
        df_model["practice_id"] = df_model.index.get_level_values("practice_id")
        df_model[var] = ds[var].to_dataframe()[var].reindex(df_index)

        # filter to practices with at least min_practice_obs observations
        df_model_clean = df_model.dropna(subset=["items", var]).copy()
        df_model_clean = df_model_clean.reset_index(drop=True)
        practice_counts = df_model_clean.groupby("practice_id").size()
        valid_practices = practice_counts[practice_counts >= min_practice_obs].index
        df_model_clean = df_model_clean[df_model_clean["practice_id"].isin(valid_practices)]
        if df_model_clean.empty:
            return dict(name=var, coef=np.nan, pval=np.nan, ci_low=np.nan, ci_high=np.nan,
                        error=f"No practices with >= {min_practice_obs} observations")
        
        # fit model
        formula = f"items ~ {var}" + (" + C(month)" if deseasonalise_output else "")
        res = fit_mixed_effects(df_model_clean, formula, re_formula=re_formula)
        res["name"] = var
        return res

    # run tasks in parallel if requested
    status(f"Running mixed-effects models for {len(tasks)} variables (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs)(delayed(run_task)(var) for var in tqdm(tasks))

    # configure save paths
    out_csv = os.path.join(results_folder, "mixed_effects_values_results.csv")
    out_txt = os.path.join(results_folder, "mixed_effects_values_results.txt")

    # save results
    results_df = pd.DataFrame(results)
    os.makedirs(results_folder, exist_ok=True)
    results_df.to_csv(out_csv, index=False)
    status(f"Results saved to {out_csv}")

    # pretty text output
    max_name_len = results_df["name"].str.len().max()
    max_coef_len = results_df["coef"].apply(lambda x: len(f"{x:.2f}") if pd.notna(x) else 4).max()
    max_ci_len = results_df[["ci_low","ci_high"]].map(lambda x: len(f"{x:.2f}") if pd.notna(x) else 4).max().max()
    max_pval_len = results_df["pval"].apply(lambda x: len(f"{x:.3g}") if pd.notna(x) else 4).max()
    with open(out_txt, "w") as f:
        for row in results_df.itertuples():
            name_str = row.name.ljust(max_name_len)
            coef_str = f"{row.coef:.2f}".rjust(max_coef_len) if pd.notna(row.coef) else "NaN".rjust(max_coef_len)
            ci_str = f"(CI: {row.ci_low:.2f}, {row.ci_high:.2f})".ljust(max_ci_len+10) if pd.notna(row.ci_low) else "(CI: NaN, NaN)".ljust(max_ci_len+16)
            pval_str = f"p = {row.pval:.3g}".rjust(max_pval_len+5) if pd.notna(row.pval) else "p = NaN".rjust(max_pval_len+5)
            error_str = f"** error: {row.error}" if pd.notna(row.error) and row.error != "" else ""
            f.write(f"{name_str}  {coef_str}  {ci_str}  {pval_str}  {error_str}\n")
    status(f"Pretty results saved to {out_txt}")

    return results_df

def run_bayesian_model(
    ds,
    raw_vars: list,
    results_folder: str,
    file_suffix: str = "",
    use_pca: bool = False,
    n_components: int = None,
    deseasonalise_output: bool = True,
    practice_correction: int = 1,
    min_practice_obs: int = 20,
    interactions: list = None,
    poly_terms: dict = None,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
):
    """
    Fit a hierarchical Bayesian model of prescription 'items' using raw environmental variables.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'items' and raw variables.
    raw_vars : list of str
        Raw variable names to include as predictors.
    results_folder : str
        Folder path for saving results.
    use_pca : bool
        If True, apply PCA to predictors and use PCs as covariates.
    n_components : int or None
        Number of PCA components; if None, keep all.
    seasonal_correction_out : bool
        Whether to include month-of-year seasonal correction terms for the output variable.
    seasonal_correction_in : bool
        Whether to apply month-of-year seasonal correction to the predictor variables.
    practice_correction : int
        Level of practice-specific random effects:
        0 = none
        1 = intercept only
        2 = intercept + slope
        3 = intercept + slope + correlation.
    adjust_inputs : str or None
        Method to adjust predictor inputs before modelling. Options include:
        'z-global': standardise values globally
        'z-practice': standardise per practice
        'c-global': centre globally
        'c-practice': centre per practice
        None: raw values
    standardise_items : bool
        Whether to standardise 'items' (per practice) before modelling.
    n_practices : int or None
        If specified, limit practices to this many with the most items (for testing).
    min_practice_obs : int
        Practices excluded if they have fewer than this number of observations.
    interactions : list of str
        Interactions to include, specified as "variable1 x variable2".
    poly_terms : dict
        Dictionary of {var_name: max_power} to create polynomial terms.
    draws, tune, chains : int
        Sampling parameters for Bambi.
    
    Returns
    -------
    model : bambi.Model
        Fitted model.
    idata : arviz.InferenceData
        Posterior draws.
    """
    # check output folder
    os.makedirs(results_folder, exist_ok=True)

    # prepare dataframe
    status("Preparing dataframe for model input...")
    df = ds[["items", "date_code"] + raw_vars].to_dataframe().reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df = df.dropna(subset=["items"]).copy()
    predictors = raw_vars.copy()

    # PCA if requested
    if use_pca:
        status("Applying PCA to raw variables...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[predictors])
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(X_scaled)
        pc_names = [f"PC{i+1}" for i in range(pcs.shape[1])]
        df_pca = pd.DataFrame(pcs, columns=pc_names, index=df.index)
        df = pd.concat([df, df_pca], axis=1)
        predictors = pc_names

    # polynomial terms
    if poly_terms is not None:
        status("Adding polynomial terms...")
        poly_predictors = []
        for var, power in poly_terms.items():
            if var not in df.columns:
                status(f"Warning: variable {var} not in dataframe, skipping polynomial term.")
                continue
            for p in range(2, power+1):
                col_name = f"{var}_pow{p}"
                df[col_name] = df[var]**p
                poly_predictors.append(col_name)
        predictors += poly_predictors

    # interaction terms
    interaction_terms = []
    if interactions and not use_pca:
        status("Adding interaction terms...")
        for inter in interactions:
            inter = inter.strip().replace(" ", "")
            if "*" in inter:
                interaction_terms.append(inter)
            else:
                raise ValueError(f"Interaction '{inter}' must be specified using '*' between variable names.")
        predictors += interaction_terms

    # build formula
    formula = "items ~ " + " + ".join(predictors)
    if deseasonalise_output:
        formula += " + C(month)"
    if practice_correction == 1:
        formula += " + (1 | practice_id)"  # intercept
    elif practice_correction == 2:
        formula += " + (1 | practice_id) + (0 + date_code | practice_id)"  # intercept + slope, uncorrelated
    elif practice_correction == 3:
        formula += " + (date_code | practice_id)"  # intercept + slope, correlated
    elif practice_correction != 0:
        raise ValueError("practice_correction must be 0, 1, 2, or 3")

    # clear out any nan predictors
    df = df.dropna(subset=predictors).copy()
    df = df[['items'] + predictors + ['practice_id', 'month', 'date_code']]

    # filter practices
    practice_counts = df.groupby('practice_id').size()
    valid_practices = practice_counts[practice_counts >= min_practice_obs].index
    df = df[df['practice_id'].isin(valid_practices)]
    if df.empty:
        raise ValueError("No practices with sufficient observations after filtering.")
    else:
        status(f"Using {len(valid_practices)} practices with >= {min_practice_obs} observations.")

    # fit Bayesian model
    status(f"Fitting Bambi model with formula '{formula}'...")
    model = bmb.Model(formula, df)
    # progress_callback = make_progress_callback(draws, tune)
    idata = model.fit(draws=draws,
                      tune=tune,
                      chains=chains,
                      cores=cores,)
                    #   progressbar=False,
                    #   callback=[progress_callback])

    # save summary
    summary_df = az.summary(idata)
    summary_csv = os.path.join(results_folder, f"bayesian_model_summary{file_suffix}.csv")
    summary_df.to_csv(summary_csv)
    status(f"Posterior summary saved to: {summary_csv}")

    # prettier text summary
    out_txt = os.path.join(results_folder, f"bayesian_model_summary{file_suffix}.txt")
    max_name_len = summary_df.index.str.len().max()
    with open(out_txt, "w") as f:
        for var, row in summary_df.iterrows():
            mean = row['mean']
            hdi_3pc = row['hdi_3%']
            hdi_97pc = row['hdi_97%']
            f.write(f"{var.ljust(max_name_len)} : {mean:8.2f} (CI: {hdi_3pc:8.2f}, {hdi_97pc:8.2f})\n")
    status(f"Text summary saved to: {out_txt}")

    # save the model
    idata_path = os.path.join(results_folder, f"bayesian_model_idata{file_suffix}.nc")
    az.to_netcdf(idata, idata_path)
    status(f"Model results saved to: {idata_path}")

    return model, idata

# MODELING HELPERS --------------------------------------------------------------------------------
def prepare_ds(ds, n_practices=None, standardise_items=False, adjust_predictors=None, deseasonalise_predictors=False):
    """
    Prepare dataset for analysis by adding date_code, limiting practices and
    standardising variables/items, as requested.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with 'date' and 'practice_id' dimensions.
    n_practices : int or None
        If specified, limit to this many practices with most items.
    adjust_predictors : str or None
        Method to adjust predictor inputs before modelling. Options include:
        'z-global': standardise values globally
        'z-practice': standardise per practice
        'c-global': centre globally
        'c-practice': centre per practice
        None: raw values
    standardise_items : bool
        Whether to standardise 'items' per practice.
    deseasonalise_predictors : bool
        Whether to do seasonal correction on predictors.
    """
    # add date code as simpler data variable for models
    ds["date_code"] = ("date", np.arange(len(ds["date"])))  # integer months since start
    ds["date_code"] = (ds["date_code"] - ds["date_code"].mean()) / ds["date_code"].std()  # standardised

    # limit practices if requested
    if n_practices is not None:
        practice_counts = ds["items"].sum(dim="date").sortby(ds["items"].sum(dim="date"),
                                                             ascending=False)
        top_practices = practice_counts["practice_id"].values[:n_practices]
        status(f"Limiting to top {n_practices} practices with most items")
        ds = ds.sel(practice_id=top_practices)
    
    # adjust predictor variables if requested
    if adjust_predictors is not None:
        # per practice centering/standardising
        if adjust_predictors in ["c-practice", "z-practice"]:
            status("Adjusting predictor values per practice")
            for var in ds.data_vars:
                if var.endswith("_values"):
                    mean = ds[var].mean(dim="date")
                    ds[var] = ds[var] - mean
                    if adjust_predictors == "z-practice":
                        std = ds[var].std(dim="date") + 1e-8  # prevent div by zero
                        ds[var] = ds[var] / std
        # global centering/standardising
        elif adjust_predictors in ["c-global", "z-global"]:
            status("Adjusting predictor values globally")
            for var in ds.data_vars:
                if var.endswith("_values"):
                    mean = ds[var].mean().item()
                    ds[var] = ds[var] - mean
                    if adjust_predictors == "z-global":
                        std = ds[var].std().item() + 1e-8  # prevent div by zero
                        ds[var] = ds[var] / std
        else:
            raise ValueError(f"Unknown adjust_predictors method: {adjust_predictors}")
    
    # seasonal correction of predictors if requested
    if deseasonalise_predictors:
        status("Applying seasonal correction to predictor variables")
        for var in ds.data_vars:
            if var.endswith("_values"):
                monthly_means = ds[var].groupby("date.month").mean(dim="date")
                monthly_means_expanded = monthly_means.sel(month=ds["date.month"])
                ds[var] = ds[var] - monthly_means_expanded
    
    # standardise prescription 'items' per practice if requested
    if standardise_items:
        status("Standardising 'items' per practice")
        mean = ds["items"].mean(dim="date")
        std = ds["items"].std(dim="date") + 1e-8  # prevent div by zero
        ds["items"] = (ds["items"] - mean) / std

    return ds

def make_progress_callback(total_draws, total_tune):
    def progress_callback(trace, draw_idx, tune_idx, chain):
        pct = int((draw_idx + tune_idx + 1) / (total_draws + total_tune) * 100)
        if pct % 10 == 0:
            print(f"Chain {chain}, {pct}% complete", flush=True)
    return progress_callback



# ANALYSIS FUNCTIONS ==============================================================================
def compare_individual_analyses(results_folder, y_jitter=0.15, xlim_flag=(None, None), xlim_values=(None, None)):
    """Compare individual mixed effects model analyses across prescription types.
    Parameters
    ----------
    results_folder : str
        Root folder where results are stored.
    y_jitter : float, optional
        Amount to jitter y-axis for visibility on combined plots, by default 0.15
    xlim_flag : tuple, optional
        (min, max) for x-axis limits on flag plot; None for automatic, by default (None, None)
    xlim_values : tuple, optional
        (min, max) for x-axis limits on values plot; None for automatic, by default (None, None)
    """
    # prescription type definitions (hardcoded for now)
    prescription_codes = ['02_03_0501', '02', '03', '0501']
    labels = ['All', 'Cardiovascular', 'Respiratory', 'Antibiotics']
    colours = ['black', 'red', 'blue', 'orange']

    # =============================================================================================
    # load the data

    # set input and output paths correctly
    plot_root = f"{results_folder}"
    results_folders = [f"{results_folder}{c}/" for c in prescription_codes]

    # load data
    os.makedirs(plot_root, exist_ok=True)
    data_flag = []
    data_values = []
    datasets = []
    for folder in results_folders:
        try:
            df_flag = pd.read_csv(f"{folder}/mixed_effects_flag_results.csv")
            df_val = pd.read_csv(f"{folder}/mixed_effects_values_results.csv")
        except FileNotFoundError:
            print(f"Warning: could not find results in {folder}, skipping...")
            continue
        # skip empty results
        if df_flag.empty and df_val.empty:
            print(f"Warning: results in {folder} are empty, skipping...")
            continue
        df_flag["dataset"] = folder
        df_val["dataset"] = folder
        data_flag.append(df_flag)
        data_values.append(df_val)

    # safe concatenation: create empty DataFrames with expected columns if nothing was found
    if data_flag:
        df_flag_all = pd.concat(data_flag, ignore_index=True)
    else:
        df_flag_all = pd.DataFrame(columns=["name", "coef", "ci_low", "ci_high", "dataset"])

    if data_values:
        df_val_all = pd.concat(data_values, ignore_index=True)
    else:
        df_val_all = pd.DataFrame(columns=["name", "coef", "ci_low", "ci_high", "dataset"])

    # =============================================================================================
    # plot all on one figure

    for hide_sulfur in [True, False]:
        if not data_flag and not data_values:
            print("No data available for plotting 'all' figure; skipping.")
            continue

        # determine how many series we actually have
        n_flag_series = len(data_flag)
        n_val_series = len(data_values)
        labels_flag = labels[:n_flag_series]
        colours_flag = colours[:n_flag_series]
        labels_val = labels[:n_val_series]
        colours_val = colours[:n_val_series]

        # flagged + values in a 2-row figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)

        # flagged
        if n_flag_series > 0:
            last_df = None
            for i, (df, label, color) in enumerate(zip(data_flag, labels_flag, colours_flag)):
                if hide_sulfur:
                    df = df[~df["name"].str.contains("aqrean_daqi_sulfur_dioxide")]
                last_df = df
                y_pos = np.arange(len(df["name"])) + (i - n_flag_series / 2) * y_jitter
                axes[0].errorbar(df["coef"], y_pos,
                                    xerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
                                    fmt='o', color=color, label=label, markersize=3, capsize=2)
            if last_df is not None:
                axes[0].set_yticks(np.arange(len(last_df["name"])))
                axes[0].set_yticklabels(last_df["name"])
        else:
            axes[0].text(0.5, 0.5, 'No flagged results', ha='center', va='center')

        # values
        if n_val_series > 0:
            last_x = None
            for i, (df, label, color) in enumerate(zip(data_values, labels_val, colours_val)):
                x = df["name"].str.replace("_values", "")
                last_x = x
                y_pos = np.arange(len(df["name"])) + (i - n_val_series / 2) * y_jitter
                axes[1].errorbar(df["coef"], y_pos,
                                    xerr=[df["coef"] - df["ci_low"], df["ci_high"] - df["coef"]],
                                    fmt='o', color=color, label=label, markersize=3, capsize=2)
            if last_x is not None:
                axes[1].set_yticks(np.arange(len(last_x)))
                axes[1].set_yticklabels(last_x)
        else:
            axes[1].text(0.5, 0.5, 'No values results', ha='center', va='center')

        # plot formatting
        for ax in axes:
            ax.axvline(0, color='black', lw=0.8)
            ax.grid(True, linestyle=':', alpha=0.6)
        axes[0].set_xlim(xlim_flag)
        axes[1].set_xlim(xlim_values)
        axes[0].set_title("Flagged models")
        axes[1].set_title("Values models")
        axes[1].set_xlabel("Coefficient estimate")
        plt.tight_layout()
        
        # shared legend
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='lower center', ncol=4, frameon=False)
        fig.subplots_adjust(bottom=0.09)

        if hide_sulfur:
            plt.savefig(f"{plot_root}mixed_no_sulfur.png", dpi=300)
        else:
            plt.savefig(f"{plot_root}mixed.png", dpi=300)
        plt.close(fig)


    # =============================================================================================
    # plot coefficient estimates for flagged variables individually (comparing datasets)

    # if there are no flagged results, skip the per-variable flagged plots
    if df_flag_all.empty:
        print("No per-practice flagged results found; skipping flag variable plots.")
        flag_varnames = []
    else:
        flag_varnames = sorted(df_flag_all["name"].unique())
    os.makedirs(f"{plot_root}flags", exist_ok=True)
    for var in flag_varnames:
        fig, ax = plt.subplots(figsize=(6, 4))
        matched = False
        for df, label, color in zip(data_flag, labels, colours):
            sub = df[df["name"] == var]
            if not sub.empty:
                ax.errorbar(sub["coef"], label,
                            xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                            fmt='o', color=color, markersize=5, capsize=4)
                matched = True
        if not matched:
            plt.close(fig)
            continue
        ax.axvline(0, color='black', lw=0.8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(f"{var}")
        ax.set_xlabel("Coefficient estimate")
        fig.tight_layout()
        plt.savefig(f"{plot_root}flags/{var}.png", dpi=300)
        plt.close(fig)

    # =============================================================================================
    # plot coefficient estimates for values variables individually (comparing datasets)

    if df_val_all.empty:
        print("No values results found; skipping value variable plots.")
        val_varnames = []
    else:
        val_varnames = sorted(df_val_all["name"].str.replace("_values", "").unique())
    os.makedirs(f"{plot_root}values", exist_ok=True)
    for var in val_varnames:
        fig, ax = plt.subplots(figsize=(6, 4))
        matched = False
        for df, label, color in zip(data_values, labels, colours):
            sub = df[df["name"].str.replace("_values", "") == var]
            if not sub.empty:
                ax.errorbar(sub["coef"], label,
                            xerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                            fmt='o', color=color, markersize=5, capsize=4)
                matched = True
        if not matched:
            plt.close(fig)
            continue
        ax.axvline(0, color='black', lw=0.8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(f"{var}")
        ax.set_xlabel("Coefficient estimate")
        fig.tight_layout()
        plt.savefig(f"{plot_root}values/{var}.png", dpi=300)
        plt.close(fig)

def compare_bayesian_analyses(results_folder, y_jitter=0.15, generate_arviz_plots=True):
    '''Generates plots to compare the Bayesian analysis results across different prescription types.

    Parameters
    ----------
    results_folder : str
        Root folder containing results subfolders for each prescription type.
    y_jitter : float, optional
        Vertical jitter amount between overlapping categories in the combined plot.
    generate_arviz_plots : bool, optional
        Whether to generate ArviZ diagnostic plots for each model.
    '''
    # prescription type definitions (hardcoded for now)
    prescription_codes = ['02_03_0501', '02', '03', '0501']
    labels = ['All', 'Cardiovascular', 'Respiratory', 'Antibiotics']
    colours = ['black', 'red', 'blue', 'orange']

    # fields in the analysis
    value_vars = [
        "hydro_rain",
        "met_rain",
        "met_tmax",
        "aqrean_carbon_monoxide",
        "aqrean_daqi_overall",
        "aqrean_nitrogen_monoxide",
        "aqrean_nitrogen_dioxide",
        "aqrean_daqi_nitrogen_dioxide",
        "aqrean_nox_expressed_as_nitrogen_dioxide",
        "aqrean_ozone",
        "aqrean_daqi_ozone",
        "aqrean_pm2p5",
        "aqrean_pm10",
        "aqrean_daqi_pm10",
        "aqrean_sulfur_dioxide",
        "aqrean_daqi_sulfur_dioxide"
    ]
    value_vars = [ft + "_values" for ft in value_vars]

    # set input and output paths correctly
    results_roots = [
        f"{results_folder}{c}/" for c in prescription_codes
    ]
    plot_root = f"{results_folder}"

    # load data
    idatas = []
    dataframes = []
    for root in results_roots:
        csv_path = root + f"bayesian_model_summary.csv"
        if not os.path.exists(csv_path):
            print(f"Warning: could not find results at {csv_path}, skipping...")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: results in {csv_path} are empty, skipping...")
            continue
        df = df.rename(columns={"Unnamed: 0": "name"})
        dataframes.append(df)

    if not dataframes:
        raise FileNotFoundError("No Bayesian results CSVs found — nothing to plot.")

    # =================================================================================================
    # all in one figure

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (df, label, color) in enumerate(zip(dataframes, labels, colours)):
        if df.empty:  # skip missing ones
            continue

        # get data for this var
        df = df[df["name"].isin(value_vars)].copy()
        df = df.set_index("name").reindex(value_vars)

        # set jitter position and plot
        y_pos = np.arange(len(value_vars)) + (i - len(dataframes) / 2) * y_jitter
        ax.errorbar(
            df["mean"], y_pos,
            xerr=[df["mean"] - df["hdi_3%"], df["hdi_97%"] - df["mean"]],
            fmt='o', color=color, label=label, markersize=4, capsize=3
        )

    # plot formatting
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(np.arange(len(value_vars)))
    ax.set_yticklabels(value_vars)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel("Posterior mean (effect size)")
    ax.set_title("Bayesian model coefficients comparison")
    plt.tight_layout()
    
    # shared legend
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.09)
    plt.savefig(plot_root + f"bayesian.png", dpi=300)
    plt.close(fig='all')

    # =================================================================================================
    # individual variable plots

    plot_folder = f"{plot_root}values_bayes/"
    os.makedirs(plot_folder, exist_ok=True)
    for var in value_vars:
        fig, ax = plt.subplots(figsize=(6, 4))
        matched = False
        for df, label, color in zip(dataframes, labels, colours):
            if df.empty:  # skip missing ones
                continue

            # extract var and plot
            sub = df[df["name"] == var]
            if not sub.empty:
                ax.errorbar(
                    sub["mean"], label,
                    xerr=[sub["mean"] - sub["hdi_3%"], sub["hdi_97%"] - sub["mean"]],
                    fmt='o', color=color, markersize=5, capsize=4
                )
                matched = True
        if not matched:
            plt.close(fig='all')
            continue

        # plot formatting and save
        ax.axvline(0, color='black', lw=0.8)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_title(f"{var}")
        ax.set_xlabel("Posterior mean (effect size)")
        fig.tight_layout()
        plt.savefig(f"{plot_folder}{var}.png", dpi=300)
        plt.close(fig='all')

    print(f"Plots saved to {plot_root}")

    # =================================================================================================
    # generate ArviZ diagnostic plots for each model

    if generate_arviz_plots:
        az.rcParams["plot.max_subplots"] = 40
        for root in results_roots:
            # get the inference data
            nc_path = os.path.join(root, f"bayesian_model_idata.nc")
            if not os.path.exists(nc_path):
                print(f"Warning: no netCDF file found for at {nc_path}, skipping...")
                continue
            idata = az.from_netcdf(nc_path)
            arviz_out_dir = root + f"bayesian_diagnostics/"
            os.makedirs(arviz_out_dir, exist_ok=True)
            posterior_vars = [v for v in idata.posterior.data_vars
                            if "|" not in v and "C(" not in v]

            # convergence plot
            fig = az.plot_trace(idata, var_names=posterior_vars)
            plt.tight_layout()
            plt.savefig(os.path.join(arviz_out_dir, "convergence.png"), dpi=200)
            plt.close(fig='all')

            # posterior plot
            fig = az.plot_posterior(
                idata,
                var_names=posterior_vars,
                hdi_prob=0.94,
                point_estimate="mean",
                kind="hist",
            )
            plt.tight_layout()
            plt.savefig(os.path.join(arviz_out_dir, "posterior.png"), dpi=200)
            plt.close(fig='all')

            # rank plots
            fig = az.plot_rank(idata, var_names=posterior_vars)
            plt.tight_layout()
            plt.savefig(os.path.join(arviz_out_dir, "rank.png"), dpi=200)
            plt.close(fig='all')

            # posterior predictive checks
            if "posterior_predictive" in idata.groups():
                fig = az.plot_ppc(idata, num_pp_samples=100)
                plt.tight_layout()
                plt.savefig(os.path.join(arviz_out_dir, "ppc.png"), dpi=200)
                plt.close(fig='all')
            else:
                print(f"Warning: no posterior predictive data foundin {nc_path}, skipping PPC.")
            
            print(f"Analysis diagnostic plots saved to {arviz_out_dir}")



# GENERAL HELPERS =================================================================================
def status(*message):
    '''Works like the Python print function but preceded by current datetime in a similar
       format to tensorflow logging.'''
    message = " ".join([str(m) for m in message])
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f:"), message, flush=True)

def standardise_mad(values):
    """Compute Median Absolute Deviation of a 2D array with dimensions (date, practice_id)."""
    median = np.nanmedian(values, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=0, keepdims=True)
    return (values - median) / (mad * 1.4826 + 1e-8)











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

def remove_seasonal_effects_old(datetimes, values):
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
