import os, json, ast
import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pymc as pm
import pytensor.sparse as pts
import scipy.sparse as sp
from numpy.linalg import qr
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from datetime import datetime
from pyproj import Transformer
from tqdm import tqdm
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
from typing import List, Dict, Tuple

PRES_CODES = ['02_03_0501', '02', '03', '0501']
PRES_LABELS = ['All', 'Cardiovascular', 'Respiratory', 'Antibiotics']
PRES_COLOURS = ['black', 'red', 'blue', 'orange']
# ordered north->south, east->west with "South East" last (reference category)
REGION_NAMES = ["North East", "North West", "Yorkshire and The Humber",
                "East Midlands", "West Midlands", "East of England",
                "London", "South West", "South East"]
PRACTICE_SIZES = ["small", "large"]
# ordered months with "September" last (reference category)
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

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

def add_geojson_flood_flags(prescriptions_ds, geojson_features, search_radius_m=5000, simplify_tol=50):
    """
    Add flood flags to the dataset based on geojson polygons.
    """
    lat_vec = prescriptions_ds.coords['latitude'].values
    lon_vec = prescriptions_ds.coords['longitude'].values
    earliest_year = prescriptions_ds['date'].dt.year.min().item()

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
        if pd.isna(start) or pd.isna(end) or end.year < earliest_year:
            continue
        geom = shape(f["geometry"])
        geom = geom.simplify(simplify_tol, preserve_topology=False)
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
                  observed_properties=["tmax", "tmin", "rain"],
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
        status("Adding MET flags for", observed_property, level=1)
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

def add_deprivation_index(prescriptions_ds, deprivation_ds):
    """
    Add imd_centile_values and imd_lad_code to prescriptions_ds by matching each
    (date, practice_id) to the nearest LAD centroid in deprivation_ds for the
    nearest deprivation year.

    Assumes:
      - prescriptions_ds.latitude/practices_ds.longitude are coords aligned with practice_id
      - prescriptions_ds.date is datetime64[ns]
      - deprivation_ds has dims (LAD_code, year) and coords:
            deprivation_ds['latitude'] (LAD_code, year)
            deprivation_ds['longitude'] (LAD_code, year)
            deprivation_ds['year'] (year)
            deprivation_ds['LAD_code'] (LAD_code)   (coordinate labels)
        and data variable 'centile' with shape (LAD_code, year)
    """

    # deprivation data
    dep_years = deprivation_ds["year"].values            # shape (n_dep_years,)
    dep_lad_codes = deprivation_ds["LAD_code"].values    # shape (n_dep_lads,)
    dep_lat_arr = deprivation_ds["latitude"].values      # shape (n_dep_lads, n_dep_years)
    dep_lon_arr = deprivation_ds["longitude"].values     # shape (n_dep_lads, n_dep_years)
    dep_cent_arr = deprivation_ds["centile"].values      # shape (n_dep_lads, n_dep_years)
    n_dep_years = dep_years.shape[0]
    n_dep_lads = dep_lad_codes.shape[0]

    # practice data
    pres_prac_lats = prescriptions_ds["latitude"].values       # shape (n_pres_prac,)
    pres_prac_lons = prescriptions_ds["longitude"].values      # shape (n_pres_prac,)
    pres_date_years = prescriptions_ds["date"].dt.year.values  # shape (n_pres_dates,)
    n_pres_prac = pres_prac_lats.shape[0]
    n_pres_dates = pres_date_years.shape[0]
    prac_coords = np.column_stack([pres_prac_lats, pres_prac_lons])  # (n_pres_prac, 2)

    # Precompute nearest LAD index (original index into LAD_code axis) for every practice x deprivation-year
    # Initialize with -1 for "no valid LAD"
    lad_idx_per_practice_and_year = np.full((n_pres_prac, n_dep_years), -1, dtype=int)

    for j in range(n_dep_years):
        # LAD coordinates for year j
        lat_j = dep_lat_arr[:, j]   # shape (n_dep_lads,)
        lon_j = dep_lon_arr[:, j]   # shape (n_dep_lads,)

        valid_lad_mask = np.isfinite(lat_j) & np.isfinite(lon_j)
        if not np.any(valid_lad_mask):
            continue

        valid_lad_indices = np.where(valid_lad_mask)[0]   # original LAD indices
        lad_coords_j = np.column_stack([lat_j[valid_lad_mask], lon_j[valid_lad_mask]])  # (n_valid_lads, 2)

        # build KDTree for this year's valid LAD centroids
        tree = cKDTree(lad_coords_j)
        _, idx = tree.query(prac_coords)   # idx into lad_coords_j

        # map back to original LAD indices
        original_lad_idx = valid_lad_indices[idx]  # (n_pres_prac,)
        lad_idx_per_practice_and_year[:, j] = original_lad_idx

    # find index of closest deprivation year to each prescription date year
    # e.g. if pres_date_years = [2015, 2016, 2017, 2018, 2019]
    # and dep_years = [2015, 2019, 2025]
    # then [0, 0, 0, 1, 1]
    closest_dep_year_idx_per_date = np.argmin(np.abs(pres_date_years[:, None] - dep_years[None, :]), axis=1)  # (n_pres_dates,)

    # prepare output arrays
    imd_centile = np.full((n_pres_dates, n_pres_prac), np.nan, dtype=np.float32)
    imd_lad_code = np.full((n_pres_dates, n_pres_prac), "", dtype=object)
    imd_lon = np.full((n_pres_dates, n_pres_prac), np.nan, dtype=np.float32)
    imd_lat = np.full((n_pres_dates, n_pres_prac), np.nan, dtype=np.float32)

    # for each deprivation year j, fill rows for dates that map to j
    for j in range(n_dep_years):
        date_mask = (closest_dep_year_idx_per_date == j)
        if not np.any(date_mask):
            continue  # no dates map to this deprivation year

        # lad indices per practice for this year (original lad index or -1)
        lad_idx_for_pracs = lad_idx_per_practice_and_year[:, j]  # shape (n_pres_prac,)

        # where lad_idx_for_pracs == -1 we should keep NaN/''.
        valid_prac_mask_for_year = lad_idx_for_pracs >= 0
        if np.any(valid_prac_mask_for_year):
            # centiles for those LADs at year j
            cent_for_pracs = np.full(n_pres_prac, np.nan, dtype=np.float32)
            lats_for_pracs = np.full(n_pres_prac, np.nan, dtype=np.float32)
            lons_for_pracs = np.full(n_pres_prac, np.nan, dtype=np.float32)
            ladcode_for_pracs = np.full(n_pres_prac, "", dtype=object)

            valid_lad_original_idx = lad_idx_for_pracs[valid_prac_mask_for_year]  # original lad indices
            # fetch centile values from cent_arr using advanced indexing
            cent_for_pracs[valid_prac_mask_for_year] = dep_cent_arr[valid_lad_original_idx, j]
            lats_for_pracs[valid_prac_mask_for_year] = dep_lat_arr[valid_lad_original_idx, j]
            lons_for_pracs[valid_prac_mask_for_year] = dep_lon_arr[valid_lad_original_idx, j]
            ladcode_for_pracs[valid_prac_mask_for_year] = dep_lad_codes[valid_lad_original_idx]

            # assign to all dates that map to this year (broadcast over rows)
            imd_centile[np.ix_(date_mask, np.arange(n_pres_prac))] = cent_for_pracs[None, :]
            imd_lad_code[np.ix_(date_mask, np.arange(n_pres_prac))] = ladcode_for_pracs[None, :]
            imd_lat[np.ix_(date_mask, np.arange(n_pres_prac))] = lats_for_pracs[None, :]
            imd_lon[np.ix_(date_mask, np.arange(n_pres_prac))] = lons_for_pracs[None, :]

    # add to prescriptions dataset
    prescriptions_ds = prescriptions_ds.copy()
    prescriptions_ds["imd_centile_values"] = (("date", "practice_id"), imd_centile)
    prescriptions_ds["imd_lad_code"] = (("date", "practice_id"), imd_lad_code)
    prescriptions_ds["imd_latitude"] = (("date", "practice_id"), imd_lat)
    prescriptions_ds["imd_longitude"] = (("date", "practice_id"), imd_lon)

    return prescriptions_ds

def add_practice_regions(prescriptions_ds, regions_geojson):
    """
    Add region codes to prescriptions_ds by matching each practice
    to the region polygon it falls within.

    Assumes:
      - prescriptions_ds.latitude/longitude are coords aligned with practice_id
      - regions_geojson is a list of geojson features with 'geometry' and 'properties' containing:
          - "nuts115nm" : region name like "North East (England)"
    
    link: https://geoportal.statistics.gov.uk/datasets/44c039e762d94a42bf5e0580e8dd9f84_0/explore?location=53.000805%2C-2.813670%2C6.58
    """

    # practice data
    pres_prac_lats = prescriptions_ds["latitude"].values
    pres_prac_lons = prescriptions_ds["longitude"].values
    n_pres_prac = pres_prac_lats.shape[0]

    # preprocess region polygons
    geoms = []
    region_names = []
    for f in tqdm(regions_geojson,
                  desc="      Processing region polygons",
                  total=len(regions_geojson)):
        geom = shape(f["geometry"])
        region_name = f["properties"].get("nuts115nm", "")
        region_name = region_name.replace(" (England)", "").strip()
        geoms.append(geom)
        region_names.append(region_name)

    # build spatial index
    tree = STRtree(geoms)

    # assign region names to practices
    pres_region_names = np.full(n_pres_prac, "", dtype=object)
    for i, (lat, lon) in enumerate(zip(pres_prac_lats, pres_prac_lons)):
        pt = Point(lon, lat)
        candidates = tree.query(pt)
        assigned = False
        for j in candidates:
            if geoms[j].contains(pt) or geoms[j].touches(pt):
                pres_region_names[i] = region_names[j]
                assigned = True
                break
        
        if not assigned:
            distances = [geoms[j].distance(pt) for j in candidates]
            pres_region_names[i] = region_names[candidates[np.argmin(distances)]]

    # add to prescriptions dataset
    prescriptions_ds = prescriptions_ds.copy()
    prescriptions_ds["region"] = (("practice_id"), pres_region_names)

    return prescriptions_ds

# DATA HELPERS ------------------------------------------------------------------------------------
def download_file(url, session, out_dir='', timeout=20, overwrite=False):
    """Download file streaming to disk. Skip if already exists."""
    fname = url.split("/")[-1]
    out_path = out_dir / fname

    if out_path.exists() and not overwrite:
        # optionally check file size to avoid partial downloads
        if out_path.stat().st_size > 1_000_000:  # >1MB sanity check
            status(f"Skipping already downloaded ZIP: {fname}", level=1)
            return out_path
        else:
            status(f"Re-downloading incomplete ZIP: {fname}", level=1)

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

def standardise_mad(values):
    """Compute Median Absolute Deviation of a 2D array with dimensions (date, practice_id)."""
    median = np.nanmedian(values, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=0, keepdims=True)
    return (values - median) / (mad * 1.4826 + 1e-8)

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
        plt.close(fig="all")

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

def plot_deprivation_lad_map(prescriptions_ds):
    inds = np.random.choice(np.arange(len(prescriptions_ds["practice_id"])), size=20, replace=False)
    colors = plt.get_cmap('tab20').colors
    prac_lons = prescriptions_ds["longitude"].values[inds]
    prac_lats = prescriptions_ds["latitude"].values[inds]
    lad_lons = prescriptions_ds["imd_longitude"].values[:, inds]
    lad_lats = prescriptions_ds["imd_latitude"].values[:, inds]
    plt.figure(figsize=(10,10))
    for i, ind in enumerate(inds):
        plt.scatter(prac_lons[i], prac_lats[i], color=colors[i], label=f'Practice {ind}', marker='o', alpha=0.3)
        plt.scatter(lad_lons[:, i], lad_lats[:, i], color=colors[i], marker='x', alpha=0.3)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.savefig("test.png")

def plot_regions_map(prescriptions_ds, regions_geojson, outpath="outputs/practice_region_map.png"):
    # get practice coordinates and regions
    lats = np.asarray(prescriptions_ds['latitude'].values)
    lons = np.asarray(prescriptions_ds['longitude'].values)
    regions = np.asarray(prescriptions_ds['region'].values, dtype=str)

    # unique region names in a reproducible order (sorted or preserve original ordering)
    region_names = sorted(np.unique(regions))
    N = len(region_names)
    region_to_num = {name: i for i, name in enumerate(region_names)}
    region_nums = np.array([region_to_num.get(r, -1) for r in regions])

    # prepare a discrete colormap with exactly N colors
    cmap = plt.cm.get_cmap("tab20", N)   # discrete sampling of tab20
    # create ListedColormap so colors are stable and indexable
    listed_cmap = ListedColormap([cmap(i) for i in range(N)])

    # Create norm and scalar mappable for colorbar with discrete segments
    bounds = np.arange(-0.5, N + 0.5, 1.0)        # boundaries between color bins
    norm = BoundaryNorm(bounds, listed_cmap.N)
    sm = ScalarMappable(cmap=listed_cmap, norm=norm)
    sm.set_array([])  # needed for colorbar

    # plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # If the geojson is the whole dict with 'features', accept that too.
    features_iter = regions_geojson.get("features", regions_geojson) if isinstance(regions_geojson, dict) else regions_geojson

    # draw polygons (regions) using the same color mapping
    for feature in features_iter:
        geom = shape(feature['geometry'])
        # attempt to match your property name for region label:
        region_name = feature['properties'].get("nuts115nm", "").replace(" (England)", "").strip()
        region_num = region_to_num.get(region_name, None)
        if region_num is None:
            # skip features that don't match any region name
            continue
        color = listed_cmap(region_num)
        ax.add_geometries([geom], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='k', linewidth=0.2, alpha=0.5)

    # plot practices (use only points with valid region mapping)
    valid_mask = region_nums >= 0
    sc = ax.scatter(lons[valid_mask], lats[valid_mask],
                    c=region_nums[valid_mask],
                    cmap=listed_cmap,
                    norm=norm,
                    s=10, marker='o',
                    transform=ccrs.PlateCarree(), zorder=3, linewidths=0)

    # colorbar: ticks centered on each band
    cbar = plt.colorbar(sm, ax=ax, boundaries=bounds, ticks=np.arange(N))
    # set tick labels to region names and rotate or wrap if needed
    cbar.set_ticklabels(region_names)
    cbar.ax.tick_params(length=0)  # hide ticks if preferred

    # optionally move labels to center - ticks already centered by above
    # cbar.ax.set_yticklabels(region_names, rotation=0, va='center')

    # set xlims and ylims to England
    ax.set_extent([-6, 2, 49, 61], crs=ccrs.PlateCarree())

    # tidy and save
    plt.title("Practice regions")
    plt.savefig(outpath, bbox_inches='tight', dpi=600)
    plt.close(fig)



# MODELING FUNCTIONS ==============================================================================
def run_mixed_effects_models(
    ds,
    predictors_spec: List[Dict],
    results_folder: str,
    deseasonalise_output: bool = False,
    practice_correction: int = 1,
    min_practice_obs: int = 20,
    n_jobs: int = 1,
):
    """
    Unified runner for mixed-effects models on log1p(items).
    Supports:
      - binary_simple: single column (e.g. flood)
      - binary_pair: compare high vs median (e.g. hydro_rain_high vs hydro_rain_median)
      - daqi_pair1 / daqi_pair2: DAQI groupings
      - continuous: single continuous predictor
    Generates model with formula like:
        items_log ~ predictor[*imd_centile_values] [+ predictor*C(region)] + C(practice_size) [+ C(month)]

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing monthly 'items', date_code, imd_centile_values and the predictor variables.
        Indexed by (date, practice_id) when converted to dataframe.
    predictors_spec : list of dict
        Each dict describes one predictor to test. Required keys depend on 'type'.
        Examples:
          {"name": "flood", "type": "binary_simple", "var": "flood"}
          {"name": "rain_high_vs_median", "type": "binary_pair", "var_anom": "hydro_rain_high", "var_med": "hydro_rain_median"}
          {"name": "aq_daqi_pair1", "type": "daqi_pair1", "vars": {"very_high": "...", "high": "...", "moderate": "...", "low": "..."}}
          {"name": "pm10", "type": "continuous", "var": "aqrean_pm10_values"}
    results_folder : str
        Directory where CSV and text outputs will be saved (subfolders not used).
    deseasonalise_output : bool
        Add month fixed effects (C(month)) to the formula.
    practice_correction : int
        0 = no random effects, 1 = random intercept, 2 = random intercept + slope on date_code
    min_practice_obs : int
        Minimum monthly observations per practice required.
    n_jobs : int
        Number of parallel jobs (joblib).
    """

    os.makedirs(results_folder, exist_ok=True)

    # build base dataframe
    status("Preparing DataFrame for modeling...", level=1)
    df_items = (
    ds[["items", "date_code", "practice_size", "region", "imd_centile_values"]]
        .to_dataframe()
        .reset_index()
    )
    # create month column for seasonal correction
    df_items["date"] = pd.to_datetime(df_items["date"])
    df_items["month"] = df_items["date"].dt.month
    # transform prescription items
    df_items = df_items.rename(columns={"items": "items_raw"})
    df_items["items_log"] = np.log1p(df_items["items_raw"])
    # get indices
    df_items = df_items.set_index(["date", "practice_id"])
    df_index = df_items.index

    # random effects formula
    if practice_correction == 0:
        re_formula = None
    elif practice_correction == 1:
        re_formula = "~1"
    elif practice_correction == 2:
        re_formula = "~1 + date_code"
    else:
        raise ValueError("practice_correction must be 0, 1 or 2")

    # for checking if variables exist
    def var_exists(*vars_to_check) -> Tuple[bool, List[str]]:
        missing = [v for v in vars_to_check if v not in ds]
        return (len(missing) == 0, missing)

    # extract confidence intervals safely
    def safe_ci(mdf, term):
        try:
            lo, hi = mdf.conf_int().loc[term]
            return float(lo), float(hi)
        except Exception:
            return np.nan, np.nan

    # convert log-coef to percent change
    def percent_from_log(beta):
        if pd.isna(beta):
            return np.nan
        return float(100.0 * (np.exp(beta) - 1.0))

    # fit wrapper
    def fit_mixed(formula, df_model, group_col="practice_id", re_formula_local=re_formula):
        try:
            md = smf.mixedlm(formula, df_model, groups=df_model[group_col],
                             re_formula=re_formula_local)
            mdf = md.fit(method="lbfgs", reml=True, disp=False)
            return mdf, None
        except Exception as e:
            return None, str(e)

    # fitting function for single predictor
    def fit_spec(spec: Dict):
        """
        Each spec must include a 'name' and 'type' key. Other required keys depend on type.
        Returns a dict of outputs for that predictor.
        """
        out = {
            "name": spec.get("name"),
            "type": spec.get("type"),

            # predictor
            "pred_coef": np.nan,
            "pred_ci_low": np.nan,
            "pred_ci_high": np.nan,
            "pred_p": np.nan,

            # imd
            "imd_coef": np.nan,
            "imd_ci_low": np.nan,
            "imd_ci_high": np.nan,
            "imd_p": np.nan,

            # region effects: store dictionaries indexed by category
            "region_main": {},             # region → {coef, ci_low, ci_high, p}

            # metadata
            "n_practices": 0,
            "n_obs": 0,
            "error": None,
        }
        typ = spec.get("type")

        # construct df_model with required predictor column(s)
        df_model = df_items[["items_raw", "items_log",
                             "month", "date_code",
                             "practice_size", "region",
                             "imd_centile_values"]].copy()
        df_model["practice_id"] = df_model.index.get_level_values("practice_id")

        # handling for the various predictor types
        if typ == "binary_simple":
            var = spec.get("var")
            ok, missing = var_exists(var)
            if not ok:
                out["error"] = f"missing variable(s): {missing}"
                return out
            s = ds[var].to_dataframe()[var].reindex(df_index)
            # 1->1, 0->0, else NaN
            df_model["flag_binary"] = np.where(s == 1, 1,
                                               np.where(s == 0, 0, np.nan))
            pred_term = "flag_binary"

        elif typ == "binary_pair":
            var_anom = spec.get("var_anom")
            var_med = spec.get("var_med")
            ok, missing = var_exists(var_anom, var_med)
            if not ok:
                out["error"] = f"missing variable(s): {missing}"
                return out
            s_high = ds[var_anom].to_dataframe()[var_anom].reindex(df_index)
            s_med = ds[var_med].to_dataframe()[var_med].reindex(df_index)
            # flag: 1 if high==1, 0 if med==1, else NaN
            df_model["flag_binary"] = np.where(s_high == 1, 1,
                                               np.where(s_med == 1, 0, np.nan))
            pred_term = "flag_binary"

        elif typ in ("daqi_pair1", "daqi_pair2"):
            vars_dict = spec.get("vars", {})
            vh = vars_dict.get("very_high")
            h = vars_dict.get("high")
            m = vars_dict.get("moderate")
            l = vars_dict.get("low")
            ok, missing = var_exists(vh, h, m, l)
            if not ok:
                out["error"] = f"missing DAQI vars: {missing}"
                return out
            df_daqi = pd.DataFrame(index=df_index)
            df_daqi["vh"] = ds[vh].to_dataframe()[vh].reindex(df_index)
            df_daqi["h"]  = ds[h].to_dataframe()[h].reindex(df_index)
            df_daqi["m"]  = ds[m].to_dataframe()[m].reindex(df_index)
            df_daqi["l"]  = ds[l].to_dataframe()[l].reindex(df_index)
            any_known = (~df_daqi[["vh","h","m","l"]].isna()).any(axis=1)

            flag_high = np.where((df_daqi["vh"] == 1) | (df_daqi["h"] == 1), 1, 0)
            flag_mod  = np.where(df_daqi["m"] == 1, 1, 0)
            flag_low  = np.where(df_daqi["l"] == 1, 1, 0)

            if typ == "daqi_pair1":
                # (high+vhigh) vs (low+moderate)
                cond1 = (flag_high == 1)
                cond0 = (flag_low == 1) | (flag_mod == 1)
            else:
                # pair2: (high+vhigh+moderate) vs low
                cond1 = (flag_high == 1) | (flag_mod == 1)
                cond0 = (flag_low == 1)

            flag_binary = np.where(cond1, 1, np.where(cond0, 0, np.nan))
            # mask unknowns
            flag_binary = np.where(any_known, flag_binary, np.nan)
            df_model["flag_binary"] = flag_binary
            pred_term = "flag_binary"

        elif typ == "continuous":
            var = spec.get("var")
            ok, missing = var_exists(var)
            if not ok:
                out["error"] = f"missing variable(s): {missing}"
                return out
            df_model[var] = ds[var].to_dataframe()[var].reindex(df_index)
            pred_term = var

        else:
            out["error"] = f"unknown spec type '{typ}'"
            return out

        # add practice covariates
        formula = f"items_log ~ {pred_term} + imd_centile_values"
        formula += " + C(region, Treatment(reference=\"South East\")) + C(practice_size)"

        # seasonal correction term (if requested)
        if deseasonalise_output:
            formula = formula + " + C(month)"

        # restrict df_model to required columns
        required_cols = ["items_log", pred_term, "imd_centile_values",
                         "region", "practice_size"]
        df_model = df_model.dropna(subset=required_cols).copy()

        # filter to practices with enough observations
        df_model = df_model.reset_index(drop=True)
        counts = df_model.groupby("practice_id").size()
        valid_pracs = counts[counts >= min_practice_obs].index
        df_model = df_model[df_model["practice_id"].isin(valid_pracs)].copy()
        out["n_practices"] = int(df_model["practice_id"].nunique())
        out["n_obs"] = int(len(df_model))
        if df_model.empty:
            out["error"] = f"No practices with >= {min_practice_obs} observations"
            return out

        # fit model
        mdf, err = fit_mixed(formula, df_model, group_col="practice_id",
                             re_formula_local=re_formula)
        if err is not None:
            out["error"] = err
            return out
        
        # generate diagnostics
        model_folder = os.path.join(results_folder, out["name"])
        os.makedirs(model_folder, exist_ok=True)
        generate_mixed_effects_diagnostics(mdf, df_model, model_folder, out["name"])

        # predictor effect (converted to percentage)
        t_pred = pred_term
        beta = mdf.params.get(t_pred)
        out["pred_coef"] = percent_from_log(beta)
        cis = tuple(percent_from_log(x) for x in safe_ci(mdf, t_pred))
        out["pred_ci_low"] = cis[0]
        out["pred_ci_high"] = cis[1]
        out["pred_p"] = float(mdf.pvalues.get(t_pred, np.nan))

        # imd effect
        beta = mdf.params.get("imd_centile_values", np.nan)
        out["imd_coef"] = percent_from_log(beta)
        cis = tuple(percent_from_log(x) for x in safe_ci(mdf, "imd_centile_values"))
        out["imd_ci_low"] = cis[0]
        out["imd_ci_high"] = cis[1]
        out["imd_p"] = float(mdf.pvalues.get("imd_centile_values", np.nan))

        # region effects
        for term in mdf.params.index:
            if term.startswith("C(region"):
                region = term.split("[T.")[1][:-1]
                beta = mdf.params.get(term, np.nan)
                cis = tuple(percent_from_log(x) for x in safe_ci(mdf, term))
                out["region_main"][region] = {
                    "coef": percent_from_log(beta),
                    "ci_low": cis[0],
                    "ci_high": cis[1],
                    "p": float(mdf.pvalues.get(term, np.nan)),
                }

        return out

    # run models in parallel
    status(f"Running {len(predictors_spec)} model(s) (n_jobs={n_jobs})...", level=1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_spec)(spec) for spec in tqdm(predictors_spec)
    )

    # save results
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(results_folder, "mixed_effects_results.csv")
    results_df.to_csv(csv_path, index=False)
    status(f"Saved results to {csv_path}", level=1)

    return results_df

def run_bayesian_model(
    ds,
    raw_vars: list,
    results_folder: str,
    no_lag_vars: list = [],
    no_time_vars: list = [],
    lag: int = 0,
    almon_order: int = 1,
    deseasonalise_output: bool = True,
    practice_correction: int = 1,
    min_practice_obs: int = 20,
    min_obs_for_slope: int = 50,
    likelihood: str = "normal",
    use_gpu: bool = False,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    cores: int = 4,
):
    """
    Fit a hierarchical Bayesian model of prescription 'items' with optional lagged predictors.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing monthly 'items', date_code, practice covariates and the predictor variables.
    raw_vars : list
        List of predictor variable names (strings) in ds to include in the model.
    results_folder : str
        Directory where CSV and traceplot outputs will be saved.
    no_lag_vars : list
        List of predictor variable names (strings) in ds to include without lagging.
    no_time_vars : list
        List of predictor variable names (strings) in ds to use practice-level mean instead of time-varying effects.
    lag : int
        Number of lagged months to include for each predictor variable.
    almon_order : int
        Order of Almon lag polynomial to use if lag > 0.
    individual_priors : bool
        Use individual priors per predictor variable based on their stddev. Single Normal(0, 2) prior otherwise.
    deseasonalise_output : bool
        Include month fixed effects to deseasonalise output.
    practice_correction : int
        0 = no random effects, 1 = random intercept, 2 = random intercept + slope on date_code.
    min_practice_obs : int
        Minimum monthly observations per practice required.
    likelihood : str
        Likelihood to use: "normal" or "studentt".
    use_gpu : bool
        Use GPU acceleration if available.
    draws : int
        Number of MCMC draws per chain.
    tune : int
        Number of tuning steps per chain.
    chains : int
        Number of MCMC chains.
    cores : int
        Number of CPU cores to use.
    """
    os.makedirs(results_folder, exist_ok=True)

    # prepare dataframe ===========================================================================
    status("Preparing dataframe for model input...", level=1)
    df = ds[
            ["items", "date_code", "region", "practice_size"] + raw_vars
        ].to_dataframe().reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df = df.rename(columns={"items": "items_raw"})
    df["items_log"] = np.log1p(df["items_raw"])
    df = df.dropna(subset=["items_log"] + raw_vars).copy()

    # filter practices
    practice_counts = df.groupby('practice_id').size()
    min_practice_obs += lag  # need extra obs for lagged vars
    valid_practices = practice_counts[practice_counts >= min_practice_obs].index
    df = df[df['practice_id'].isin(valid_practices)]
    if df.empty:
        raise ValueError("no practices with sufficient observations after filtering")
    status(f"Using {len(valid_practices)} practices with >= {min_practice_obs} observations",
           level=1)
    
    if no_time_vars:
        status(f"Collapsing no-time variables to practice means: {no_time_vars}", level=1)
        practice_means = (
            df.groupby("practice_id")[no_time_vars]
            .mean()
            .reset_index()
        )
        df = df.drop(columns=no_time_vars)
        df = df.merge(practice_means, on="practice_id", how="left")

    # generate lagged variables if lag > 0
    lag_vars = [v for v in raw_vars
                if v not in no_lag_vars
                and v not in no_time_vars]
    if lag > 0:
        status(f"Creating lagged variables up to lag {lag}...", level=1)
        overlap = set(raw_vars) & set(no_lag_vars)
        if overlap:
            status(f"Treating these variables as non-lagged: {sorted(overlap)}", level=1)
        for var in lag_vars:
            for l in range(lag+1):
                df[f"{var}_lag{l}"] = df.groupby("practice_id")[var].shift(l)
        df = df.dropna(subset=[f"{v}_lag{l}" for v in lag_vars for l in range(lag+1)])

        # create Almon bases
        status(f"Creating almon lag basis of order {almon_order}...", level=1)
        def almon_weights(lags, order):
            X = np.vstack([lags**k for k in range(order+1)]).T
            Q, _ = qr(X)
            return Q
        lag_basis = np.arange(lag+1)
        almon_X = almon_weights(lag_basis, almon_order)

        # create Almon-basis columns for each variable
        for var in lag_vars:
            lagged_vals = df[[f"{var}_lag{l}" for l in range(lag+1)]].values
            M = lagged_vals @ almon_X
            for j in range(M.shape[1]):
                colname = f"{var}_almon_basis_{j}"
                df[colname] = M[:, j]

    # # identify continuous and categorical predictors
    # categorical_vars = []
    # if deseasonalise_output:
    #     categorical_vars.append("month")

    # # map categorical codes for CSV later
    # cat_code_maps = {}
    # for cat in categorical_vars:
    #     # reference category at end of categories list
    #     if cat == "month":
    #         ref = "September"
    #         num_ref = MONTHS.index(ref) + 1
    #         categories = [m for m in range(1, 13) if m != num_ref] + [num_ref]
    #         cat_code_maps[cat] = pd.Categorical(df[cat], categories=categories)
    #     else:
    #         cat_code_maps[cat] = pd.Categorical(df[cat])

    # # warn for missing categories
    # for cat, cat_obj in cat_code_maps.items():
    #     present = set(df[cat].unique())
    #     expected = list(cat_obj.categories)
    #     missing = [c for c in expected if c not in present]
    #     if missing:
    #         status(f"Warning: for categorical {cat}, these levels were NOT present in data: {missing}",
    #                level=1)

    # precompute and combine design matrics =======================================================
    status("Building combined design matrices and gathering statistics...", level=1)
    fixed_cols = []  # for column names in X_fixed
    X_fixed_list = []  # list for numpy arrays in X_fixed

    # continuous predictors
    if lag == 0:
        for var in raw_vars:
            if var in df.columns:
                fixed_cols.append(var)
                X_fixed_list.append(df[var].values.astype("float32").reshape(-1, 1))
    elif lag > 0:
        for var in lag_vars:
            basis_cols = [
                c for c in df.columns
                if c.startswith(f"{var}_almon_basis_")
            ]
            if not basis_cols:
                continue
            M_var = df[basis_cols].values.astype("float32")
            fixed_cols.extend(basis_cols)
            X_fixed_list.append(M_var)
        for var in no_lag_vars:
            if var in df.columns:
                fixed_cols.append(var)
                X_fixed_list.append(df[var].values.astype("float32").reshape(-1, 1))

    # # categorical predictors
    # cat_col_slices = {}
    # for cat in categorical_vars:
    #     cat_obj = pd.Categorical(df[cat], categories=cat_code_maps[cat].categories)
    #     levels = list(cat_obj.categories)
    #     cat_codes = pd.get_dummies(cat_obj, prefix=cat, drop_first=False)
    #     ref_level = levels[-1]
    #     ref_col = f"{cat}_{ref_level}"
    #     # keep only non-reference columns
    #     nonref_cols = [c for c in cat_codes.columns if c != ref_col]
    #     nonref_levels = [lvl for lvl in levels if lvl != ref_level]
    #     cat_codes_nonref = cat_codes[nonref_cols]
    #     fixed_cols.extend([f"{cat}[{lvl}]" for lvl in nonref_levels])
    #     X_fixed_list.append(cat_codes_nonref.values.astype("float32"))
    #     cat_col_slices[cat] = (len(fixed_cols) - cat_codes_nonref.shape[1], len(fixed_cols))

    # prepare data arrays
    X_fixed = np.concatenate(X_fixed_list, axis=1).astype("float32")
    y_obs = df["items_log"].values.astype("float32")
    practice_idx = pd.Categorical(df["practice_id"]).codes.astype("int32")
    n_practice = int(df["practice_id"].nunique())
    date_code = df["date_code"].values.astype("float32")
    month_idx = (df["month"].values - 1).astype("int32")  # 0–11

    # ensure all x_fixed vars are standardised
    X_means = X_fixed.mean(axis=0)
    X_stds = X_fixed.std(axis=0)
    X_stds[X_stds == 0] = 1.0
    flood_idx = [i for i, name in enumerate(fixed_cols)
                 if name == "flood" or name.startswith("flood_lag")]
    X_stds[flood_idx] = 1.0  # do not scale flood variable (or lags) as binary fields
    X_fixed = (X_fixed - X_means) / X_stds

    # prepare practice covariates
    region_idx = pd.Categorical(df["region"], categories=REGION_NAMES).codes
    size_idx = pd.Categorical(df["practice_size"], categories=PRACTICE_SIZES).codes
    n_region = len(REGION_NAMES)
    n_size = len(PRACTICE_SIZES)
    n_obs = len(df)

    # random effect design matrices
    # use sparse matrix for practice random effects (many practices)
    Z_practice = sp.csr_matrix(
        (np.ones(n_obs, dtype="float32"), (np.arange(n_obs), practice_idx)),
        shape=(n_obs, n_practice)
    ).astype("float32")
    # slope RE only for practices with sufficient observations
    sufficient_obs = df.groupby("practice_id").size() > min_obs_for_slope
    slope_practices = sufficient_obs[sufficient_obs].index  # only these get slope RE
    slope_rows = df["practice_id"].isin(slope_practices)
    Z_practice_slope = sp.csr_matrix(
        (np.ones(slope_rows.sum(), dtype="float32"),
        (np.where(slope_rows)[0],
        pd.Categorical(df.loc[slope_rows, "practice_id"], categories=slope_practices).codes)),
        shape=(len(df), len(slope_practices))
    )
    Z_practice_slope = Z_practice_slope.multiply(date_code[:, None]).tocsr().astype("float32")
    # use dense matrices for region and size random effects (few levels)
    Z_region_data = np.eye(n_region, dtype="float32")[region_idx]
    Z_size_data   = np.eye(n_size,   dtype="float32")[size_idx]

    # categorise fixed effect columns
    env_idx   = [i for i, n in enumerate(fixed_cols) if classify_beta(n) == "environmental"]
    flood_idx = [i for i, n in enumerate(fixed_cols) if classify_beta(n) == "flood"]
    # month_idx = [i for i, n in enumerate(fixed_cols) if classify_beta(n) == "month"]

    # create and run the model ====================================================================
    status("Running model...", level=1)
    with pm.Model() as model:
        # register data
        X_fixed_data = pm.Data("X_fixed", X_fixed)
        y_obs_data = pm.Data("y_obs", y_obs)
        Z_practice_data = pts.as_sparse_variable(Z_practice)
        Z_practice_slope_data = pts.as_sparse_variable(Z_practice_slope)
        Z_region_data = pm.Data("Z_region", Z_region_data)
        Z_size_data = pm.Data("Z_size", Z_size_data)
        size_idx_data = pm.Data("size_idx", size_idx)
        month_idx_data = pm.Data("month_idx", month_idx)

        # practice-size-specific intercepts
        mu_alpha = pm.Normal("mu_alpha", 6.0, 2.0)  # mid-point between small and large practices in log-space
        sigma_alpha = pm.HalfNormal("sigma_alpha", 2.0)  # intercept variability between practice sizes in log-space
        alpha_size = pm.Normal(
            "alpha_size",
            mu=mu_alpha,
            sigma=sigma_alpha,
            shape=n_size
        )
        size_intercept = alpha_size[size_idx_data]

        # random effects
        if practice_correction == 0:
            practice_intercept = 0.0
            practice_slope_term = 0.0
        else:
            # region-level intercepts
            sigma_region = pm.HalfNormal("sigma_region", 0.5)
            region_raw = pm.Normal("region_raw", 0.0, 1.0, shape=n_region)
            region_mean = pm.Deterministic(
                "region_mean", (region_raw - region_raw.mean()) * sigma_region
            )
            # practice-level deviations (non-centred)
            sigma_practice = pm.HalfNormal("sigma_practice", 1.0)
            practice_offset = pm.Normal("practice_offset", 0.0, 1.0, shape=n_practice)
            # intercept
            practice_intercept = (
                pm.math.dot(
                    Z_practice_data,
                    (practice_offset * sigma_practice)[:, None]
                ).squeeze() +
                pm.math.dot(Z_region_data, region_mean)
            )
            # slope (only for practice_correction == 2)
            if practice_correction == 2:
                sigma_slope = pm.HalfNormal("sigma_slope", 0.25)
                slope_re_offset = pm.Normal("slope_re_offset", 0.0, 1.0,
                                            shape=len(slope_practices))
                slope_re = pm.Deterministic("slope_re", slope_re_offset * sigma_slope)
                practice_slope_term = pm.math.dot(Z_practice_slope_data,
                                                  slope_re[:, None]).squeeze()
            else:
                practice_slope_term = 0.0

        # seasonal (month) effects: sum-to-zero for deseasonalisation
        if deseasonalise_output:
            sigma_month = pm.HalfNormal("sigma_month", 0.3)
            month_raw = pm.Normal("month_raw", 0.0, 1.0, shape=12)

            month_effect = pm.Deterministic("month_effect",
                                            (month_raw - month_raw.mean()) * sigma_month
            )
            month_term = month_effect[month_idx_data]
        else:
            month_term = 0.0

        # fixed effects
        tau_env = pm.HalfNormal("tau_env", 0.5)
        beta_env = pm.Normal("beta_env", mu=0.0, sigma=tau_env, shape=len(env_idx))
        beta_flood = pm.Normal("beta_flood", mu=0.0, sigma=1.0, shape=len(flood_idx))
        # beta_month = pm.Normal("beta_month", mu=0.0, sigma=0.5, shape=len(month_idx))

        # assemble full beta vector
        beta_list = []
        for i in range(X_fixed.shape[1]):
            if i in env_idx:
                beta_list.append(beta_env[env_idx.index(i)])
            elif i in flood_idx:
                beta_list.append(beta_flood[flood_idx.index(i)])
            # elif i in month_idx:
            #     beta_list.append(beta_month[month_idx.index(i)])
            else:
                raise ValueError(f"Variable {fixed_cols[i]} not categorised for beta assembly")
        beta = pm.Deterministic("beta", pm.math.stack(beta_list, axis=0))

        # assemble full beta vector in original column order
        eta = size_intercept + \
              practice_intercept + \
              practice_slope_term + \
              month_term + \
              pm.math.dot(X_fixed_data, beta)

        # residual
        if likelihood == "normal":
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal("items_obs", mu=eta, sigma=sigma, observed=y_obs_data)
        elif likelihood == "studentt":
            sigma = pm.HalfNormal("sigma", 1.0)
            pm.StudentT("items_obs", nu=4, mu=eta, sigma=sigma, observed=y_obs_data)
        else:
            raise ValueError("likelihood must be 'normal' or 'studentt'")

        # sampling
        if use_gpu:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                target_accept=0.95,
                max_tree_depth=15,
                nuts_sampler="numpyro",
                nuts_sampler_kwargs={"chain_method": "vectorized"},
            )
        else:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=0.95,
                max_tree_depth=15,
            )

    # save inference data =========================================================================
    # drop unnecessary groups
    drop_groups = ["log_likelihood", "prior", "prior_predictive",
                   "constant_data", "sample_stats_prior"]
    for group in drop_groups:
        if group in idata.groups():
            del idata[group]
    
    # recombined beta objects
    drop_beta_components = [
        "beta_env",
        "beta_flood",
        # "beta_month",
    ]
    existing = [v for v in drop_beta_components if v in idata.posterior.data_vars]
    idata.posterior = idata.posterior.drop_vars(existing)

    # restructure inference data for named predictors
    beta_da = idata.posterior["beta"]
    named_betas = {
        name: beta_da.isel(beta_dim_0=i).drop_vars("beta_dim_0")
        for i, name in enumerate(fixed_cols)
    }
    idata.posterior = idata.posterior.drop_vars("beta")
    idata.posterior = xr.merge([idata.posterior, xr.Dataset(named_betas)])

    # replace categorical codes with names
    def map_hierarchical_vars(idata):
        hierarchical_mappings = {
            "region_mean": REGION_NAMES,
            "region_raw": REGION_NAMES,
            "alpha_size": PRACTICE_SIZES,
            "month_effect": MONTHS,
        }
        new_vars = {}
        vars_to_drop = []
        for var, names in hierarchical_mappings.items():
            if var not in idata.posterior:
                continue
            da = idata.posterior[var]
            dim = da.dims[-1]
            for i, name in enumerate(names):
                new_name = f"{var}[{name}]"
                new_vars[new_name] = da.isel({dim: i}).drop_vars(dim)
            vars_to_drop.append(var)
        if new_vars:
            idata.posterior = idata.posterior.drop_vars(vars_to_drop)
            idata.posterior = xr.merge([idata.posterior, xr.Dataset(new_vars)])
        return idata
    idata = map_hierarchical_vars(idata)

    # compute and save lag effects
    if lag > 0:
        lag_effect_vars = {}
        for var in lag_vars:
            basis_names = [
                name for name in fixed_cols
                if name.startswith(f"{var}_almon_basis_")
            ]
            if not basis_names:
                continue
            for l in range(lag + 1):
                weights = almon_X[l, :len(basis_names)]
                effect = sum(
                    idata.posterior[name] * w
                    for name, w in zip(basis_names, weights)
                )
                lag_effect_vars[f"{var}_lag_effect[{l}]"] = effect
        if lag_effect_vars:
            idata.posterior = xr.merge(
                [idata.posterior, xr.Dataset(lag_effect_vars)]
            )

    # save posterior summary with categorical names
    summary_df = az.summary(idata, hdi_prob=0.95)
    summary_df["mean_pct"] = 100 * (np.exp(summary_df["mean"]) - 1)
    summary_df["hdi_2.5pc_pct"] = 100 * (np.exp(summary_df["hdi_2.5%"]) - 1)
    summary_df["hdi_97.5pc_pct"] = 100 * (np.exp(summary_df["hdi_97.5%"]) - 1)
    summary_df = summary_df.reset_index().rename(columns={"index": "parameter"})
    summary_csv = os.path.join(results_folder, "bayesian_model_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    status(f"Posterior summary saved to: {summary_csv}", level=1)

    # save inference data
    idata_path = os.path.join(results_folder, "bayesian_model_idata.nc")
    az.to_netcdf(idata, idata_path)
    status(f"Model results saved to: {idata_path}", level=1)

    # posterior predictive sampling
    status("Generating posterior predictive samples for PPC...", level=1)
    idata_pp = pm.sample_posterior_predictive(idata, model=model, var_names=["items_obs"])
    idata.extend(idata_pp)

    # generate diagnostics
    status("Generating Bayesian diagnostic plots...", level=1)
    generate_bayesian_diagnostics(idata, results_folder)

    return model, idata

# MODELING HELPERS --------------------------------------------------------------------------------
def prepare_ds(
        ds,
        n_practices=None,
        standardise_items=False,
        clean_items=False,
        adjust_predictors=None,
        deseasonalise_predictors=False,
        practice_mean_thresh=500,
    ):
    """
    Prepare dataset for analysis by adding date_code, limiting practices and
    standardising variables/items, as requested.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with 'date' and 'practice_id' dimensions.
    n_practices : int or None
        If specified, limit to this many practices with most items.
    standardise_items : bool
        Whether to standardise 'items' per practice.
    clean_items : bool
        Whether to clean 'items' by removing low values and practices with low means.
    adjust_predictors : str or None
        Method to adjust predictor inputs before modelling. Options include:
        'z-global': standardise values globally
        'z-practice': standardise per practice
        'c-global': centre globally
        'c-practice': centre per practice
        None: raw values
    deseasonalise_predictors : bool
        Whether to do seasonal correction on predictors.
    """
    # add date code as simpler data variable for models
    ds["date_code"] = ("date", np.arange(len(ds["date"])))  # integer months since start
    ds["date_code"] = (ds["date_code"] - ds["date_code"].mean()) / ds["date_code"].std()  # standardised

    # add practice size (creates two groups, large and small, with roughly normal distributions)
    practice_means = ds["items"].mean(dim="date")
    ds["practice_size"] = ("practice_id", np.where(practice_means >= practice_mean_thresh,
                                                   "large", "small"))
    
    # exclude random single welsh practice
    ds = ds.isel(practice_id=ds["region"] != "Wales")

    # limit practices if requested
    if n_practices is not None:
        status(f"Limiting to {n_practices} randomly selected practices", level=1)
        selected = np.random.choice(ds["practice_id"].values,
                                    size=n_practices, replace=False)
        ds = ds.sel(practice_id=selected)
    
    # adjust predictor variables if requested
    if adjust_predictors is not None:
        # per practice centering/standardising
        if adjust_predictors in ["c-practice", "z-practice"]:
            status("Adjusting predictor values per practice", level=1)
            for var in ds.data_vars:
                if var.endswith("_values"):
                    mean = ds[var].mean(dim="date")
                    ds[var] = ds[var] - mean
                    if adjust_predictors == "z-practice":
                        std = ds[var].std(dim="date") + 1e-8  # prevent div by zero
                        ds[var] = ds[var] / std
        # global centering/standardising
        elif adjust_predictors in ["c-global", "z-global"]:
            status("Adjusting predictor values globally", level=1)
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
        status("Applying seasonal correction to predictor variables", level=1)
        for var in ds.data_vars:
            if var.endswith("_values"):
                monthly_means = ds[var].groupby("date.month").mean(dim="date")
                monthly_means_expanded = monthly_means.sel(month=ds["date.month"])
                ds[var] = ds[var] - monthly_means_expanded
    
    # clean prescription items if requested
    if clean_items:
        n_prac_before = len(ds["practice_id"])
        ds = clean_prescription_items(ds)
        n_prac_after = len(ds["practice_id"])
        status(f"Cleaning prescription 'items': {n_prac_before} -> {n_prac_after} practices", level=1)

    # standardise prescription 'items' per practice if requested
    if standardise_items:
        status("Standardising 'items' per practice", level=1)
        mean = ds["items"].mean(dim="date")
        std = ds["items"].std(dim="date") + 1e-8  # prevent div by zero
        ds["items"] = (ds["items"] - mean) / std

    return ds

def clean_prescription_items(ds, mean_thresh=10, mean_fraction=0.1, min_val=1):
    # drop practices with low means
    practice_means = ds['items'].mean(dim='date')
    mean_condition = (practice_means > mean_thresh)
    ds = ds.isel(practice_id=mean_condition)

    # drop (set nan) values below max(mean_fraction of mean, min_val)
    practice_means = ds['items'].mean(dim='date')
    cutoff = np.maximum(practice_means * mean_fraction, min_val)
    cutoff = cutoff.broadcast_like(ds['items'])
    ds['items'] = ds['items'].where(ds['items'] > cutoff, other=np.nan)

    return ds

def drop_collinear_cols(df_design, tol=None):
    """
    returns (kept_cols, dropped_cols)
    df_design : DataFrame (n_obs x n_cols)
    tol : float | None - threshold on singular values; if None compute default tol
    """
    X = np.asarray(df_design, dtype=float)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if tol is None:
        eps = np.finfo(float).eps
        tol = max(X.shape) * s.max() * eps
    # identify tiny singular values
    drop_svals = np.where(s <= tol)[0]
    if drop_svals.size == 0:
        return list(df_design.columns), []
    # find dependent column indices from Vt rows corresponding to tiny s
    dependent_cols = set()
    for idx in drop_svals:
        v = Vt[idx, :]
        # pick the largest magnitude coefficient(s) as indicative columns
        abs_v = np.abs(v)
        # mark columns with small contribution as candidates; we choose threshold relative to max
        rel = abs_v / abs_v.max()
        # pick columns with relative weight > 0.2 (heuristic) as candidates to remove
        candidate_idx = np.where(rel > 0.2)[0]
        for ci in candidate_idx:
            dependent_cols.add(df_design.columns[ci])
    kept = [c for c in df_design.columns if c not in dependent_cols]
    dropped = sorted(list(dependent_cols))
    return kept, dropped

def generate_mixed_effects_diagnostics(mdf, df_model, model_folder, model_name):
    """Generate residual, QQ, histogram and random effects plots without saving the model."""
    fitted = mdf.fittedvalues
    resid = df_model.loc[fitted.index, "items_log"] - fitted

    # residual vs fitted
    plt.figure(figsize=(6,4))
    plt.scatter(fitted, resid, alpha=0.3)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"Residual vs Fitted: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "resid_vs_fitted.png"))
    plt.close()

    # QQ plot
    plt.figure(figsize=(6,4))
    sm.qqplot(resid, line="45", fit=True)
    plt.title(f"Residual QQ: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "resid_qq.png"))
    plt.close()

    # histogram
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=100, edgecolor="black")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual Histogram: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "resid_hist.png"))
    plt.close()

    # random effects
    re = mdf.random_effects
    re_df = pd.DataFrame(re).T

    for col in re_df.columns:
        plt.figure(figsize=(6,4))
        plt.hist(re_df[col], bins=100, edgecolor="black")
        plt.title(f"Random effect {col}: {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, f"ranef_{col}_hist.png"))
        plt.close()

        plt.figure(figsize=(6,4))
        sm.qqplot(re_df[col], line="45", fit=True)
        plt.title(f"Random effect QQ {col}: {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, f"ranef_{col}_qq.png"))
        plt.close()
    
    plt.close(fig="all")

def generate_bayesian_diagnostics(idata, results_folder, max_subplots=40):
    """
    Generate Bayesian diagnostic plots (trace, posterior, rank, autocorr, PPC),
    automatically splitting figures so no figure has more than `max_subplots` subplots.
    Random effects are excluded.
    """
    az.rcParams["plot.max_subplots"] = max_subplots
    auto_corr_folder = os.path.join(results_folder, "autocorr")
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(auto_corr_folder, exist_ok=True)

    # combine intercept and slope RE if available
    re_vars = []
    if "practice_offset" in idata.posterior.data_vars:
        re_vars.append("practice_offset")
    if "slope_re" in idata.posterior.data_vars:
        re_vars.append("slope_re")
        re_vars.append("slope_re_offset")

    # compute rhat for all random effects
    worst_re_coords = {}
    if re_vars:
        rhat = az.rhat(idata, var_names=re_vars)
        for var in re_vars:
            rhat_var = rhat[var]
            # assume last dimension corresponds to "practice"
            practice_dim = rhat_var.dims[-1]
            # mean over all other dimensions
            practice_mean_rhat = rhat_var.mean(dim=[d for d in rhat_var.dims if d != practice_dim])
            worst_idx = np.argsort(practice_mean_rhat.values)[-10:]  # worst 10
            worst_re_coords[var] = (practice_dim, worst_idx)

    # select posterior variables including only 10 worst rhat random effects
    exclude = {  # exclude these as diagnostics uninformative
        "region_raw",
        "size_raw",
        "practice_offset",
        "slope_re_offset",
    }
    posterior_vars = [var for var in idata.posterior.data_vars
                      if var not in re_vars
                      and var.split("[")[0] not in exclude]

    # convergence (trace) plots
    for i, start in enumerate(range(0, len(posterior_vars), max_subplots)):
        end = min(start + max_subplots//2, len(posterior_vars))
        vars_subset = posterior_vars[start:end]
        fig = az.plot_trace(idata, var_names=vars_subset)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"convergence_{i + 1}.png"), bbox_inches='tight')
        plt.close(fig='all')

    # convergence plots for worst random effects
    for var, (dim_name, idx) in worst_re_coords.items():
        fig = az.plot_trace(
            idata,
            var_names=[var],
            coords={dim_name: idx},
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_folder, f"convergence_{var}_worst.png"),
            bbox_inches="tight",
        )
        plt.close(fig="all")

    def count_subplots(var):
        # estimate number of subplots needed for variable
        shape = idata.posterior[var].shape
        # exclude chain and draw dimensions (assume dims 0 and 1)
        return int(np.prod(shape[2:])) if len(shape) > 2 else 1

    # posterior and rank plots
    for plot_type, az_func, kwargs in [
        ("posterior", az.plot_posterior, {"hdi_prob": 0.95, "point_estimate": "mean", "kind": "hist"}),
        ("rank", az.plot_rank, {}),
    ]:
        # chunk variables by estimated subplots
        chunks = []
        current_chunk, current_count = [], 0
        for var in posterior_vars:
            n_subplots = count_subplots(var)
            if current_count + n_subplots > max_subplots and current_chunk:
                chunks.append(current_chunk)
                current_chunk, current_count = [], 0
            current_chunk.append(var)
            current_count += n_subplots
        if current_chunk:
            chunks.append(current_chunk)

        # generate plots
        for i, vars_subset in enumerate(chunks):
            fig = az_func(idata, var_names=vars_subset, **kwargs)
            plt.tight_layout()
            plt.savefig(os.path.join(results_folder, f"{plot_type}_{i + 1}.png"), bbox_inches='tight')
            plt.close(fig='all')

    # autocorrelation plots (one per var as incalculable subplots)
    for var in posterior_vars:
        fig = az.plot_autocorr(idata, var_names=[var])
        plt.tight_layout()
        safe_var_name = var.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(auto_corr_folder, f"autocorr_{safe_var_name}.png"), bbox_inches='tight')
        plt.close(fig='all')

    # energy plot
    fig = az.plot_energy(idata)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "energy.png"), bbox_inches='tight')
    plt.close(fig='all')

    # posterior predictive checks
    if "posterior_predictive" in idata.groups():
        fig = az.plot_ppc(idata, num_pp_samples=50)
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, "ppc.png"), bbox_inches='tight')
        plt.close(fig='all')
    else:
        status(f"Warning: no posterior predictive data found in idata, skipping PPC.", level=2)
    status(f"All Bayesian diagnostic plots saved to {results_folder}", level=2)

def classify_beta(name: str) -> str:
    name = name.lower()
    if "flood" in name:
        return "flood"
    if "_almon_basis_" in name:
        parent = name.split("_almon_basis_")[0]
        return classify_beta(parent)
    if name.startswith("month["):
        return "month"
    return "environmental"



# ANALYSIS FUNCTIONS ==============================================================================
def compare_mixed_models(results_folder, save_folder, legend_y_offset_px=42):
    os.makedirs(save_folder, exist_ok=True)

    # load results for each prescription type
    all_results = {}
    for code in PRES_CODES:
        csv_path = os.path.join(results_folder, code, "mixed_effects_results.csv")
        if not os.path.exists(csv_path):
            status(f"Warning: missing results for {code}, skipping.", level=1)
            continue
        df = pd.read_csv(csv_path)
        df["pres_code"] = code
        all_results[code] = df

    if not all_results:
        raise ValueError("No results found in any PRES_CODE subfolder.")

    # flag predictor effects
    flag_dfs = []
    for code, df in all_results.items():
        tmp = df[df["type"].isin(["binary_simple",
                                  "binary_pair",
                                  "daqi_pair1",
                                  "daqi_pair2"])].copy()
        tmp["coef"] = tmp["pred_coef"]
        tmp["ci_low"] = tmp["pred_ci_low"]
        tmp["ci_high"] = tmp["pred_ci_high"]
        tmp["name"] = tmp["name"]  # already the predictor name
        flag_dfs.append(tmp)
    fig_flag, ax_flag = plot_combined(flag_dfs, PRES_LABELS, PRES_COLOURS,
                                      x_label="Predictor effect (%)")
    add_legend(fig_flag, ax_flag, labels=PRES_LABELS, legend_y_offset_px=legend_y_offset_px)
    fig_flag.savefig(os.path.join(save_folder, "mixed_flags.png"), dpi=600, bbox_inches='tight')
    plt.close(fig_flag)

    # flag predictor effects (no sulphur dioxide)
    flag_dfs = []
    for code, df in all_results.items():
        tmp = df[df["type"].isin(["binary_simple",
                                  "binary_pair",
                                  "daqi_pair1",
                                  "daqi_pair2"])].copy()
        tmp["coef"] = tmp["pred_coef"]
        tmp["ci_low"] = tmp["pred_ci_low"]
        tmp["ci_high"] = tmp["pred_ci_high"]
        tmp["name"] = tmp["name"]  # already the predictor name
        flag_dfs.append(tmp)
    fig_flag, ax_flag = plot_combined(flag_dfs, PRES_LABELS, PRES_COLOURS,
                                      x_label="Predictor effect (%)",
                                      hide_sulphur=True)
    add_legend(fig_flag, ax_flag, labels=PRES_LABELS, legend_y_offset_px=legend_y_offset_px)
    fig_flag.savefig(os.path.join(save_folder, "mixed_flags_nosulphur.png"), dpi=600, bbox_inches='tight')
    plt.close(fig_flag)

    # continuous predictor effects (no sulphur dioxide)
    flag_dfs = []
    for code, df in all_results.items():
        tmp = df[df["type"].isin(["continuous"])].copy()
        tmp["coef"] = tmp["pred_coef"]
        tmp["ci_low"] = tmp["pred_ci_low"]
        tmp["ci_high"] = tmp["pred_ci_high"]
        tmp["name"] = tmp["name"]  # already the predictor name
        flag_dfs.append(tmp)
    fig_flag, ax_flag = plot_combined(flag_dfs, PRES_LABELS, PRES_COLOURS,
                                      x_label="Predictor effect (%)")
    add_legend(fig_flag, ax_flag, labels=PRES_LABELS, legend_y_offset_px=legend_y_offset_px)
    fig_flag.savefig(os.path.join(save_folder, "mixed_continuous.png"), dpi=600, bbox_inches='tight')
    plt.close(fig_flag)

    # deprivation effects
    imd_dfs = []
    for code, df in all_results.items():
        tmp = df.copy()
        tmp["coef"] = tmp["imd_coef"]
        tmp["ci_low"] = tmp["imd_ci_low"]
        tmp["ci_high"] = tmp["imd_ci_high"]
        tmp["name"] = tmp["name"]  # predictor name
        imd_dfs.append(tmp)
    fig_imd, ax_imd = plot_combined(imd_dfs, PRES_LABELS, PRES_COLOURS,
                                    x_label="IMD effect (%)")
    add_legend(fig_imd, ax_imd, labels=PRES_LABELS, legend_y_offset_px=legend_y_offset_px)
    fig_imd.savefig(os.path.join(save_folder, "mixed_imd.png"), dpi=600, bbox_inches='tight')
    plt.close(fig_imd)

    # region effects
    region_dfs = []
    for code, df in all_results.items():
        for _, row in df.iterrows():
            region_dict = ast.literal_eval(row["region_main"])
            for region, val in region_dict.items():
                region_dfs.append({
                    "name": region,
                    "coef": val["coef"],
                    "ci_low": val["ci_low"],
                    "ci_high": val["ci_high"],
                    "pres_code": code
                })
    # find the reference region
    unique_regions = set(r["name"] for r in region_dfs)
    ref_region = unique_regions.symmetric_difference(set(REGION_NAMES))
    ref_region = next(iter(ref_region), "")
    # plot
    region_dfs = pd.DataFrame(region_dfs)
    region_df_list = [region_dfs[region_dfs["pres_code"] == code] for code in PRES_CODES]
    fig_region, ax_region = plot_combined(region_df_list, PRES_LABELS, PRES_COLOURS,
                                          x_label=f"Effect relative to {ref_region} (%)",
                                          region_plot=True, order=REGION_NAMES)
    add_legend(fig_region, ax_region, labels=PRES_LABELS, legend_y_offset_px=legend_y_offset_px)
    fig_region.savefig(os.path.join(save_folder, "mixed_region.png"), dpi=600, bbox_inches='tight')
    plt.close(fig_region)

def compare_bayesian_models(results_root):
    results_folders = [os.path.join(results_root, c) for c in PRES_CODES]
    plot_root = results_root

    numeric_dataframes, lagged_dataframes, lagged_data_inds, numeric_data_inds = [], [], [], []

    for i, folder in enumerate(results_folders):
        summary_path = os.path.join(folder, "bayesian_model_summary.csv")
        if not os.path.exists(summary_path):
            status(f"Warning: results not found at {summary_path}, skipping...", level=1)
            continue

        df = pd.read_csv(summary_path).rename(columns={"parameter": "name"})

        # exclude random effects
        exclude_pattern = (
            r"^Intercept$|^sigma(?:$|_)|"
            r"^sigma_|^region_|^size_|^practice_|"
            r"_raw$|_offset$|"
            r"^slope_re$|^slope_re_offset$|"
            r"^alpha_size(?:\[.*\])?$|"
            r"^mu_alpha$|^tau_env$"
        )
        df = df[~df["name"].str.contains(exclude_pattern, regex=True)]

        # identify lagged variables
        lagged_rows = df[df["name"].str.contains(r"_lag_effect\[\d+\]")].copy()
        if not lagged_rows.empty:
            lagged_rows["variable"] = lagged_rows["name"].str.replace(
                r"_lag_effect\[\d+\]", "", regex=True
            )
            lagged_rows["lag"] = (lagged_rows["name"].str.extract(r"\[(\d+)\]").astype(int))
            lagged_dataframes.append(lagged_rows)
            lagged_data_inds.append(i)

        # numeric predictors (exclude intercept, sigma, month, region)
        numeric_rows = df[~df["name"].str.contains(r"_lag_effect|^month\[|^region_|^size_",
                                                   regex=True)]
        if not numeric_rows.empty:
            numeric_dataframes.append(numeric_rows)
            numeric_data_inds.append(i)

    if not numeric_dataframes and not lagged_dataframes:
        raise FileNotFoundError("No Bayesian results found — nothing to plot.")

    os.makedirs(plot_root, exist_ok=True)

    # lagged predictor plots
    if lagged_dataframes:
        status("Generating combined lagged predictor plots...", level=1)
        lags = sorted({l for df in lagged_dataframes for l in df["lag"].unique()})
        for lag in lags:
            df_list = [extract_lag_values(df, lag) for df in lagged_dataframes]
            fig, ax = plot_combined(
                df_list=df_list,
                labels_list=[PRES_LABELS[i] for i in lagged_data_inds],
                colours_list=[PRES_COLOURS[i] for i in lagged_data_inds],
                col_est="mean_pct",
                col_low="hdi_2.5pc_pct",
                col_high="hdi_97.5pc_pct",
                x_label=f"Effect estimate for {lag} months lag (%)",
            )
            fig.tight_layout()
            fig.savefig(os.path.join(plot_root, f"compare_predictors_lag_{lag}.png"), dpi=600, bbox_inches='tight')
            plt.close(fig)

        status("Generating per-variable Almon lag plots...", level=1)
        lag_plot_folder = os.path.join(plot_root, "predictors")
        os.makedirs(lag_plot_folder, exist_ok=True)
        variables = sorted({v for df in lagged_dataframes for v in df["variable"].unique()})
        for var in variables:
            plot_lagged_variable(
                var=var,
                dataframes=lagged_dataframes,
                labels=[PRES_LABELS[i] for i in lagged_data_inds],
                colours=[PRES_COLOURS[i] for i in lagged_data_inds],
                outpath=os.path.join(lag_plot_folder, f"{var}.png"),
                col_est="mean_pct",
                col_low="hdi_2.5pc_pct",
                col_high="hdi_97.5pc_pct",
            )
        status(f"Per-variable lag plots saved to {lag_plot_folder}", level=1)

    # combined numeric predictor plot
    elif numeric_dataframes:
        status("Generating combined predictor plot...", level=1)
        fig, ax = plot_combined(
            df_list=numeric_dataframes,
            labels_list=[PRES_LABELS[i] for i in numeric_data_inds],
            colours_list=[PRES_COLOURS[i] for i in numeric_data_inds],
            col_est="mean_pct",
            col_low="hdi_2.5pc_pct",
            col_high="hdi_97.5pc_pct",
            x_label="Effect estimate (%)"
        )
        fig.tight_layout()
        add_legend(fig, ax, labels=PRES_LABELS)
        fig.savefig(os.path.join(plot_root, "compare_predictors.png"), dpi=600, bbox_inches='tight')
        plt.close(fig)
        status(f"Combined predictor plot saved to {plot_root}compare_predictors.png", level=1)

        # per-variable numeric plots
        status("Generating per-variable predictor plots...", level=1)
        plot_folder = os.path.join(plot_root, "predictors")
        os.makedirs(plot_folder, exist_ok=True)

        def bayes_extractor(df, var):
            return df[df["name"] == var]

        all_vars = sorted({v for df in numeric_dataframes for v in df["name"].unique()})
        for var in all_vars:
            plot_variable(
                var=var,
                dataframes=numeric_dataframes,
                labels=[PRES_LABELS[i] for i in numeric_data_inds],
                colours=[PRES_COLOURS[i] for i in numeric_data_inds],
                var_extractor=bayes_extractor,
                outpath=os.path.join(plot_folder, f"{var}.png"),
                col_est="mean_pct",
                col_low="hdi_2.5pc_pct",
                col_high="hdi_97.5pc_pct",
            )
        status(f"Per-variable predictor plots saved to {plot_folder}", level=1)

    # regional effects
    status("Generating regional effects plot...", level=1)
    plot_regions_bayes(results_folders, plot_root)

    # seasonal effects
    status("Generating month effects plot...", level=1)
    plot_months_bayes(results_folders, plot_root)

# ANALYSIS HELPERS --------------------------------------------------------------------------------
def plot_combined(
    df_list: list,
    labels_list: list,
    colours_list: list,
    ax=None,
    x_label: str = "Effect estimate",
    xlim: tuple = (None, None),
    y_jitter: float = 0.15,
    hide_sulphur: bool = False,
    region_plot: bool = False,
    order: list = None,
    col_est: str = "coef",
    col_low: str = "ci_low",
    col_high: str = "ci_high",
):
    """
    Plot error-bar estimates for multiple prescription types.
    If ax is None, creates a new figure and axes; otherwise plots on the supplied axes.
    """
    if not df_list:
        status("No data to plot.", level=1)
        return None, None

    # create figure if needed
    if ax is None:
        if region_plot:
            height = 1 + len(np.unique(df_list[0]["name"]))*5/18
        else:
            height = 1 + len(df_list[0])*5/18
        fig, ax = plt.subplots(figsize=(10, height))
    else:
        fig = ax.figure

    # identify variables that have non-NaN estimates
    all_names = []
    for df in df_list:
        df_temp = df.copy()
        if hide_sulphur:
            df_temp = df_temp[~df_temp["name"].str.contains("aqrean_daqi_sulfur_dioxide")]
        all_names.extend(df_temp["name"].unique())
    all_names = sorted(set(all_names))
    valid_names = []
    for name in all_names:
        if any(df.loc[df["name"] == name, col_est].notna().any() for df in df_list):
            valid_names.append(name)

    # sort variables names alphabetically or by supplied order
    if order is None:
        final_varnames = sorted((var_name_to_plot_name(n) for n in valid_names), reverse=True)
    else:
        ordered_names = np.flip([n for n in order if n in valid_names])
        final_varnames = [var_name_to_plot_name(n) for n in ordered_names]
    name_to_idx = {name: j for j, name in enumerate(final_varnames)}

    # plot each prescription type
    for i, (df, label, colour) in enumerate(zip(df_list, labels_list, colours_list)):
        df_plot = df.copy()
        if hide_sulphur:
            df_plot = df_plot[~df_plot["name"].str.contains("aqrean_daqi_sulfur_dioxide")]
        if df_plot.empty:
            continue

        varnames = df_plot["name"].apply(var_name_to_plot_name)
        df_plot["plot_name"] = varnames
        df_plot = df_plot[df_plot["plot_name"].isin(final_varnames)].copy()
        df_plot["idx"] = df_plot["plot_name"].map(name_to_idx)
        df_plot = df_plot.sort_values("idx")

        y_positions = df_plot["idx"].values + (i - (len(df_list) - 1)/2) * y_jitter
        ax.errorbar(
            df_plot[col_est], y_positions,
            xerr=[df_plot[col_est] - df_plot[col_low], df_plot[col_high] - df_plot[col_est]],
            fmt='o', color=colour, label=label, markersize=4, capsize=3
        )

    ax.axvline(0, color="black", alpha=0.8, label="_nolegend_")
    ax.grid(alpha=0.8)
    ax.set_yticks(np.arange(len(final_varnames)))
    ax.set_yticklabels(final_varnames)
    ax.set_xlim(xlim)
    ax.set_xlabel(x_label)

    return fig, ax

def plot_variable(
        var,
        dataframes, labels, colours,
        var_extractor,
        outpath,
        col_est="coef",
        col_low="ci_low",
        col_high="ci_high",
        figsize=(6, 4)
):
    """
    Generic helper for making per-variable coefficient/mean plots.
    Works with frequentist and Bayesian summaries.

    Parameters
    ----------
    var : str
        Variable name to plot.
    dataframes : list
        List of dataframes containing summary statistics.
    labels, colours : lists
        Matching labels/colours for each dataframe.
    var_extractor : callable
        Function(df, var) -> subset dataframe containing rows for this variable.
    outpath : str
        Location to save plot.
    col_est : str
        Column name for the estimate (coef/mean).
    col_low, col_high : str
        Column names for the lower and upper interval bounds.
    """
    fig, ax = plt.subplots(figsize=figsize)
    matched = False

    for df, label, colour in zip(dataframes, labels, colours):
        sub = var_extractor(df, var)
        if not sub.empty:
            ax.errorbar(
                sub[col_est], label,
                xerr=[sub[col_est] - sub[col_low], sub[col_high] - sub[col_est]],
                fmt="o", color=colour, markersize=5, capsize=4,
            )
            matched = True

    if not matched:
        plt.close(fig)
        return

    ax.axvline(0, color="black", alpha=0.8, label="_nolegend_")
    ax.grid(alpha=0.8, zorder=-2)
    ax.set_xlabel("Effect estimate")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

def extract_lag_values(df, lag):
    return df[df["lag"] == lag].copy()

def plot_lagged_variable(
        var,
        dataframes, labels, colours,
        outpath,
        col_est="mean_pct",
        col_low="hdi_2.5pc_pct",
        col_high="hdi_97.5pc_pct",
        figsize=(6, 4)
):
    fig, ax = plt.subplots(figsize=figsize)
    matched = False

    for df, label, colour in zip(dataframes, labels, colours):
        sub = df[df["variable"] == var].sort_values("lag")
        if sub.empty:
            continue

        ax.plot(sub["lag"], sub[col_est], "-o", color=colour, label=label)
        ax.fill_between(
            sub["lag"],
            sub[col_low],
            sub[col_high],
            alpha=0.25,
            color=colour
        )
        matched = True

    if not matched:
        plt.close(fig)
        return

    ax.axhline(0, color="black", alpha=0.5)
    ax.grid(alpha=0.7)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("Effect (%)")
    ax.set_title(var)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_prior_distributions(ds):
    values_vars = [var for var in ds.data_vars if var.endswith("_values")]
    for var in values_vars:
        var_data = (ds[var] - ds[var].mean()).values.flatten()
        sd = np.nanstd(var_data)
        dist = np.random.normal(loc=0, scale=2*sd, size=1000)
        data_min = np.nanmin(np.concatenate([var_data, dist]))
        data_max = np.nanmax(np.concatenate([var_data, dist]))
        bins = np.linspace(data_min, data_max, 50)

        plt.figure()
        plt.hist(var_data, bins=bins, alpha=0.5, color='blue', density=True, label='Data')
        plt.hist(dist, bins=bins, alpha=0.5, color='orange', density=True, label='Prior')
        plt.title(f"Histogram of {var} and Prior Distribution")
        plt.xlabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/prior_test/{var}.png")
        plt.close()

def plot_regions_bayes(results_folders, plot_root):
    """
    Plots regional effects across prescription types from Bayesian model summaries.
    Effects are deviations from the national mean (centred hierarchical intercepts).
    """
    region_dfs = []
    labels = []
    colours = []
    for i, folder in enumerate(results_folders):
        # load csv data
        summary_path = os.path.join(folder, "bayesian_model_summary.csv")
        if not os.path.exists(summary_path):
            continue
        df = pd.read_csv(summary_path).rename(columns={"parameter": "name"})

        # extract relative regional effects
        region_rows = df[df["name"].str.startswith("region_mean")]
        if region_rows.empty:
            continue
        region_data = []
        for _, row in region_rows.iterrows():
            region_name = row["name"].split("[")[1].replace("]", "")
            region_data.append({
                "name": region_name,
                "coef": row["mean_pct"],
                "ci_low": row["hdi_2.5pc_pct"],
                "ci_high": row["hdi_97.5pc_pct"],
            })

        # add to dataframe list
        region_dfs.append(pd.DataFrame(region_data))
        labels.append(PRES_LABELS[i])
        colours.append(PRES_COLOURS[i])

    if not region_dfs:
        status("No regional data found.", level=1)
        return

    # plot combined regional effects
    fig_region, ax_region = plot_combined(
        region_dfs,
        labels_list=labels,
        colours_list=colours,
        x_label="Deviation from national mean (%)",
        region_plot=True,
        order=REGION_NAMES,
        col_est="coef",
        col_low="ci_low",
        col_high="ci_high",
    )

    # finalise plot
    fig_region.tight_layout()
    add_legend(fig_region, ax_region, labels=labels, legend_y_offset_px=42)
    os.makedirs(plot_root, exist_ok=True)
    fig_region.savefig(
        os.path.join(plot_root, "compare_regions.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig_region)
    status(f"Bayesian regional effects plot saved to {plot_root}compare_regions.png", level=1)

def plot_months_bayes(results_folders, plot_root):
    """
    Plots month effects across prescription types from Bayesian model summaries.
    Month coefficients are named month[1] ... month[12], with one omitted
    reference month. Effects are relative to that missing month.
    """
    month_dfs = []
    labels = []
    colours = []
    ref_month_idx = None
    for i, folder in enumerate(results_folders):
        # load csv data
        summary_path = os.path.join(folder, "bayesian_model_summary.csv")
        if not os.path.exists(summary_path):
            continue
        df = pd.read_csv(summary_path).rename(columns={"parameter": "name"})

        # extract month coefficients
        month_rows = df[df["name"].str.match(r"^month\[\d+\]$")].copy()
        if month_rows.empty:
            continue
        month_rows["month_idx"] = (
            month_rows["name"]
            .str.extract(r"\[(\d+)\]")
            .astype(int)
        )

        # detect reference month (the missing one)
        present = set(month_rows["month_idx"])
        missing = sorted(set(range(1, 13)) - present)
        if len(missing) != 1:
            raise ValueError(
                f"Could not uniquely determine reference month in {folder}: {missing}"
            )

        # enforce consistency across prescriptions
        if ref_month_idx is None:
            ref_month_idx = missing[0]
        elif ref_month_idx != missing[0]:
            raise ValueError(
                "Reference month differs across prescriptions - models are not comparable."
            )

        # build plotting dataframe
        month_rows["name"] = month_rows["month_idx"].apply(
            lambda m: MONTHS[m - 1]
        )
        month_rows = month_rows.rename(
            columns={
                "mean_pct": "coef",
                "hdi_2.5pc_pct": "ci_low",
                "hdi_97.5pc_pct": "ci_high",
            }
        )

        # add to list
        month_dfs.append(
            month_rows[["name", "coef", "ci_low", "ci_high"]]
        )
        labels.append(PRES_LABELS[i])
        colours.append(PRES_COLOURS[i])

    if not month_dfs:
        status("No month data found.", level=1)
        return
    ref_month_name = MONTHS[ref_month_idx - 1]

    # plot combined month effects
    fig, ax = plot_combined(
        df_list=month_dfs,
        labels_list=labels,
        colours_list=colours,
        x_label=f"Effect relative to {ref_month_name} (%)",
        order=MONTHS,
        col_est="coef",
        col_low="ci_low",
        col_high="ci_high",
    )

    # finalise plot
    fig.tight_layout()
    add_legend(fig, ax, labels=labels, legend_y_offset_px=42)
    os.makedirs(plot_root, exist_ok=True)
    outpath = os.path.join(plot_root, "compare_months.png")
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)
    status(f"Bayesian month effects plot saved to {outpath}", level=1)

def fitted_scatter_plot(x, y, x_label, y_label, title, save_path):
    plt.figure(figsize=(4, 3))
    plt.scatter(x, y, s=1, alpha=0.4)
    non_nan_mask = ~np.isnan(x) & ~np.isnan(y)
    m, b = np.polyfit(x[non_nan_mask], y[non_nan_mask], 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    plt.plot(xx, m*xx + b, 'k--')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def add_legend(fig, axes, labels=None, legend_y_offset_px=42):
    """
    Add a legend to a figure, handling single or multiple axes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the axes.
    axes : matplotlib.axes.Axes or list/array of Axes
        Axes to attach the legend to.
    labels : list of str, optional
        Labels for the legend. If None, uses existing handles/labels.
    legend_y_offset_px : int, optional
        Vertical offset for single-axes legend, in pixels.
    """
    # convert axes to list if a single Axes is passed
    if isinstance(axes, plt.Axes):
        axes = [axes]

    # compute fraction offset based on axes height
    renderer = fig.canvas.get_renderer()
    bbox = axes[0].get_window_extent(renderer=renderer)
    axes_height_px = bbox.height
    y_offset_frac = legend_y_offset_px / axes_height_px

    # if one set of axes, place legend centered below
    if len(axes) == 1:
        ax = axes[0]
        ax.legend(
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -y_offset_frac),
            bbox_transform=ax.transAxes,
            ncol=4,
            frameon=False
        )
    # if odd number of axes, place under center of middle axes
    elif len(axes) % 2 == 1:
        mid_ax = axes[len(axes) // 2]
        mid_ax.legend(
            labels,
            bbox_to_anchor=(0.5, -y_offset_frac),
            loc="upper center",
            bbox_transform=mid_ax.transAxes,
            ncol=len(labels),
            frameon=False
        )
    # if even number of axes, place under point between two central axes
    else:
        ax = axes[len(axes) // 2 - 1]
        ax.legend(
            labels,
            bbox_to_anchor=(1.02, -y_offset_frac),
            loc="upper center",
            ncols=len(labels),
            frameon=False
        )



# GENERAL HELPERS =================================================================================
def status(*message, level=0):
    '''Works like the Python print function but preceded by current datetime in a similar
       format to tensorflow logging.'''
    prefix = " " * (level * 2)
    message = prefix + " ".join([str(m) for m in message])
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f:"), message, flush=True)

def var_name_to_plot_name(var_name):
    """
    Prettify variable names for plotting.

    - Categorical vars: C(var)[cat] -> "var"
      (will not work for multi-level, but currently only handling C(flood)[1.0])
    - Interactions: var1:var2 -> "Interaction: pretty_var1 - pretty_var2"
    - Otherwise moves dataset name to end in brackets, capitalises,
      removes "_values" suffix and replaces unusual strings like pm2p5 with PM2.5.
      aqrean_daqi_nitrogen_dioxide_values -> "DAQI Nitrogen Dioxide (AQRean)"
    """
    def clean_single_var(v):
        # capitalise and split
        name_list = v.title().split("_")

        # remove "Values" (from "_values" suffix)
        if "Values" in name_list:
            name_list.remove("Values")
        # flood only needs to be flood for Bayesian model
        if "Flood" in name_list:
            if "Effect[0]" in name_list:
                name_list.remove("Effect[0]")
            if "Effect[1]" in name_list:
                name_list.remove("Effect[1]")
        
        # replace categorical variable
        if name_list[0].startswith("C("):
            temp = name_list[0].split("(")[1].split(")")[0]
            if "," in temp:
                temp = temp.split(",")[0]
            name_list[0] = temp

        # move dataset names to end
        if "Met" in name_list:
            name_list.remove("Met")
            name_list.append("(Met Office)")
        if "Hydro" in name_list:
            name_list.remove("Hydro")
            name_list.append("(Hydrology)")
        if "Aqrean" in name_list:
            # put DAQI first to keep together when sorted
            if "Daqi" in name_list:
                name_list.remove("Daqi")
                name_list.insert(0, "DAQI")
            if "Daqipair1" in name_list:
                name_list[name_list.index("Daqipair1")] = "VH+H vs. M+L"
            if "Daqipair2" in name_list:
                name_list[name_list.index("Daqipair2")] = "VH+H+M vs. L"
            name_list.remove("Aqrean")
            name_list.append("(AQRean)")
        
        # modify strings with unusual names
        if "Nox" in name_list:
            name_list[name_list.index("Nox")] = "NOx"
        if "Pm10" in name_list:
            name_list[name_list.index("Pm10")] = "PM10"
        if "Pm2P5" in name_list:  # note the P capitalisation
            name_list[name_list.index("Pm2P5")] = "PM2.5"
        if "Sulfur" in name_list:
            name_list[name_list.index("Sulfur")] = "Sulphur"
        if "Rain" in name_list:
            name_list[name_list.index("Rain")] = "Rainfall"
        if "Tmin" in name_list:
            name_list[name_list.index("Tmin")] = "Temperature Minima"
        if "Tmax" in name_list:
            name_list[name_list.index("Tmax")] = "Temperature Maxima"
        if "Imd" in name_list:
            name_list[name_list.index("Imd")] = "IMD District"
        if "Centile" in name_list:
            name_list[name_list.index("Centile")] = "Percentile"
        
        if "Lag" in name_list:
            name_list.remove("Lag")
            for entry in name_list:
                if entry.startswith("Effect"):
                    name_list.remove(entry)
                    break

        return " ".join(name_list)

    if ":" in var_name:
        # handle interaction terms
        var1, var2 = var_name.split(":")
        pretty_var1 = clean_single_var(var1.strip())
        pretty_var2 = clean_single_var(var2.strip())
        return f"Int: {pretty_var1} - {pretty_var2}"  # "Int" is shorter than "Interaction"
    else:
        # handle normal or categorical terms
        return clean_single_var(var_name)
