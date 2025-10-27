import pandas as pd
import numpy as np
import requests
from scipy.spatial import cKDTree
import xarray as xr
from tqdm import tqdm

def fetch_station_measures(station_guid, observed_property=None):
    """Fetch all measures (timeseries) for a station."""
    url = f"{API_BASE}/id/stations/{station_guid}/measures"
    params = {"_view": "default"}
    if observed_property:
        params["observedProperty"] = observed_property
    r = requests.get(url, params=params)
    r.raise_for_status()
    items = r.json().get("items", [])
    # return a list of measure IDs
    return [m["@id"].split("/")[-1] for m in items]

def fetch_measure_readings(measure_id, start_date, end_date):
    """Fetch readings for a measure using min-date and max-date filters."""
    url = f"{API_BASE}/id/measures/{measure_id}/readings"
    params = {
        "mineq-date": start_date,
        "max-date": end_date,
        "_limit": 2000000  # hard cap allowed
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    items = r.json().get("items", [])
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    return df

def create_flags(measurements, kind, z_thresh=2.0, min_persist=2):
    """Create persistent anomaly flags using monthly climatology and robust std (MAD).
    kind: 'drought' -> negative anomalies; 'flood' -> positive anomalies.
    Returns a Series indexed by Period (monthly) with integer flags (0/1).
    """
    if measurements.empty:
        return pd.Series(dtype=int)

    df = measurements.copy()
    df["dateTime"] = pd.to_datetime(df["dateTime"])
    # aggregate to monthly values (median reduces sensitivity to single spikes)
    monthly = df.set_index("dateTime")["value"].resample("MS").median().to_frame("value")
    monthly.index = monthly.index.to_period("M")

    # monthly climatology by calendar month (1..12)
    monthly["month"] = monthly.index.month
    clim = monthly.groupby("month")["value"].agg(["median", lambda x: (x - x.median()).abs().median()])
    clim.columns = ["clim_median", "clim_mad"]
    clim["clim_std_robust"] = clim["clim_mad"] * 1.4826  # approx std from MAD

    monthly = monthly.merge(clim, left_on="month", right_index=True, how="left")
    monthly["anom"] = (monthly["value"] - monthly["clim_median"]) / (monthly["clim_std_robust"] + 1e-9)

    if kind == "drought":
        raw_flag = monthly["anom"] <= -abs(z_thresh)
    else:
        raw_flag = monthly["anom"] >= abs(z_thresh)

    # require persistence: keep only runs of consecutive True of length >= min_persist
    runs = (raw_flag != raw_flag.shift()).cumsum()
    run_lengths = raw_flag.groupby(runs).transform("sum")
    persisted = raw_flag & (run_lengths >= min_persist)

    flags = persisted.astype(int)
    flags.index = monthly.index
    return flags

def get_station_flags(station_guid, start_date, end_date):
    """Fetch drought and flood flags for a given station using robust anomaly method."""
    measures = fetch_station_measures(station_guid, observed_property="waterFlow")

    # prefer min/max style measures if present, otherwise fall back to first measure
    min_measures = [m for m in measures if "-min-" in m]
    max_measures = [m for m in measures if "-max-" in m]

    df_min = fetch_measure_readings(min_measures[0], start_date, end_date) if min_measures else pd.DataFrame()
    df_max = fetch_measure_readings(max_measures[0], start_date, end_date) if max_measures else pd.DataFrame()

    # create flags using robust seasonal anomaly method
    drought_flags = create_flags(df_min, kind="drought") if not df_min.empty else pd.Series(dtype=int)
    flood_flags = create_flags(df_max, kind="flood") if not df_max.empty else pd.Series(dtype=int)

    # align and combine to a monthly index
    idx = drought_flags.index.union(flood_flags.index)
    combined = pd.DataFrame(index=idx)
    combined["drought"] = drought_flags.reindex(idx).fillna(0).astype(int)
    combined["flood"] = flood_flags.reindex(idx).fillna(0).astype(int)
    return combined

# parameters
API_BASE = "https://environment.data.gov.uk/hydrology"
prescriptions_file = "prescriptions_0501_02_03_2020-09-01_2025-08-01.nc"
hydrology_file = "hydrology_stations.nc"

# load prescription dataset
prescriptions_ds = xr.load_dataset(prescriptions_file)
prescriptions_df = prescriptions_ds.to_dataframe().reset_index()
print(f"Loaded prescription data: {prescriptions_df.shape[0]} rows")
prescriptions_df = prescriptions_df.dropna(subset=["latitude", "longitude"])
print(f"After dropping missing lat/lon: {prescriptions_df.shape[0]} rows")

# load hydrology station dataset
stations_ds = xr.load_dataset(hydrology_file)
stations_df = stations_ds.to_dataframe().reset_index()
print(f"Loaded {len(stations_df)} stations")

# find nearest station for each practice
tree = cKDTree(stations_df[["latitude", "longitude"]].values)
distances, indices = tree.query(prescriptions_df[["latitude", "longitude"]].values)
prescriptions_df["nearest_station_id"] = stations_df.iloc[indices]["station_id"].values

# find drought/flood flags for each station
start_date = prescriptions_df["date"].min().strftime("%Y-%m-%d")
end_date = prescriptions_df["date"].max().strftime("%Y-%m-%d")
unique_station_ids = stations_df["station_id"].unique()
station_flags = {}
for station_guid in tqdm(unique_station_ids, desc="Fetching station data", total=len(unique_station_ids)):
    try:
        station_flags[station_guid] = get_station_flags(station_guid, start_date, end_date)
    except Exception as e:
        print(f"Error fetching data for station {station_guid}: {e}")
        station_flags[station_guid] = pd.DataFrame(columns=["drought", "flood"])

# map flags back to prescriptions dataframe
flood_list = []
drought_list = []
for idx, row in tqdm(prescriptions_df.iterrows(), desc="Mapping flags to prescriptions", total=len(prescriptions_df)):
    station_guid = row["nearest_station_id"]
    month = pd.Period(row["date"], freq="M")
    flags = station_flags.get(station_guid)
    if flags is not None and month in flags.index:
        flood_list.append(flags.loc[month, "flood"])
        drought_list.append(flags.loc[month, "drought"])
    else:
        flood_list.append(np.nan)
        drought_list.append(np.nan)

prescriptions_df["flood"] = flood_list
prescriptions_df["drought"] = drought_list

# convert to xarray and save
numeric_cols = ["items", "quantity", "actual_cost", "latitude", "longitude", "flood", "drought"]
ds_out = xr.Dataset.from_dataframe(
    prescriptions_df.set_index(["date", "row_id"])[numeric_cols]
)
filename = prescriptions_file.replace(".nc", "_flags.nc")
prescriptions_ds.close()
try:
    ds_out.to_netcdf(prescriptions_file)
except Exception as e:
    ds_out.to_netcdf(filename)
print(ds_out)
