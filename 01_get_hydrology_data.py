import pandas as pd
import numpy as np
import requests
import xarray as xr
from tqdm import tqdm
import time

HYDROLOGY_API_BASE = "https://environment.data.gov.uk/hydrology"
SESSION = requests.Session()

def parse_station_type(type_list):
    """Extract type IDs from API type list."""
    if isinstance(type_list, list):
        return ",".join([t["@id"].split("/")[-1] for t in type_list])
    return None

def extract_guid(url):
    """Extract GUID from full @id URL."""
    if pd.isna(url):
        return None
    return url.rstrip("/").split("/")[-1]

def fetch_hydro_measures(station_guid):
    """Fetch all measures (timeseries) for a station."""
    url = f"{HYDROLOGY_API_BASE}/id/stations/{station_guid}/measures"
    params = {"_view": "default"}
    r = SESSION.get(url, params=params)
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
    r = SESSION.get(url, params=params, timeout=60)
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
    r = SESSION.get(url, params=params, timeout=60)
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


# find all rainfall stations
all_stations = []
base_url = "https://environment.data.gov.uk/hydrology/id/stations.json"
params = {
    "observedProperty": "rainfall",
    "_limit": 500,
    "_offset": 0
}
print("Fetching rainfall station metadata...")
while True:
    r = SESSION.get(base_url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    items = data.get("items", [])
    if not items:
        break

    df = pd.json_normalize(items)
    all_stations.append(df)
    print(f"Fetched {len(items)} stations (total so far {sum(len(x) for x in all_stations)})")
    if len(items) < params["_limit"]:
        break  # last page reached

    params["_offset"] += params["_limit"]
    time.sleep(0.2)

stations_df = pd.concat(all_stations, ignore_index=True)
print(f"Total rainfall stations fetched: {len(stations_df)}")

# extract relevant fields
stations_df = stations_df.rename(columns={
    "@id": "station_id",
    "label": "station_name",
    "type": "station_type",
    "lat": "latitude",
    "long": "longitude"
})[["station_id", "station_name", "station_type", "latitude", "longitude"]]

# clean the station id and type
stations_df["station_id"] = stations_df["station_id"].apply(extract_guid)
stations_df["station_type"] = stations_df["station_type"].apply(parse_station_type)

# remove any stations without coordinates
stations_df = stations_df.dropna(subset=["latitude", "longitude"])
print(f"Rainfall stations with coordinates: {len(stations_df)}")

# convert to xarray
ds = xr.Dataset.from_dataframe(stations_df.set_index("station_id"))

# add measurements
all_dates = []
station_data = {}
for station_id in tqdm(ds['station_id'].values,
                       desc="Fetching station data",
                       total=len(ds['station_id'].values)):
    # get relevant measures for this station
    measures = fetch_hydro_measures(station_id)
    measure_id = [m for m in measures if "rainfall-t-86400" in m]

    if len(measure_id) == 0:
        print(f"No rainfall measure found for station {station_id}")
        continue
    elif len(measure_id) > 1:
        print(f"Warning: multiple rainfall measures found for station {station_id}, using the first one.")
    measure_id = measure_id[0]

    # fetch readings
    datetimes, readings = fetch_hydro_readings(measure_id)

    if len(datetimes) == 0:
        print(f"No readings found for station {station_id}, measure {measure_id}.")
        continue

    # save for later
    station_data[station_id] = (datetimes, readings)
    all_dates.extend(datetimes)

# create unified date index
all_dates = pd.to_datetime(sorted(set(all_dates)))
n_dates = len(all_dates)
n_stations = len(ds['station_id'])

# prepare arrays for readings
readings_array = np.full((n_dates, n_stations), np.nan, dtype=np.float32)

# fill array for each station
station_id_list = ds['station_id'].values
for j, station_id in enumerate(station_id_list):
    if station_id not in station_data:
        continue
    dt, reads = station_data[station_id]
    dt_idx = pd.Index(all_dates).get_indexer(pd.to_datetime(dt))
    readings_array[dt_idx, j] = reads

# add to dataset
ds = ds.assign_coords(date=("date", all_dates))
ds["rainfall"] = (("date", "station_id"), readings_array)

# exclude stations with all NaN readings
all_nan_mask = np.all(np.isnan(ds["rainfall"]), axis=0)
if np.any(all_nan_mask):
    ds = ds.drop_isel(station_id=all_nan_mask)

# save to netcdf
output_file = "data/hydrology_rainfall_stations.nc"
ds.to_netcdf(output_file)
print(f"Saved rainfall station dataset to {output_file}")
print(ds)
