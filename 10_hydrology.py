import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm
import time

def api_fetch(url, params=None, timeout=60):
    """Fetch JSON data from an API and return as a DataFrame."""
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    df = pd.json_normalize(data.get("items", []))
    return df if not df.empty else None

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

# find all the hydrology stations
all_stations = []
url = "https://environment.data.gov.uk/hydrology/id/stations.json"
print("Fetching station metadata...")
while url:
    df = api_fetch(url)
    if df is not None:
        all_stations.append(df)
    r = requests.get(url)
    r.raise_for_status()
    url = r.json().get("meta", {}).get("next")
    time.sleep(0.2)

stations_df = pd.concat(all_stations, ignore_index=True)
print(f"Total stations fetched: {len(stations_df)}")

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
print(f"Stations with coordinates: {len(stations_df)}")

# save as xarray netcdf file
ds = xr.Dataset.from_dataframe(stations_df.set_index("station_id"))
output_file = "hydrology_stations.nc"
ds.to_netcdf(output_file)
print(f"Saved station dataset to {output_file}")
print(ds)