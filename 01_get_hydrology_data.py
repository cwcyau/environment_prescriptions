import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm
import time

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


# find all rainfall stations
all_stations = []
base_url = "https://environment.data.gov.uk/hydrology/id/stations.json"
params = {
    "observedProperty": "rainfall",
    "_limit": 500,
    "_offset": 0
}

print("Fetching rainfall station metadata...")

# iterate pages manually using _offset and _limit
while True:
    r = requests.get(base_url, params=params, timeout=60)
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

# save as xarray netcdf file
ds = xr.Dataset.from_dataframe(stations_df.set_index("station_id"))
output_file = "data/hydrology_rainfall_stations.nc"
ds.to_netcdf(output_file)
print(f"Saved rainfall station dataset to {output_file}")
print(ds)
