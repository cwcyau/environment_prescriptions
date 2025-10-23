import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm
import time

def api_fetch(url, params=None, timeout=60):
    """Fetch and flatten JSON or GeoJSON data."""
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "features" in data:
        data = data["features"]
    df = pd.json_normalize(data)
    return df if not df.empty else None


# parameters
bnf_code = "0501"  # antibiotics
start_date = "2020-09-01"
end_date = "2025-08-01"
dates = pd.date_range(start_date, end_date, freq="MS").strftime("%Y-%m-%d")

# get prescription data for each month in the date range
all_data = []

for date in tqdm(dates, desc="Fetching monthly prescribing data", total=len(dates)):
    url = f"https://openprescribing.net/api/1.0/spending_by_practice.json?code={bnf_code}&date={date}"
    df = api_fetch(url)
    if df is not None:
        df["date"] = pd.to_datetime(date)
        all_data.append(df)
        time.sleep(0.3)

prescription_data = pd.concat(all_data, ignore_index=True)
print(f"Collected {len(prescription_data)} prescribing records across {len(dates)} months")

# extract all the unique practice IDs
practice_ids = sorted(prescription_data["row_id"].dropna().unique().tolist())
print(f"Unique practices: {len(practice_ids)}")

# get practice locations in batches
batch_size = 100
location_results = []

for i in tqdm(range(0, len(practice_ids), batch_size), desc="Fetching practice locations"):
    batch = practice_ids[i:i+batch_size]
    query = ",".join(batch)
    url = f"https://openprescribing.net/api/1.0/org_location/?q={query}"
    df = api_fetch(url)
    if df is not None:
        location_results.append(df)
    else:
        print(f"Warning: No data returned for query: {query}")
    time.sleep(0.5)

locations_df = pd.concat(location_results, ignore_index=True)

# clean and format locations
locations_df = locations_df.rename(
    columns={
        "properties.code": "row_id",
        "properties.name": "practice_name",
        "geometry.coordinates": "lonlat"
    }
)[["row_id", "practice_name", "lonlat"]]
locations_df["longitude"] = locations_df["lonlat"].apply(lambda x: x[0] if isinstance(x, list) else None)
locations_df["latitude"]  = locations_df["lonlat"].apply(lambda x: x[1] if isinstance(x, list) else None)
print(f"Fetched {len(locations_df)} location records")

# merge and convert to xarray
merged_df = prescription_data.merge(locations_df, on="row_id", how="left")
numeric_cols = ["items", "quantity", "actual_cost"]
merged_df = merged_df.set_index(["date", "row_id"])[numeric_cols + ["latitude", "longitude"]]
ds = xr.Dataset.from_dataframe(merged_df)

# save
filename = f"prescriptions_{bnf_code}_{start_date}_{end_date}.nc"
ds.to_netcdf()
print(ds)
print(f"Saved dataset to {filename}")
