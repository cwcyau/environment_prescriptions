import pandas as pd
import requests
import xarray as xr
import numpy as np
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
bnf_codes_list = [  # 0501 (antibiotics), 02 (cardiovascular), 03 (respiratory)
    ["02"],
    ["03"],
    ["0501"],
    ["02", "03", "0501"]
]
start_date = "2020-09-01"
end_date = "2025-08-01"
dates = pd.date_range(start_date, end_date, freq="MS").strftime("%Y-%m-%d")

for bnf_codes in bnf_codes_list:
    print("Getting data for BNF codes:", bnf_codes)
    # ensure list
    if isinstance(bnf_codes, str):
        bnf_codes = [bnf_codes]

    # =================================================================================================
    # fetch prescription data
    all_data = []
    for date in tqdm(dates, desc="Fetching monthly prescribing data", total=len(dates)):
        monthly_frames = []
        for bnf_code in bnf_codes:
            url = f"https://openprescribing.net/api/1.0/spending_by_practice.json?code={bnf_code}&date={date}"
            df = api_fetch(url)
            if df is not None:
                df["bnf_code_prefix"] = bnf_code
                df["date"] = pd.to_datetime(date)
                monthly_frames.append(df)
            time.sleep(0.2)
        if monthly_frames:
            all_data.append(pd.concat(monthly_frames, ignore_index=True))

    if not all_data:
        raise ValueError("No prescribing data retrieved — check BNF codes or date range.")

    prescription_data = pd.concat(all_data, ignore_index=True)
    print(f"Collected {len(prescription_data)} prescribing records across {len(dates)} months")

    # sum over BNF prefixes if multiple
    numeric_cols = ["items", "quantity", "actual_cost"]
    prescription_data[numeric_cols] = prescription_data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if len(bnf_codes) > 1:
        prescription_data = (
            prescription_data
            .groupby(["date", "row_id"], as_index=False)[numeric_cols]
            .sum()
        )

    # =================================================================================================
    # fetch practice locations
    practice_ids = sorted(prescription_data["row_id"].dropna().unique().tolist())
    print(f"Unique practices: {len(practice_ids)}")

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

    # =================================================================================================
    # set date and row_id as indexes and convert to xarray
    prescription_data = prescription_data.set_index(["date", "row_id"])[numeric_cols]
    ds = xr.Dataset.from_dataframe(prescription_data)

    # map latitude and longitude to practices
    practices = ds['row_id'].values
    loc_map_lat = locations_df.set_index('row_id')['latitude'].to_dict()
    loc_map_lon = locations_df.set_index('row_id')['longitude'].to_dict()
    lat_vec = np.array([loc_map_lat.get(r, np.nan) for r in practices], dtype=float)
    lon_vec = np.array([loc_map_lon.get(r, np.nan) for r in practices], dtype=float)

    # attach as 1-D coordinates on the row_id dimension
    ds = ds.assign_coords({
        'latitude': (('row_id',), lat_vec),
        'longitude': (('row_id',), lon_vec)
    })

    # drop practices that still lack coordinates (lat or lon is NaN)
    invalid_mask = np.isnan(ds.coords['latitude'].values) | np.isnan(ds.coords['longitude'].values)
    if invalid_mask.any():
        invalid_ids = list(ds['row_id'].values[invalid_mask])
        print(f"Dropping {len(invalid_ids)} practices with missing coordinates")
        ds = ds.drop_sel(row_id=invalid_ids)

    # =================================================================================================
    # save
    bnf_str = "_".join(bnf_codes)
    filename = f"data/prescriptions_{bnf_str}_{start_date[:-3]}_{end_date[:-3]}.nc"
    ds.to_netcdf(filename)
    print(ds)
    print(f"Saved dataset to {filename}")
    if bnf_codes != bnf_codes_list[-1]:
        time.sleep(10)
