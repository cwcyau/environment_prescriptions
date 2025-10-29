import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import requests
import time

# parameters
NUMERIC_COLS = ["ITEMS", "QUANTITY", "ACT COST"]
ORIGINAL_COLS = ["items", "quantity", "actual_cost"]
csv_dir = Path("data/prescriptions_pre20200101/csvs")
prescriptions_paths = [  # takes ~20-30 minutes per file
    "data/prescriptions_02_03_0501_2020-09_2025-08.nc",
    "data/prescriptions_02_2020-09_2025-08.nc",
    "data/prescriptions_03_2020-09_2025-08.nc",
    "data/prescriptions_0501_2020-09_2025-08.nc"
]

def api_fetch(url, params=None, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "features" in data:
        data = data["features"]
    df = pd.json_normalize(data)
    return df if not df.empty else pd.DataFrame()

def read_and_aggregate(csv_path, bnf_prefixes, existing_practices):
    """
    Read CSV and return:
      - df_agg: aggregated DataFrame for ALL practices (old + new) matching bnf_prefixes
                with columns renamed to original dataset names
      - new_practices: set of practice_ids present in CSV but not in existing_practices
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip().str.upper()

    # parse date + practice
    df["date"] = pd.to_datetime(df["PERIOD"].astype(str).str.strip() + "01",
                                format="%Y%m%d", errors="coerce")
    df["practice_id"] = df["PRACTICE"].astype(str).str.strip()

    # numeric columns
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")

    df["BNF CODE"] = df["BNF CODE"].astype(str).str.strip()

    # filter by BNF prefixes
    if not bnf_prefixes:
        return None, set()
    df = df[df["BNF CODE"].str.startswith(tuple(bnf_prefixes))]

    # drop invalid rows
    df = df[df["practice_id"].notna() & df["date"].notna()]
    if df.empty:
        return None, set()

    # determine new practice ids (relative to existing_practices)
    if existing_practices is not None:
        new_practices = set(df["practice_id"].unique()) - set(existing_practices)
    else:
        new_practices = set(df["practice_id"].unique())

    # aggregate for all practices (we keep new-practice aggregates too)
    df_agg = df.groupby(["date", "practice_id"], as_index=False)[NUMERIC_COLS].sum()

    # rename numeric columns to match existing dataset variable names
    df_agg = df_agg.rename(columns=dict(zip(NUMERIC_COLS, ORIGINAL_COLS)))

    return df_agg, new_practices

def fetch_locations(practice_ids, batch_size=100):
    """Query openprescribing org_location API in batches and return DataFrame of practice_id, latitude, longitude."""
    all_locations = []
    for i in range(0, len(practice_ids), batch_size):
        batch = practice_ids[i:i+batch_size]
        url = f"https://openprescribing.net/api/1.0/org_location/?q={','.join(batch)}"
        df = api_fetch(url)
        if not df.empty:
            df = df.rename(columns={
                "properties.code": "practice_id",
                "geometry.coordinates": "lonlat"
            })
            df["longitude"] = df["lonlat"].apply(lambda x: x[0] if isinstance(x, list) else np.nan)
            df["latitude"]  = df["lonlat"].apply(lambda x: x[1] if isinstance(x, list) else np.nan)
            all_locations.append(df[["practice_id", "latitude", "longitude"]])
        time.sleep(0.5)  # polite delay
    if all_locations:
        return pd.concat(all_locations, ignore_index=True).dropna(subset=["latitude", "longitude"])
    return pd.DataFrame(columns=["practice_id", "latitude", "longitude"])

# main loop
for nc_path_str in prescriptions_paths:
    nc_path = Path(nc_path_str)
    print(f"\nLoading dataset: {nc_path}")
    ds_old = xr.load_dataset(nc_path)
    existing_practices = set(ds_old["practice_id"].values)
    print(f"Existing practices in nc: {len(existing_practices)}")

    # extract BNF codes from filename (everything between "prescriptions_" and the date part)
    bnf_codes = nc_path.stem.split("_")[1:-2]
    print(f"BNF prefixes for this file: {bnf_codes}")
    print(f"Existing date range: {str(ds_old['date'].values.min())[:10]} → {str(ds_old['date'].values.max())[:10]}")

    # collect aggregated data from CSVs and record new practice ids
    all_new_data = []
    all_new_practices = set()
    csv_list = sorted(csv_dir.glob("*.csv"))
    print(f"Processing {len(csv_list)} CSV files")
    for csv_file in tqdm(csv_list, desc="Processing CSVs"):
        try:
            df_agg, new_pracs = read_and_aggregate(csv_file, bnf_codes, existing_practices)
            if df_agg is not None:
                all_new_data.append(df_agg)
            all_new_practices.update(new_pracs)
        except Exception as e:
            print(f"  Failed to parse {csv_file.name}: {e}")

    if not all_new_data:
        print("No CSV data found for this file, skipping.")
        continue

    # combine aggregated CSV data (includes both old & new practices' aggregates)
    new_df = pd.concat(all_new_data, ignore_index=True)

    # at this point new_df contains aggregated rows for practices that might already be in ds_old
    # and also for new practices discovered in CSVs (listed in all_new_practices).

    # fetch lat/lon only for truly new practice ids (those not in existing_practices)
    truly_new_practices = sorted([p for p in all_new_practices if p not in existing_practices])
    print(f"Found {len(truly_new_practices)} truly new practice_ids (will attempt to fetch locations)")

    new_locations_df = pd.DataFrame(columns=["practice_id", "latitude", "longitude"])
    if truly_new_practices:
        new_locations_df = fetch_locations(truly_new_practices)

    # build loc maps initialised from old ds
    loc_map_lat = dict(zip(ds_old["practice_id"].values, ds_old["latitude"].values))
    loc_map_lon = dict(zip(ds_old["practice_id"].values, ds_old["longitude"].values))

    # update maps with newly fetched locations
    for _, row in new_locations_df.iterrows():
        loc_map_lat[row["practice_id"]] = row["latitude"]
        loc_map_lon[row["practice_id"]] = row["longitude"]

    # convert new_df to xarray
    new_df = new_df.set_index(["date", "practice_id"])
    ds_new = xr.Dataset.from_dataframe(new_df)

    # assign coordinates (1-D over practice_id)
    practices = ds_new["practice_id"].values
    lat_arr = np.array([loc_map_lat.get(p, np.nan) for p in practices], dtype=float)
    lon_arr = np.array([loc_map_lon.get(p, np.nan) for p in practices], dtype=float)
    ds_new = ds_new.assign_coords({
        "latitude": ("practice_id", lat_arr),
        "longitude": ("practice_id", lon_arr),
    })

    # drop practices without lat/lon (we cannot keep practices without coordinates)
    missing_mask = np.isnan(ds_new["latitude"].values) | np.isnan(ds_new["longitude"].values)
    if missing_mask.any():
        missing_pracs = ds_new["practice_id"].values[missing_mask]
        print(f"Dropping {len(missing_pracs)} practices with missing coordinates (could not find lat/lon).")
        ds_new = ds_new.drop_sel(practice_id=missing_pracs)

    # merge with existing dataset along date dim
    ds_combined = xr.concat([ds_old, ds_new], dim="date").sortby("date")

    # latitude and longitude often get set to 2D (date, practice_id) during concat, so we need to fix that
    # some rows may have NaNs so we take the first non-NaN value for each practice_id
    lat_var = ds_combined["latitude"]
    lon_var = ds_combined["longitude"]
    if lat_var.ndim == 2:
        lat_1d = np.array([lat_var.sel(practice_id=pid).values[
                                    ~np.isnan(lat_var.sel(practice_id=pid).values)][0]
                        for pid in ds_combined["practice_id"].values])
    else:
        lat_1d = ds_combined["latitude"].values
    if lon_var.ndim == 2:
        lon_1d = np.array([lon_var.sel(practice_id=pid).values[
                                    ~np.isnan(lon_var.sel(practice_id=pid).values)][0]
                        for pid in ds_combined["practice_id"].values])
    else:
        lon_1d = ds_combined["longitude"].values
    ds = ds.assign_coords({
        "latitude": ("practice_id", lat_1d),
        "longitude": ("practice_id", lon_1d)
    })

    # save with updated date range in filename
    new_start = str(ds_combined["date"].values.min())[:7]
    new_end = str(ds_combined["date"].values.max())[:7]
    old_date_range_token = f"{str(ds_old['date'].values.min())[:7]}_{str(ds_old['date'].values.max())[:7]}"
    new_filename = nc_path.parent / nc_path.name.replace(old_date_range_token, f"{new_start}_{new_end}")

    ds_combined.to_netcdf(new_filename)
    print(f"Saved {new_filename}: {len(ds_combined['practice_id'])} practices, {len(ds_combined['date'])} months")
