import json
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from pyproj import Transformer
from tqdm import tqdm

# parameters
nc_path = "prescriptions_0501_02_03_2020-09-01_2025-08-01.nc"
geojson_path = "Recorded_Flood_Outlines.geojson"
search_radius_m = 5000  # ≈ 5 km buffer
simplify_tol = 50       # simplify complex polygons by ~50m to reduce vertices

# load prescriptions dataset
print("Loading datasets...")
ds = xr.load_dataset(nc_path)
df = ds.to_dataframe().reset_index()

# convert lat/lon to projected coordinates (EPSG:27700, British National Grid)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
df["x"], df["y"] = transformer.transform(df["longitude"].values, df["latitude"].values)
df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

# load flood polygons from geojson
with open(geojson_path, "r", encoding="utf-8") as fh:
    data = json.load(fh)
features = data.get("features", [])

# preprocess flood polygons
geoms = []
months = []
for f in tqdm(features, desc="Processing flood polygons", total=len(features)):
    start = pd.to_datetime(f["properties"].get("start_date"), errors="coerce")
    end = pd.to_datetime(f["properties"].get("end_date"), errors="coerce")
    if pd.isna(start) or start.year < 2020:
        continue
    geom = shape(f["geometry"])
    geom = geom.simplify(simplify_tol, preserve_topology=True)
    geom = geom.buffer(search_radius_m)
    end = end or start
    # assign polygon to each month it spans
    for m in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        geoms.append(geom)
        months.append(m)

geom_months = np.array(months)
tree = STRtree(geoms)

# generate flood_geo flags
df["flood_geo"] = 0
for month, grp in tqdm(df.groupby("month"), total=df["month"].nunique(), desc="Flagging flood months"):
    idxs = np.where(geom_months == month)[0]
    if len(idxs) == 0:
        continue

    month_geoms = [geoms[i] for i in idxs]
    month_tree = STRtree(month_geoms)

    for i, row in grp.iterrows():
        pt = Point(row["x"], row["y"])
        nearby_idxs = month_tree.query(pt)
        if any(month_geoms[j].intersects(pt) for j in nearby_idxs):
            df.at[i, "flood_geo"] = 1

# save
ds_out = ds.copy()
pivoted = df.pivot(index="row_id", columns="date", values="flood_geo").fillna(np.nan).astype(np.float64).T
ds_out["flood_geo"] = (("date", "row_id"), pivoted.values)
ds.close()
try:
    ds_out.to_netcdf(nc_path)
except Exception:
    ds_out.to_netcdf(nc_path.replace(".nc", "_geo.nc"))
print(ds_out)