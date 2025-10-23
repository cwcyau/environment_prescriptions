import json
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from shapely.ops import unary_union
from pyproj import Transformer
from tqdm import tqdm

# parameters
nc_path = "prescriptions_0501_2020-09-01_2025-08-01_flags.nc"
geojson_path = "Recorded_Flood_Outlines.geojson"
search_radius_m = 5000  # ≈ 5 km buffer

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

# preprocess flood polygons efficiently
geoms, months = [], []
for f in tqdm(features, desc="Processing flood polygons", total=len(features)):
    props = f.get("properties", {})
    start = pd.to_datetime(props.get("start_date"), errors="coerce")
    end = pd.to_datetime(props.get("end_date"), errors="coerce")
    if pd.isna(start):
        continue
    geom_obj = shape(f["geometry"]).simplify(200)  # simplify to ~200 m
    m1, m2 = start.to_period("M"), (end or start).to_period("M")
    for m in pd.period_range(m1, m2, freq="M"):
        geoms.append(geom_obj)
        months.append(m)




# CHECK GPT DIDNT BREAK THIS PART

# Prepare month-index mapping alongside geoms
geoms = []
months = []
for f in tqdm(features, desc="Processing flood polygons", total=len(features)):
    g = shape(f["geometry"])
    start = pd.to_datetime(f["properties"].get("start_date"), errors="coerce")
    end = pd.to_datetime(f["properties"].get("end_date"), errors="coerce")
    if pd.isna(start):
        continue
    m1, m2 = start.to_period("M"), (end or start).to_period("M")
    for m in pd.period_range(m1, m2, freq="M"):
        geoms.append(g.buffer(search_radius_m))
        months.append(m)

tree = STRtree(geoms)
geom_months = np.array(months)

# Generate flood_geo flags
df["flood_geo"] = 0
for m, grp in tqdm(df.groupby("month"), total=df["month"].nunique(), desc="Flood flagging"):
    # subset polygons for this month
    idxs = np.where(geom_months == m)[0]
    if len(idxs) == 0:
        continue
    month_geoms = [geoms[i] for i in idxs]
    month_tree = STRtree(month_geoms)
    
    for i, row in grp.iterrows():
        pt = Point(row["x"], row["y"])
        if any(g.intersects(pt) for g in month_tree.query(pt)):
            df.at[i, "flood_geo"] = 1





# build one big spatial index instead of monthly rebuilds
tree = STRtree(geoms)
geom_to_month = {id(g): m for g, m in zip(geoms, months)}

# generate flood_geo flags
df["flood_geo"] = 0
points = [Point(xy) for xy in zip(df["x"], df["y"])]

for i, pt in tqdm(enumerate(points), total=len(points), desc="Flagging floods"):
    # query once for all nearby polygons
    nearby = tree.query(pt.buffer(search_radius_m))
    if not len(nearby):
        continue
    # check if any of those polygons overlap the same month
    practice_month = df.at[i, "month"]
    for g in nearby:
        if geom_to_month[id(g)] == practice_month and g.buffer(search_radius_m).intersects(pt):
            df.at[i, "flood_geo"] = 1
            break

# save
ds_out = ds.copy()
pivoted = df.pivot(index="row_id", columns="date", values="flood_geo").fillna(0).astype(np.int8)
ds_out["flood_geo"] = (("row_id", "date"), pivoted.values)
ds.close()
try:
    ds_out.to_netcdf(nc_path)
    print(ds_out)
except Exception as e:
    ds_out.to_netcdf(nc_path.replace(".nc", "_geo.nc"))