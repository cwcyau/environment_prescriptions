import json
from shapely.geometry import shape
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import pandas as pd
from tqdm import tqdm

# path to your GeoJSON
geojson_path = "Recorded_Flood_Outlines.geojson"

# load features
with open(geojson_path, "r", encoding="utf-8") as fh:
    data = json.load(fh)

features = data.get("features", [])

# extract start and end dates
start_dates = []
end_dates = []

for f in features:
    start = pd.to_datetime(f["properties"].get("start_date"), errors="coerce")
    end = pd.to_datetime(f["properties"].get("end_date"), errors="coerce")
    if pd.notna(start):
        start_dates.append(start)
    if pd.notna(end):
        end_dates.append(end)

start_dates = pd.Series(start_dates)
end_dates = pd.Series(end_dates)

first_flood = start_dates.min()
last_flood = end_dates.max()

print(f"First flood in dataset: {first_flood.date()}")
print(f"Last flood in dataset: {last_flood.date()}")

# function to plot a specific polygon by index
def plot_polygon(index):
    f = features[index]
    geom = shape(f["geometry"])
    name = f["properties"].get("name", "Unnamed")
    start = f["properties"].get("start_date")
    end = f["properties"].get("end_date")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if geom.geom_type == 'Polygon':
        patch = PolygonPatch(geom, facecolor='red', edgecolor='black', alpha=0.5)
        ax.add_patch(patch)
    elif geom.geom_type == 'MultiPolygon':
        for p in geom.geoms:
            patch = PolygonPatch(p, facecolor='red', edgecolor='black', alpha=0.5)
            ax.add_patch(patch)
    
    ax.set_aspect('equal')
    ax.set_title(f"{name}\n{start} → {end}")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.show()

# Example usage: plot the first 5 polygons
for idx in range(5):
    plot_polygon(idx)