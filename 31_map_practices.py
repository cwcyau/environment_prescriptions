import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ds = xr.open_dataset("data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc")

# Example points (latitude, longitude)
points = [
    (lon, lat) for lon, lat in zip(ds.longitude.values, ds.latitude.values)
]
lat_min = min(ds.latitude.values)
lat_max = max(ds.latitude.values)
lon_min = min(ds.longitude.values)
lon_max = max(ds.longitude.values)

# Create the figure with PlateCarree projection (lat/lon)
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_aspect('equal', adjustable='datalim')

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

# Set the map extent roughly around the UK
ax.set_extent([lon_min - 1, lon_max + 1, lat_min - 1, lat_max + 1], crs=ccrs.PlateCarree())

# Plot points
for lon, lat in points:
    ax.plot(lon, lat, marker='o', markersize=2, color='red',
            transform=ccrs.PlateCarree())

plt.savefig("outputs/practice_map.png", bbox_inches='tight', dpi=600)