import pandas as pd
import numpy as np
import shapely as shp
import xarray as xr

# file paths
index_2025_file = "data/deprivation_indices/la-summary-2025.json"
index_2019_file = "data/deprivation_indices/2019_data.xlsx"
index_2015_file = "data/deprivation_indices/2015_data.xlsx"
loc_2024_file = "data/deprivation_indices/la-2024-lonlats.csv"
loc_2019_file = "data/deprivation_indices/la-2019-lonlats.csv"

# load data
index_2025_data = pd.read_json(index_2025_file)
index_2019_data = pd.read_excel(index_2019_file, sheet_name='IMD')
index_2015_data = pd.read_excel(index_2015_file, sheet_name='data')
loc_2024_data = pd.read_csv(loc_2024_file)
loc_2019_data = pd.read_csv(loc_2019_file)

# 2015 column names
rank_col_2015 = "IMD 2015 (recast to 2019 LAD boundaries) - Rank of average rank "
code_col_2015 = "Local Authority District code (2019)"
name_col_2015 = "Local Authority District name (2019)"

# 2019 column names
rank_col_2019 = "IMD - Rank of average rank "
code_col_2019 = "Local Authority District code (2019)"
name_col_2019 = "Local Authority District name (2019)"
loc_code_col_2019 = "lad19cd"
loc_long_col_2019 = "long"
loc_lat_col_2019 = "lat"

# 2025 column names
rank_col_2025 = "IMD"
code_col_2025 = "Local Authority District code (2024)"
name_col_2025 = "Local Authority District name (2024)"
loc_code_col_2025 = "LAD24CD"
loc_long_col_2025 = "LONG"
loc_lat_col_2025 = "LAT"

# collect all codes and names for each year
unique_codes_2015, unique_indices_2015 = np.unique(index_2015_data[code_col_2015].values, return_index=True)
unique_names_2015 = index_2015_data[name_col_2015].values[unique_indices_2015]
unique_codes_2019, unique_indices_2019 = np.unique(index_2019_data[code_col_2019].values, return_index=True)
unique_names_2019 = index_2019_data[name_col_2019].values[unique_indices_2019]
unique_codes_2025, unique_indices_2025 = np.unique(index_2025_data[code_col_2025].values, return_index=True)
unique_names_2025 = index_2025_data[name_col_2025].values[unique_indices_2025]

# collect all unique codes and names across all years
all_codes = np.concatenate([unique_codes_2015, unique_codes_2019, unique_codes_2025])
all_names = np.concatenate([unique_names_2015, unique_names_2019, unique_names_2025])
unique_codes, unique_indices = np.unique(all_codes, return_index=True)
unique_names = all_names[unique_indices]

# calculate centiles for index of multiple deprivation rank of average ranks and retrieve lon/lats
ds = xr.Dataset()
ds = ds.assign_coords({"LAD_code": unique_codes,
                       "LAD_name": ("LAD_code", unique_names),
                       "year": [2015, 2019, 2025]})
centiles = np.zeros((len(unique_codes), 3)) * np.nan
lons = np.zeros((len(unique_codes), 3)) * np.nan
lats = np.zeros((len(unique_codes), 3)) * np.nan
for i, code in enumerate(unique_codes):
    # rank centiles
    temp = index_2015_data[index_2015_data[code_col_2015] == code][rank_col_2015]
    if len(temp) > 0:
        centiles[i, 0] = np.floor(100 * temp.values[0] / len(index_2015_data))
    temp = index_2019_data[index_2019_data[code_col_2019] == code][rank_col_2019]
    if len(temp) > 0:
        centiles[i, 1] = np.floor(100 * temp.values[0] / len(index_2019_data))
    temp = index_2025_data[index_2025_data[code_col_2025] == code][rank_col_2025]
    if len(temp) > 0:
        centiles[i, 2] = np.floor(100 * temp.values[0] / len(index_2025_data))

    # lon/lats
    temp = loc_2019_data[loc_2019_data[loc_code_col_2019] == code]
    if len(temp) > 0:
        lons[i, [0, 1]] = temp[loc_long_col_2019].values[0]
        lats[i, [0, 1]] = temp[loc_lat_col_2019].values[0]
    temp = loc_2024_data[loc_2024_data[loc_code_col_2025] == code]
    if len(temp) > 0:
        lons[i, 2] = temp[loc_long_col_2025].values[0]
        lats[i, 2] = temp[loc_lat_col_2025].values[0]

ds['centile'] = (("LAD_code", "year"), centiles)
ds = ds.assign_coords({"longitude": (("LAD_code", "year"), lons),
                       "latitude": (("LAD_code", "year"), lats)})

# save dataset
ds.to_netcdf("data/local_authority_district_index_multiple_deprivation_centiles.nc")



# plot all loc_2024_data["LONG", "LAT"] points and loc_2019_data["long", "lat"] points on same plot to see if they align
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# plt.scatter(loc_2024_data["LONG"], loc_2024_data["LAT"], label='2024', alpha=0.5)
# plt.scatter(loc_2019_data["long"], loc_2019_data["lat"], label='2019', alpha=0.5)
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.legend()
# plt.savefig("test.png")



# import numpy as np
# import matplotlib.pyplot as plt
# inds = np.random.choice(np.arange(len(prescriptions_ds["practice_id"])), size=20, replace=False)
# colors = plt.get_cmap('tab20').colors
# prac_lons = prescriptions_ds["longitude"].values[inds]
# prac_lats = prescriptions_ds["latitude"].values[inds]
# lad_lons = prescriptions_ds["imd_longitude"].values[:, inds]
# lad_lats = prescriptions_ds["imd_latitude"].values[:, inds]
# plt.figure(figsize=(10,10))
# for i, ind in enumerate(inds):
#     plt.scatter(prac_lons[i], prac_lats[i], color=colors[i], label=f'Practice {ind}', marker='o', alpha=0.3)
#     plt.scatter(lad_lons[:, i], lad_lats[:, i], color=colors[i], marker='x', alpha=0.3)
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.legend()
# plt.savefig("test.png")