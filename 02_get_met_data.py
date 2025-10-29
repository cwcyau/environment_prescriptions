import os
import re
import xarray as xr
import numpy as np

# parameters
folder_path = "data/met_office_stations/"  # met office data path
output_file = "data/met_office_stations.nc"  # output nc file path

def clean_value(v):
    """Extract numeric part, remove *, #, --- and handle non-numeric suffixes."""
    v = v.replace('*', '').replace('#', '')
    if v in ('---', '', 'NaN'):
        return 'nan'
    
    # extract leading numeric part
    match = re.match(r'^([+-]?\d+(?:\.\d*)?)', v)  # this removes annoying comments at the end of lines
    if match:
        return match.group(1)
    else:
        return 'nan'

def parse_txt_file(file_path):
    """
    Parse a single UK Met Office-style .txt file and return:
    - data: dict of column data
    - name: lowercase station name
    - lon, lat: floats
    """
    data = {}
    name = None
    lon = None
    lat = None

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"File {file_path} too short to parse.")

    name = file_path.split('/')[-1].replace('data.txt', '')  # station name

    # extract longitude and latitude
    # example line: "Location: 224100E 252100N, Lat 52.139 Lon -4.570, 133 metres amsl"
    # find the line containing "Lat" and "Lon"
    for line in lines[:10]:
        if "Lat" in line and "Lon" in line:
            loc_line = line
            break
    lat_match = re.search(r'Lat\s+([0-9.+-]+)', loc_line)
    lon_match = re.search(r'Lon\s+([0-9.+-]+)', loc_line)
    if lat_match and lon_match:
        lat = float(lat_match.group(1))
        lon = float(lon_match.group(1))
    else:
        lat = lon = float('nan')  # gracefully handle missing info

    # extract data columns
    header = None
    for line in lines[2:]:
        if line.startswith("yyyy"):
            header = line.split()
            for col in header:
                data[col.strip()] = []
            continue

        if header is None or not (line.startswith("19") or line.startswith("20")):
            continue  # skip until header found
        
        # clean all the mess the MET office left in their files
        values = [clean_value(v) for v in line.split()]
        # sometimes empty sun values are "" instead of "---" *facepalm*
        if len(values) < len(header):
            values += ['nan'] * (len(header) - len(values))
        # append to data
        for col, val in zip(header, values):
            try:
                if val.lower() == 'nan':
                    val = float('nan')
                elif '.' in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                val = float('nan')
            data[col].append(val)
        
    # create numpy datetimes from yyyy and mm
    if 'yyyy' in data and 'mm' in data:
        years = data['yyyy']
        months = data['mm']
        datetimes = [np.datetime64(f"{y:04d}-{m:02d}-01") for y, m in zip(years, months)]
        data['date'] = datetimes

    return data, name, lon, lat

# get the txt files in the folder
txt_files = [os.path.join(folder_path, f)
             for f in os.listdir(folder_path) if f.endswith(".txt")]

# containers
station_ids = []
lats = []
lons = []
all_data = []
all_dates = set()

# parse each file
for f in txt_files:
    data, name, lon, lat = parse_txt_file(f)
    station_ids.append(name)
    lats.append(lat)
    lons.append(lon)
    all_data.append(data)
    all_dates.update(data['date'])

# sort all dates and create date index
all_dates = sorted(all_dates)
n_dates = len(all_dates)
n_stations = len(station_ids)

# create mapping from date to index
date_to_idx = {d: i for i, d in enumerate(all_dates)}

# prepare data arrays
variables = [k for k in all_data[0].keys() if k not in ('yyyy', 'mm', 'date')]
data_arrays = {var: np.full((n_dates, n_stations), np.nan, dtype=np.float32)
               for var in variables}

# fill arrays
for i, station_data in enumerate(all_data):
    for j, d in enumerate(station_data['date']):
        date_idx = date_to_idx[d]
        for var in variables:
            data_arrays[var][date_idx, i] = station_data[var][j]

# build xarray dataset
coords = {
    "date": all_dates,
    "station_id": station_ids,
    "latitude": ("station_id", lats),
    "longitude": ("station_id", lons)
}

ds = xr.Dataset({var: (("date", "station_id"), data_arrays[var])
                 for var in variables},
                coords=coords)

# save to netcdf
ds.to_netcdf(output_file)
print(f"Saved Met Office dataset to {output_file}")