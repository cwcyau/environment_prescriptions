'''
Download AQREAN particulate data files as specified in the JSON configuration file from:
https://catalogue.ceda.ac.uk/uuid/7d44ef2a9e9346e79863f53193db189e
'''
import requests, json
from funcs import download_file
from pathlib import Path

files_json_path = "data/aqrean/file_downloads.json"
local_root = Path("data/aqrean/")

SESSION = requests.Session()

with open(files_json_path, "r") as f:
    files_info = json.load(f)

for file_info in files_info['items']:
    url = file_info['download'].split('?')[0]
    local_path = local_root / file_info['name']

    file_exists = local_path.exists()
    if file_exists:
        file_complete = local_path.stat().st_size >= file_info['size']
        file_done = file_exists and file_complete
    else:
        file_done = False

    if not file_done:
        print(f"Downloading {file_info['name']}...")
        download_file(url, SESSION, out_dir=local_root)
    else:
        print(f"File {file_info['name']} already exists, skipping download.")
