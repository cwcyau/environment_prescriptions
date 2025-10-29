import time
import numpy as np
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from zipfile import ZipFile

# CONFIG
BASE_INDEX = "https://digital.nhs.uk/data-and-information/publications/statistical/practice-level-prescribing-data"
OUT_DIR = Path("data/prescriptions_pre20200101")
CSV_DIR = OUT_DIR / "csvs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "epd-zip-scraper/1.0 (you@example.com)"})
SLEEP_BETWEEN_REQUESTS = 0.2
TIMEOUT = 30

def get_soup(url):
    r = SESSION.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.content, "html.parser")

def discover_month_pages():
    """Scrape main index for links to monthly subpages."""
    soup = get_soup(BASE_INDEX)
    month_urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/practice-level-prescribing-data/" in href and "Chemical-level" not in href:
            month_urls.add(urljoin(BASE_INDEX, href))
    return sorted(month_urls)

def discover_zip_links(month_page_url):
    """Return list of ZIP file URLs from a monthly page."""
    soup = get_soup(month_page_url)
    zip_urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".zip"):
            zip_urls.append(urljoin(month_page_url, href))
    return sorted(set(zip_urls))

def download_file(url, out_dir=OUT_DIR, overwrite=False):
    """Download file streaming to disk. Skip if already exists."""
    fname = url.split("/")[-1]
    out_path = out_dir / fname

    if out_path.exists() and not overwrite:
        # optionally check file size to avoid partial downloads
        if out_path.stat().st_size > 1_000_000:  # >1MB sanity check
            print(f"Skipping already downloaded ZIP: {fname}")
            return out_path
        else:
            print(f"Re-downloading incomplete ZIP: {fname}")

    with SESSION.get(url, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    fh.write(chunk)
    return out_path

def extract_pdpi_csv(zip_path, csv_dir=CSV_DIR):
    """Extract the PDPI BNFT CSV from a ZIP file, skip if already exists."""
    with ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "PDPI" in name.upper() and name.lower().endswith(".csv"):
                out_path = csv_dir / Path(name).name
                if out_path.exists() and out_path.stat().st_size > 100_000:  # >100KB sanity check
                    print(f"Skipping already extracted CSV: {out_path.name}")
                    return out_path

                zf.extract(name, csv_dir)
                extracted_path = csv_dir / name
                if extracted_path != out_path:
                    extracted_path.rename(out_path)
                return out_path
    return None

if __name__ == "__main__":
    # find the pages for each month
    month_pages = discover_month_pages()
    print(f"Discovered {len(month_pages)} monthly pages (examples):")
    sample = np.random.choice(month_pages, size=min(6, len(month_pages)), replace=False)
    for z in sample:
        print("  ", z)

    # 2) Discover ZIP links
    all_zip_urls = []
    for page in tqdm(month_pages, desc="Discovering ZIPs"):
        try:
            zips = discover_zip_links(page)
            if zips:
                all_zip_urls.extend(zips)
        except Exception as e:
            print("Failed to parse", page, e)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    all_zip_urls = sorted(set(all_zip_urls))
    print(f"Found {len(all_zip_urls)} ZIP URLs (examples):")
    sample = np.random.choice(all_zip_urls, size=min(6, len(all_zip_urls)), replace=False)
    for z in sample:
        print("  ", z)

    # 3) Download ZIPs and extract PDPI BNFT CSV
    for url in tqdm(all_zip_urls, desc="Downloading and extracting"):
        try:
            zip_path = download_file(url)
            csv_path = extract_pdpi_csv(zip_path)
            if csv_path:
                print(f"Extracted {csv_path.name}")
            else:
                print(f"No PDPI CSV found in {zip_path.name}")
        except Exception as e:
            print("Failed processing", url, e)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
