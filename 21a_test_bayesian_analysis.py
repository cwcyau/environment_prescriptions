import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from funcs import run_bayesian_model, compare_bayesian_models

PRES_CODES = ['02_03_0501', '02', '03', '0501']
PRES_LABELS = ['All', 'Cardiovascular', 'Respiratory', 'Antibiotics']
PRES_COLOURS = ['black', 'red', 'blue', 'orange']
# ordered north->south, east->west with "South East" last (reference category)
REGION_NAMES = ["North East", "North West", "Yorkshire and The Humber",
                "East Midlands", "West Midlands", "East of England",
                "London", "South West", "South East"]
PRACTICE_SIZES = ["small", "large"]
# ordered months with "September" last (reference category)
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# setup toy dataset
np.random.seed(42)
n_practices = 5
n_months = 12
n_obs = n_practices * n_months
practice_ids = [f"P{i+1}" for i in range(n_practices)]
date = pd.date_range(start="2020-01-01", periods=n_months, freq="M")
months = np.tile(np.arange(1, n_months + 1), n_practices)
date_codes = np.tile(np.arange(n_months), n_practices)
regions = np.random.choice(REGION_NAMES, n_practices)
sizes = np.random.choice(PRACTICE_SIZES, n_practices)
region_map = dict(zip(practice_ids, regions))
size_map = dict(zip(practice_ids, sizes))
X1 = np.random.normal(0, 1, n_obs)
X2 = np.random.normal(5, 2, n_obs)
items = np.exp(1 + 0.3*X1 - 0.2*X2 + np.random.normal(0, 0.2, n_obs)).astype(int)
df = pd.DataFrame({
    "practice_id": np.repeat(practice_ids, n_months),
    "date": np.tile(date, n_practices),
    "date_code": date_codes,
    "month": months,
    "region": [region_map[p] for p in np.repeat(practice_ids, n_months)],
    "practice_size": [size_map[p] for p in np.repeat(practice_ids, n_months)],
    "items": items,
    "X1": X1,
    "X2": X2
})
ds = df.set_index(["practice_id", "date"]).to_xarray()
results_folder = "temp/02/"
os.makedirs(results_folder, exist_ok=True)

# run model
model, idata = run_bayesian_model(
    ds=ds,
    raw_vars=["X1", "X2"],
    results_folder=results_folder,
    lag=1,
    almon_order=2,
    individual_priors=True,
    deseasonalise_output=True,
    practice_correction=1,
    min_practice_obs=3,
    likelihood="normal",
    draws=100,
    tune=100,
    chains=1,
    cores=1
)

# plot
compare_bayesian_models("/".join(results_folder.split("/")[:-2]) + "/")
