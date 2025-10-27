import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

# parameters
nc_file_path = "prescriptions_0501_2020-09_2025-08.nc"
sample_size = 30

# load dataset
ds = xr.load_dataset(nc_file_path)
practices = ds['row_id'].values
sample_practices = practices[np.random.choice(len(practices),
                                              size=sample_size,
                                              replace=False)]

# plot prescriptions time series with flags for a sample of practices
for practice in tqdm(sample_practices, desc="Generating plots"):
    practice_data = ds.sel(row_id=practice)
    if practice_data["items"].count() == 0:
        continue

    plt.figure(figsize=(12, 6))
    plt.plot(practice_data["date"].values,
             practice_data["items"].values,
             "ko-", markersize=10, alpha=0.2,
             label="Prescriptions")

    plot_args = (("drought","red","v"),
                 ("flood","blue","v"),
                 ("flood_geo","green","^"))
    for kind, color, marker in plot_args:
        if kind not in practice_data:
            continue
        flagged_dates = practice_data["date"].where(practice_data[kind]==1,
                                                    drop=True).values
        flagged_values = practice_data["items"].sel(date=flagged_dates).values
        plt.scatter(flagged_dates, flagged_values,
                    c=color, s=60, marker=marker, alpha=0.7,
                    label=f"{kind.title()} Flag", zorder=5)

    plt.title(f"Prescriptions Time Series for Practice {practice}")
    plt.xlabel("Date")
    plt.ylabel("Number of Prescriptions")
    plt.legend()
    plt.grid()

    # save
    save_path = "outputs/" + nc_file_path.replace(".nc", "") + "/" + practice + ".png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
