import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from funcs import var_name_to_plot_name

RECALCULATE_CORRS = False
threshold = 0.5

if RECALCULATE_CORRS:
    file_path = "data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc"
    ds = xr.open_dataset(file_path)
    var_names = np.sort([var for var in ds.data_vars if var.endswith("_values")])

    correlations = []
    corrs = np.zeros((len(var_names), len(var_names)))
    for i, v1 in enumerate(var_names):
        for j, v2 in enumerate(var_names):
            if i != j:
                corr = xr.corr(ds[v1], ds[v2], dim="date").mean(dim="practice_id").values
            else:
                corr = 1.0
            corrs[i, j] = corr
            if np.abs(corr) > threshold and i != j and f"{v2}+{v1}" not in correlations:
                correlations.append(f"{v1}+{v2}")

    to_save = dict(var_names=var_names, corrs=corrs)
    np.save("outputs/pairwise_correlation.npy", to_save)
else:
    data = np.load("outputs/pairwise_correlation.npy", allow_pickle=True).item()
    corrs = data['corrs']
    var_names = data['var_names']
    plot_names = [var_name_to_plot_name(var) for var in var_names]
    correlations = []
    for i, v1 in enumerate(var_names):
        for j, v2 in enumerate(var_names):
            if np.abs(corrs[i, j]) > threshold and i != j and f"{v2}+{v1}" not in correlations:
                correlations.append(f"{v1}+{v2}")

# report large correlations and what fraction of pairs were above threshold
num_pairs = len(var_names) * (len(var_names) - 1) / 2
np.save("outputs/pairwise_high_correlations.npy", correlations)
print(f"{len(correlations)} large correlations found out of {num_pairs} pairs:")
print(f"correlations > {threshold} = [")
for i in correlations:
    print(f'"{i}",')
print("]")

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corrs, cmap="bwr", vmin=-1, vmax=1)
for i, v1 in enumerate(var_names):
    for j, v2 in enumerate(var_names):
        if np.abs(corrs[i, j]) > threshold and i != j:
            ax.add_patch(plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8,
                                       fill=False, edgecolor='yellow', lw=2))
fig.colorbar(cax, label="Correlation coefficient")
ax.set_xticks(range(len(var_names)))
ax.set_yticks(range(len(var_names)))
ax.set_xticklabels(plot_names, rotation=90)
ax.set_yticklabels(plot_names)
plt.tight_layout()
plt.savefig("outputs/pairwise_correlation.png", bbox_inches='tight', dpi=600)