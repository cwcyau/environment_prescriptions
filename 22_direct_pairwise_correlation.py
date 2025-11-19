import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

RECALCULATE_CORRS = True
threshold = 0.6

if RECALCULATE_CORRS:
    file_path = "data/prescriptions_02_03_0501_2010-08_2025-08_with_flags.nc"
    ds = xr.open_dataset(file_path)
    var_names = np.sort([var for var in ds.data_vars if var.endswith("_values")])
    plot_names = [" ".join(var.split("_values")[0].split("_")) for var in var_names]

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

    to_save = dict(var_names=var_names, plot_names=plot_names, corrs=corrs)
    np.save("outputs/pairwise_correlation.npy", to_save)
else:
    data = np.load("outputs/pairwise_correlation.npy", allow_pickle=True).item()
    corrs = data['corrs']
    var_names = data['var_names']
    plot_names = data['plot_names']
    correlations = []
    for i, v1 in enumerate(var_names):
        for j, v2 in enumerate(var_names):
            if np.abs(corrs[i, j]) > threshold and i != j and f"{v2}+{v1}" not in correlations:
                correlations.append(f"{v1}+{v2}")

np.save("outputs/pairwise_high_correlations.npy", correlations)
print(f"{len(correlations)} large correlations found:")
print(f"correlations > {threshold} = [")
for i in correlations:
    print(f'"{i}",')
print("]")

fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corrs, cmap="bwr", vmin=-1, vmax=1)
for i, v1 in enumerate(var_names):
    for j, v2 in enumerate(var_names):
        if np.abs(corrs[i, j]) > threshold and i != j:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       fill=False, edgecolor='yellow', lw=2))
fig.colorbar(cax)
ax.set_xticks(range(len(var_names)))
ax.set_yticks(range(len(var_names)))
ax.set_xticklabels(plot_names, rotation=90)
ax.set_yticklabels(plot_names)
plt.title("Absolute Pairwise Correlation of Prescription Values")
plt.tight_layout()
plt.savefig("outputs/pairwise_correlation.png")