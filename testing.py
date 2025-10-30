import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def remove_seasonal_effects(datetimes, values):
    datetimes = pd.to_datetime(datetimes)
    month_nums = datetimes.month
    medians = np.full(12, np.nan)
    mads = np.full(12, np.nan)

    for m in range(1, 13):
        mask = (month_nums == m)
        if not np.any(mask):
            continue
        v = values[mask]
        med = np.nanmedian(v)
        medians[m - 1] = med
        mads[m - 1] = np.nanmedian(np.abs(v - med)) * 1.4826

    # Vectorized z-score calculation
    m = month_nums - 1
    monthly_anomalies = (values - medians[m]) / (mads[m] + 1e-9)
    return monthly_anomalies

datetimes = pd.date_range("2010-05-12", "2020-08-27", freq="D")
values_raw = 20 + np.cos(2 * np.pi * (datetimes.dayofyear.values / 365.25)) * 2
values_clean = values_raw + np.random.normal(0, 1, len(datetimes))

# add individual missing readings
values = values_clean.copy()
gap_inds = np.random.choice(len(values), size=len(datetimes)//10, replace=False)
values[gap_inds] = np.nan

# add random longer gaps
gap_inds = np.random.choice(len(values), size=len(datetimes)//100, replace=False)
for i in gap_inds:
    s = np.random.randint(16, 60)
    values[i:i+s] = np.nan

corrected_values = remove_seasonal_effects(datetimes, values)
corrected_clean_values = remove_seasonal_effects(datetimes, values_clean)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].plot(datetimes, values_raw, 'r', label='Clean Values', alpha=0.5)
axes[0].plot(datetimes, values, 'k', label='Original Values', alpha=0.5)
axes[0].set_title('Original Values')
axes[0].legend()

axes[1].plot(datetimes, corrected_values, 'r', label='Corrected Values', alpha=0.5)
axes[1].plot(datetimes, corrected_clean_values, 'b', label='Corrected Clean Values', alpha=0.5)
axes[1].set_title('Corrected Values')
axes[1].legend()

plt.tight_layout()
plt.savefig('testing.png')