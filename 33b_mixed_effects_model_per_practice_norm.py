import os
import numpy as np
import xarray as xr
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm

def per_practice_month_normalise(x, min_sd=1e-2):
    """
    x: 1D xarray.DataArray with dimension 'date'
    Subtract per-month mean (seasonal adjustment) and scale by SD
    Returns numpy array of same length as x
    """
    month_numbers = x['date'].dt.month
    x_adj = x.groupby(month_numbers).map(lambda m: m - m.mean(dim='date'))
    sd = x_adj.std(dim='date')
    sd = xr.where(sd < min_sd, 1.0, sd)  # avoid tiny SD
    x_adj = x_adj / sd
    return x_adj.values  # return flat array

nc_file_path = "prescriptions_0501_02_03_2020-09_2025-08.nc"

# load dataset
ds = xr.load_dataset(nc_file_path)

# prepare dataframe per practice
df_list = []
for practice in tqdm(ds['row_id'].values, desc="Preparing data"):
    practice_data = ds.sel(row_id=practice).to_dataframe().reset_index()
    practice_data['practice_id'] = practice
    practice_data['flood'] = ds['flood'].sel(row_id=practice).values
    practice_data['flood_geo'] = ds['flood_geo'].sel(row_id=practice).values
    practice_data['drought'] = ds['drought'].sel(row_id=practice).values

    # seasonal adjustment per practice
    practice_data['items_adj'] = per_practice_month_normalise(ds['items'].sel(row_id=practice))

    df_list.append(practice_data)

# combine all practices
df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=['items', 'items_adj'])

# convert flags to integers
df[['flood', 'flood_geo', 'drought']] = df[['flood', 'flood_geo', 'drought']].fillna(0).astype(int)

# clip extreme values
df['items_adj'] = df['items_adj'].clip(-5, 5)

# fit mixed-effects model
model_formula = 'items_adj ~ flood + flood_geo + drought'
md = smf.mixedlm(model_formula, df, groups=df['practice_id'])
mdf = md.fit(method='lbfgs', reml=True)

# print and save results
summary = mdf.summary()
print(summary)
save_path = os.path.join("outputs", nc_file_path.replace(".nc", ""), "mixed_effects_result.txt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    f.write(str(summary))
