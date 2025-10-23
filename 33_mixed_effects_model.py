import os
import xarray as xr
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm

nc_file_path = "prescriptions_0501_02_03_2020-09_2025-08.nc"

# load dataset
ds = xr.load_dataset(nc_file_path)

# prepare dataframe
df_list = []
for practice in tqdm(ds['row_id'].values, desc="Preparing data"):
    practice_data = ds.sel(row_id=practice).to_dataframe().reset_index()
    practice_data['practice_id'] = practice
    practice_data['flood'] = ds['flood'].sel(row_id=practice).values
    practice_data['flood_geo'] = ds['flood_geo'].sel(row_id=practice).values
    practice_data['drought'] = ds['drought'].sel(row_id=practice).values
    df_list.append(practice_data)

df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=['items'])

# convert flags to integers
df[['flood', 'flood_geo', 'drought']] = df[['flood', 'flood_geo', 'drought']].fillna(0).astype(int)

# remove month-of-year effects globally
df['items_adj'] = df['items'] - df.groupby(df['date'].dt.month)['items'].transform('mean')

# optionally scale and clip extreme values
sd_global = df['items_adj'].std()
df['items_adj'] = df['items_adj'] / (sd_global + 1e-9)
df['items_adj'] = df['items_adj'].clip(-5, 5)

# fit mixed-effects model
model_formula = 'items_adj ~ flood + flood_geo + drought - 1'
md = smf.mixedlm(model_formula, df, groups=df['practice_id'])
mdf = md.fit(method='lbfgs', reml=True)

# print and save results
summary = mdf.summary()
print(summary)

save_path = os.path.join("outputs", nc_file_path.replace(".nc", ""), "mixed_effects_result.txt")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    f.write(str(summary))
