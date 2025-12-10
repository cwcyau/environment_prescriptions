import arviz as az

input_path = "outputs/bayes_standard/02_03_0501/bayesian_model_idata.nc"
output_path = "outputs/bayes_standard/02_03_0501/bayesian_model_idata_noppc.nc"

# load existing data
idata = az.from_netcdf(input_path)

# groups to drop
drop_groups = [
    "posterior_predictive",
    "prior_predictive",
    "predictions",
    "log_likelihood",
    "sample_stats_prior",
]

for group in drop_groups:
    if group in idata.groups():
        print(f"Dropping {group}")
        del idata[group]

# save cleaned idata
az.to_netcdf(idata, output_path)

print("Saved cleaned file to:", output_path)