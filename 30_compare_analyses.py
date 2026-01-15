import os
import arviz as az
import numpy as np
from funcs import compare_mixed_models, compare_bayesian_models, generate_bayesian_diagnostics, status, plot_months_bayes


root = "outputs/bayes_standard/"
folders = ["02_03_0501", "02", "03", "0501"]
results_folders = [os.path.join(root, folder) for folder in folders]

# for folder in folders:
#     results_folder = os.path.join(root, folder)
#     status(f"Loading data from {results_folder}...")
#     idata = az.from_netcdf(os.path.join(results_folder, "bayesian_model_idata.nc"))

#     status("Plotting diagnostics...")
#     generate_bayesian_diagnostics(idata, results_folder)

status("Comparing model results...")
# compare_bayesian_models(root)
plot_months_bayes(results_folders, root)