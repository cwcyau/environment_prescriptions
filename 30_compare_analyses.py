import os
import arviz as az
import numpy as np
from funcs import compare_mixed_models, compare_bayesian_models, generate_bayesian_diagnostics, status, plot_months_bayes


root = "outputs/bayes_lagged_2/"
folders = ["02_03_0501", "02", "03", "0501"]
results_folders = [os.path.join(root, folder) for folder in folders]

status("Comparing model results...")
compare_bayesian_models(root, hide_vars=["imd_centile_values"], mean_name="eti_mean_pct",
                        lower_name="eti_2.5pc_pct", upper_name="eti_97.5pc_pct")