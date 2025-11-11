from funcs import compare_individual_analyses, compare_bayesian_analyses
import os

# plot individual analyses
compare_bayesian_analyses("outputs/inputs_c-practice/outputs_standardised/interactions/")
compare_individual_analyses("outputs/inputs_c-practice/outputs_standardised/")


# PLOT ALL ANALYSES ===============================================================================
# # parameters
# codes = ["02_03_0501", "02", "03", "0501"]
# input_types = ["inputs_c-practice", "inputs_c-practice_deseasonalised",
#                "inputs_z-global", "inputs_z-global_deseasonalised"]
# outputs_types = ["outputs_standardised", "outputs_raw"]
# suffix_types = ["interactions/"]
# mixed_file = "mixed_effects_values_results.txt"
# bayes_file = "bayesian_model_idata.nc"

# # find which configurations have results for all prescription codes
# mixed_folders, bayes_folders = [], []
# for input_type in input_types:
#     for output_type in outputs_types:
#         for suffix_type in suffix_types:
#             results_folder = f"outputs/{input_type}/{output_type}/{suffix_type}"
#             mixed_path = f"{results_folder}{mixed_file}"
#             bayes_path = f"{results_folder}{bayes_file}"
#             folders_mixed = [os.path.exists(f"{results_folder}{c}/{mixed_file}")
#                              for c in codes]
#             folders_bayes = [os.path.exists(f"{results_folder}{c}/{bayes_file}")
#                              for c in codes]
#             if all(folders_mixed):
#                 mixed_folders.append(results_folder)
#             if all(folders_bayes):
#                 bayes_folders.append(results_folder)

# # compare analyses for each configuration found
# for folder in mixed_folders:
#     compare_individual_analyses(folder)
# for folder in bayes_folders:
#     compare_bayesian_analyses(folder)
