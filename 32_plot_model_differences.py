import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from funcs import var_name_to_plot_name

predictors = [
    "flood",
    "hydro_rain_values",
    "met_rain_values",
    "met_tmax_values",
    "met_tmin_values",
    "aqrean_carbon_monoxide_values",
    "aqrean_daqi_overall_values",
    "aqrean_nitrogen_monoxide_values",
    "aqrean_nitrogen_dioxide_values",
    "aqrean_daqi_nitrogen_dioxide_values",
    "aqrean_nox_expressed_as_nitrogen_dioxide_values",
    "aqrean_ozone_values",
    "aqrean_daqi_ozone_values",
    "aqrean_pm2p5_values",
    "aqrean_daqi_pm2p5_values",
    "aqrean_pm10_values",
    "aqrean_daqi_pm10_values",
    "aqrean_sulfur_dioxide_values",
    "aqrean_daqi_sulfur_dioxide_values"
]
predictor_labels = [var_name_to_plot_name(var) for var in predictors]
order = np.flip(np.argsort(predictor_labels))
predictors = [predictors[i] for i in order]
predictor_labels = [predictor_labels[i] for i in order]

prescription_codes = ["02_03_0501", "02", "03", "0501"]
names = ["All", "Cardiovascular", "Respiratory", "Antibiotics"]
colours = ["black", "red", "blue", "orange"]

fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
y_points = np.arange(len(predictors))
for ax in axes:
    ax.grid()
    ax.axvline(0, color='black')

bayes_minus_mixed = {}
for i, code in enumerate(prescription_codes):
    results_bayes = f"outputs/bayes_standard/{code}/bayesian_model_summary_95pcCIs.csv"
    results_mixed_flags = f"outputs/mixed_effects/{code}/mixed_effects_flag_results.csv"
    results_mixed = f"outputs/mixed_effects/{code}/mixed_effects_values_results.csv"

    df_bayes = pd.read_csv(results_bayes)
    df_mixed_flags = pd.read_csv(results_mixed_flags)
    df_mixed = pd.read_csv(results_mixed)

    bayes_minus_mixed[code] = {}
    for j, predictor in enumerate(predictors):
        if predictor == "flood":
            coef_mixed = df_mixed_flags.loc[df_mixed_flags['name'] == "flood_vs_not", 'coef'].values
            coef_bayes = df_bayes.loc[df_bayes['name'] == "C(flood)[1.0]", 'mean'].values
        else:
            coef_mixed = df_mixed.loc[df_mixed['name'] == predictor, 'coef'].values
            coef_bayes = df_bayes.loc[df_bayes['name'] == predictor, 'mean'].values
        pred_diff = coef_bayes - coef_mixed
        pred_diff_abs = np.abs(pred_diff)
        bayes_minus_mixed[code][predictor] = pred_diff

        axes[0].scatter(pred_diff, y_points[j], c=colours[i],
                    label=names[i] if j == 0 else None,
                    alpha=0.7)
        axes[1].scatter(pred_diff_abs, y_points[j], c=colours[i],
                    label=names[i] if j == 0 else None,
                    alpha=0.7)

axes[0].set_yticks(y_points)
axes[0].set_yticklabels(predictor_labels)
axes[0].legend(frameon=False)
plt.tight_layout()
plt.savefig("outputs/inter_model_comparison.png")
