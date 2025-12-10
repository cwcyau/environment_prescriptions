import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------
# USER SETTINGS
# -------------------------
IDATA_PATH = "outputs/bayes_standard/02_03_0501/bayesian_model_idata.nc"
OUTPUT_DIR = "outputs/bayes_standard/02_03_0501/diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -------------------------

print("Loading idata…")
idata = az.from_netcdf(IDATA_PATH)

summary = az.summary(idata)
summary["_varname"] = summary.index

# Identify problematic parameters
bad_rhat = summary[summary["r_hat"] > 1.01]
bad_ess_bulk = summary[summary["ess_bulk"] < 300]
bad_ess_tail = summary[summary["ess_tail"] < 300]

problem_vars = set(bad_rhat.index) | set(bad_ess_bulk.index) | set(bad_ess_tail.index)

print("\n=== PROBLEMATIC PARAMETERS DETECTED ===")
if len(problem_vars) == 0:
    print("No R-hat or ESS problems detected!")
else:
    for v in problem_vars:
        print(f" - {v}")

    print("\nPlots will be generated for these parameters.")

# ---------------------------------------------------------
# Helper: Safe plot generator (avoids variable-not-found errors)
# ---------------------------------------------------------
def safe_plot(func, varname, plot_name):
    try:
        func(varname)
        plt.title(f"{plot_name}: {varname}")
        if plot_name == "Rank plot":
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{varname.replace('|', '-')}_{plot_name.replace(' ', '_').replace('|', '-')}.png"))
        plt.close()
    except Exception as e:
        print(f"   Skipped {plot_name} for {varname}: {e}")

# ---------------------------------------------------------
# Generate plots for each problematic variable
# ---------------------------------------------------------
for v in problem_vars:
    print(f"\nGenerating diagnostics for: {v}")

    # -------- Traceplot --------
    def trace(v):
        az.plot_trace(idata, var_names=[v])
    safe_plot(trace, v, "Traceplot")

    # -------- Rank plot --------
    def rank(v):
        az.plot_rank(idata, var_names=[v])
    safe_plot(rank, v, "Rank plot")

    # -------- Autocorrelation --------
    def acorr(v):
        az.plot_autocorr(idata, var_names=[v])
    safe_plot(acorr, v, "Autocorr")

# ---------------------------------------------------------
# Summary Interpretation
# ---------------------------------------------------------
print("\n=== INTERPRETATION & RECOMMENDATIONS ===")

def print_recommendations():
    if len(bad_rhat) > 0:
        print("\nParameters with **R-hat > 1.01** (non-convergence):")
        print(bad_rhat["r_hat"])
        print("\n**Recommended actions**:")
        print("   • Increase `tune` dramatically (e.g. 4000–8000).")
        print("   • Increase `target_accept` (0.9–0.98).")
        print("   • Re-check parameterization (centering, scaling).")
        print("   • Check if predictor is too collinear or unidentifiable.")

    if len(bad_ess_bulk) > 0:
        print("\nParameters with **ESS_bulk < 300** (poor exploration):")
        print(bad_ess_bulk["ess_bulk"])
        print("\n**Recommended actions**:")
        print("   • Increase number of draws.")
        print("   • Increase `tune`.")
        print("   • Reparameterize random effects (non-centered).")

    if len(bad_ess_tail) > 0:
        print("\nParameters with **ESS_tail < 300** (poor tail estimation):")
        print(bad_ess_tail["ess_tail"])
        print("\n**Recommended actions**:")
        print("   • Same as ESS_bulk, plus:")
        print("   • Consider stronger priors for problematic parameters.")

print_recommendations()

print("\nAll diagnostic plots saved to:", OUTPUT_DIR)
print("\nDone.")
