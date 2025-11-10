import bambi as bmb
import pandas as pd
import numpy as np
from time import perf_counter
import os, json
import arviz as az

n_dates = 400
n_groups = 300
tune = 1000
draws = 1000
chains = 2
cores = 2

if __name__ == "__main__":
    # generate data
    y = np.random.normal(0, 4, (n_dates, n_groups))
    x = y + np.random.normal(0, 2, (n_dates, n_groups))
    g = np.arange(n_groups)
    d = np.linspace(-1, 1, n_dates)
    m = np.tile(np.arange(1, 13), int(np.ceil(n_dates / 12)))[:n_dates]
    x2 = np.zeros((n_dates, n_groups))
    for i in range(n_groups):
        y[:, i] += np.random.normal(0, 20, n_dates)  # group intercept
        y[:, i] *= np.linspace(np.random.normal(0, 1), np.random.normal(0, 1), n_dates)  # group slope
        x2[:, i] = np.sin(np.linspace(0, n_dates/12 * np.pi, n_dates)) + np.random.normal(0, 0.5, n_dates)

    # flatten and corrupt with NaNs
    nan_inds = np.random.choice(n_dates * n_groups, size=int(0.2 * n_dates * n_groups), replace=False)
    y = y.flatten()
    y[nan_inds] = np.nan
    x = x.flatten()
    x[nan_inds] = np.nan
    g = np.repeat(g, n_dates)
    m = np.tile(m, n_groups)
    d = np.tile(d, n_groups)
    x2 = x2.flatten()

    df = pd.DataFrame({'y': y, 'x': x, 'x2': x2, 'd': d, 'g': g, 'm': m})
    df = df.dropna().reset_index(drop=True)
    times = []
    formulae = ["y ~ x",
                "y ~ x + x2",
                "y ~ x + x2 + x*x2",
                "y ~ x + C(m)",
                "y ~ x + (1 | g) + (0 + d | g)",
                "y ~ x + C(m) + (1 | g) + (0 + d | g)",
                "y ~ x + (d | g)"]
    formulae = ["y ~ x",]
    for formula in formulae:
        print(f"Formula: {formula}")
        model = bmb.Model(formula, df)
        z0 = perf_counter()
        idata = model.fit(draws=draws,
                            tune=tune,
                            chains=chains,
                            cores=cores,
                            progressbar=False)
        z1 = perf_counter()
        times.append(z1 - z0)

    for t, f in zip(times, formulae):
        print(f"{t:.2f} seconds for formula: {f}")

# 19.56 seconds for formula: y ~ x
# 19.60 seconds for formula: y ~ x + x2
# 21.84 seconds for formula: y ~ x + x2 + x*x2
# 45.08 seconds for formula: y ~ x + C(m)
# 89.57 seconds for formula: y ~ x + (1 | g) + (0 + d | g)
# 115.06 seconds for formula: y ~ x + C(m) + (1 | g) + (0 + d | g)
# 132.60 seconds for formula: y ~ x + (d | g)