import bambi as bmb
import pandas as pd
import numpy as np


n = 4000
tune = 1000
draws = 1000
chains = 1
cores = 1

if __name__ == "__main__":
    df = pd.DataFrame({'y': range(n),
                       'x': np.arange(n) + np.random.normal(0, 2, n),
                       'd': np.linspace(-1, 1, n),
                       'g': np.random.randint(0, 10, n)})
    for practice_correction in [2, 3]:
        formula = "y ~ x"
        if practice_correction == 1:
            formula += " + (1 | g)"  # intercept
        elif practice_correction == 2:
            formula += " + (1 | g) + (0 + d | g)"  # intercept + slope, uncorrelated
        elif practice_correction == 3:
            formula += " + (d | g)"  # intercept + slope, correlated
        elif practice_correction != 0:
            raise ValueError("practice_correction must be 0, 1, 2, or 3")
        print(f"Formula: {formula}")
        model = bmb.Model(formula, df)
        idata = model.fit(draws=draws,
                            tune=tune,
                            chains=chains,
                            cores=cores,
                            progressbar=False)
