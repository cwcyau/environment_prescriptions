# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from scipy.stats import norm, multivariate_normal, gaussian_kde

# # ============================================================
# # 1. Generate data with correlated predictors
# # ============================================================
# np.random.seed(42)
# n = 40

# # correlated predictors
# mean_x = [0, 0]
# cov_x = [[1.0, 0.8],
#          [0.8, 1.0]]
# X = np.random.multivariate_normal(mean_x, cov_x, size=n)
# x1, x2 = X[:, 0], X[:, 1]

# true_alpha = 1.0
# true_beta = np.array([2.0, -1.0])
# true_sigma = 1.0

# y = true_alpha + true_beta[0]*x1 + true_beta[1]*x2 + np.random.normal(0, true_sigma, size=n)

# # ============================================================
# # 2. Prior
# # ============================================================
# prior_mean = np.array([0.0, 0.0])
# prior_cov = np.array([[1.5, 1.0],
#                       [1.0, 1.5]])

# # ============================================================
# # 3. Log posterior
# # ============================================================
# def log_posterior(alpha, beta, sigma):
#     if sigma <= 0: return -np.inf
#     mu = alpha + beta[0]*x1 + beta[1]*x2
#     ll = np.sum(norm.logpdf(y, mu, sigma))
#     lp = norm.logpdf(alpha, 0, 5) + multivariate_normal.logpdf(beta, prior_mean, prior_cov) + norm.logpdf(sigma, 0, 2)
#     return ll + lp

# # ============================================================
# # 4. MCMC sampling
# # ============================================================
# n_steps = 1000

# alpha_s = np.zeros(n_steps)
# beta_s = np.zeros((n_steps, 2))
# sigma_s = np.zeros(n_steps)

# alpha = 0.0
# beta = np.array([0.0, 0.0])
# sigma = 2.0
# current_lp = log_posterior(alpha, beta, sigma)

# for i in range(n_steps):
#     a_p = alpha + np.random.normal(0, 0.2)
#     b_p = beta + np.random.normal(0, 0.2, size=2)
#     s_p = sigma + np.random.normal(0, 0.1)
#     lp_prop = log_posterior(a_p, b_p, s_p)
#     if np.log(np.random.rand()) < lp_prop - current_lp:
#         alpha, beta, sigma = a_p, b_p, s_p
#         current_lp = lp_prop
#     alpha_s[i] = alpha
#     beta_s[i] = beta
#     sigma_s[i] = sigma

# # ============================================================
# # 5. Grids for likelihood/prior surfaces
# # ============================================================
# b1 = np.linspace(-1, 4, 60)
# b2 = np.linspace(-3, 2, 60)
# B1, B2 = np.meshgrid(b1, b2)
# a_grid = np.linspace(-2, 4, 100)
# b_grid = np.linspace(-2, 4, 100)
# s_grid = np.linspace(0.3, 3, 100)

# # ============================================================
# # 6. Animation
# # ============================================================
# fig, axes = plt.subplots(3, 4, figsize=(18, 10))
# axL, axFit, axB12, axEmpty1 = axes[0]
# axAB1, axAB2, axAS, axS = axes[1]
# axTraceA, axTraceB1, axTraceB2, axEmpty3 = axes[2]

# def update(frame):
#     axL.cla(); axFit.cla(); axB12.cla()
#     axAB1.cla(); axAB2.cla(); axAS.cla()
#     axTraceA.cla(); axTraceB1.cla(); axTraceB2.cla()
    
#     # --------------------------------------------------------
#     # 6.1 Likelihood + prior surface (β1–β2 slice at current alpha/sigma)
#     # --------------------------------------------------------
#     Z = np.zeros_like(B1)
#     for i in range(B1.shape[0]):
#         for j in range(B1.shape[1]):
#             mu = alpha_s[frame] + B1[i,j]*x1 + B2[i,j]*x2
#             Z[i,j] = np.sum(norm.logpdf(y, mu, sigma_s[frame]))
#     Z -= Z.max()
#     axL.contourf(B1, B2, np.exp(Z), levels=30, cmap="viridis", alpha=0.7)
#     prior_pdf = multivariate_normal(prior_mean, prior_cov).pdf(np.dstack((B1,B2)))
#     axL.contour(B1, B2, prior_pdf, levels=8, colors="gray", linestyles="dashed")
#     axL.plot(beta_s[:frame,0], beta_s[:frame,1],'r-', alpha=0.6)
#     axL.plot(beta_s[frame,0], beta_s[frame,1],'ro')
#     axL.scatter(true_beta[0], true_beta[1], c="black", label="true β")
#     axL.set_xlabel("beta_1"); axL.set_ylabel("beta_2"); axL.set_title("β1–β2 likelihood+prior")
#     axL.legend()

#     # --------------------------------------------------------
#     # 6.2 Posterior predictive fit
#     # --------------------------------------------------------
#     axFit.scatter(x1, y, c='black', label="data")
#     y_hat = alpha_s[frame] + beta_s[frame,0]*x1 + beta_s[frame,1]*x2
#     axFit.scatter(x1, y_hat, c='red', label="fit")
#     # Fan showing σ uncertainty
#     for _ in range(30):
#         y_sim = alpha_s[frame] + beta_s[frame,0]*x1 + beta_s[frame,1]*x2 + np.random.normal(0, sigma_s[frame], size=n)
#         axFit.plot(x1, y_sim, color='blue', alpha=0.05)
#     axFit.set_title("Posterior predictive fit + σ uncertainty")
#     axFit.legend()

#     # --------------------------------------------------------
#     # 6.3 Posterior β1–β2 KDE
#     # --------------------------------------------------------
#     if frame > 20:
#         kde = gaussian_kde(beta_s[:frame].T)
#         Zp = kde(np.vstack([B1.ravel(), B2.ravel()])).reshape(B1.shape)
#         axB12.contourf(B1, B2, Zp, levels=30)
#     axB12.scatter(true_beta[0], true_beta[1], c='white')
#     axB12.set_title("Posterior β1–β2 KDE")

#     # --------------------------------------------------------
#     # 6.4 Conditional α slices (α vs β1 and α vs β2)
#     # --------------------------------------------------------
#     axAB1.plot(beta_s[:frame,0], alpha_s[:frame],'b-', alpha=0.6)
#     axAB1.plot(beta_s[frame,0], alpha_s[frame],'bo')
#     axAB1.axhline(true_alpha, color='red')
#     axAB1.set_title("α vs β1")

#     axAB2.plot(beta_s[:frame,1], alpha_s[:frame],'g-', alpha=0.6)
#     axAB2.plot(beta_s[frame,1], alpha_s[frame],'go')
#     axAB2.axhline(true_alpha, color='red')
#     axAB2.set_title("α vs β2")

#     # --------------------------------------------------------
#     # 6.5 Marginal posteriors (α, β1, β2, σ)
#     # --------------------------------------------------------
#     if frame > 20:
#         axAS.plot(a_grid, gaussian_kde(alpha_s[:frame])(a_grid))
#         axAS.axvline(true_alpha, color='red')
#         axAS.plot(a_grid, norm.pdf(a_grid, 0,5), 'k--')
#         axAS.set_title("α posterior + prior")

#         axTraceB1.plot(b_grid, gaussian_kde(beta_s[:frame,0])(b_grid))
#         axTraceB1.axvline(true_beta[0], color='red')
#         axTraceB1.plot(b_grid, norm.pdf(b_grid,0,np.sqrt(prior_cov[0,0])),'k--')
#         axTraceB1.set_title("β1 posterior + prior")

#         axTraceB2.plot(b_grid, gaussian_kde(beta_s[:frame,1])(b_grid))
#         axTraceB2.axvline(true_beta[1], color='red')
#         axTraceB2.plot(b_grid, norm.pdf(b_grid,0,np.sqrt(prior_cov[1,1])),'k--')
#         axTraceB2.set_title("β2 posterior + prior")

#         axS.plot(s_grid, gaussian_kde(sigma_s[:frame])(s_grid))
#         axS.axvline(true_sigma, color='red')
#         axS.plot(s_grid, norm.pdf(s_grid,0,2),'k--')
#         axS.set_title("σ posterior + prior")

#     # --------------------------------------------------------
#     # 6.6 Trace plots
#     # --------------------------------------------------------
#     axTraceA.plot(alpha_s[:frame],'b-', alpha=0.6)
#     axTraceA.axhline(true_alpha, color='red')
#     axTraceA.set_title("α trace")

# frames = range(0, n_steps, 10)
# ani = FuncAnimation(fig, update, frames=frames, interval=50)
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from funcs import clean_prescription_items

files = [f"data/prescriptions_{ptype}_2010-08_2025-08_with_flags.nc" for ptype in ["02_03_0501", "02", "03", "0501"]]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for file, ax in zip(files, axes.flatten()):
    ds = xr.open_dataset(file)
    ds_clean = clean_prescription_items(ds)
    bins = np.linspace(0, np.nanmax(np.log1p(ds['items'].values)), 50)
    ax.hist(np.log1p(ds['items'].values.flatten()), bins=bins, alpha=0.5, label='Original')
    ax.hist(np.log1p(ds_clean['items'].values.flatten()), bins=bins, alpha=0.5, label='Cleaned')

plt.savefig("testing.png", bbox_inches='tight')