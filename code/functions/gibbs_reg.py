import pandas as pd
import numpy as np
from tqdm import trange
import scipy.linalg as la
import arviz as az

def gibbs_regression(y, X, hyper_param, burnin=1000, draws=10000, rng=None):
    if rng == None:
        rng = np.random.default_rng(seed=1234)    
    iterations = burnin + draws
    b0, A0, nu0, lam0 = hyper_param
    n, k = X.shape
    XX = X.T @ X
    Xy = X.T @ y
    b_ols = la.solve(XX, Xy)
    rss = np.square(y - X @ b_ols).sum()
    lam_hat = rss + lam0
    nu_star = 0.5 * (n + nu0)
    A0b0 = A0 @ b0
    sigma2 = rss / (n - k)
    runs = np.empty((iterations, k + 1))
    for it in trange(iterations):
        cov_b = la.inv(XX / sigma2 + A0)
        mean_b = cov_b @ (Xy / sigma2 + A0b0)
        b = rng.multivariate_normal(mean_b, cov_b)
        diff = b - b_ols
        lam_star = 0.5 * (diff @ XX @ diff + lam_hat)
        sigma2 = lam_star / rng.gamma(nu_star)
        runs[it, :-1] = b
        runs[it, -1] = sigma2
    return runs