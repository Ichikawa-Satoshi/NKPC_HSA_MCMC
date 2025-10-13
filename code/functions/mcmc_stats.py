import pandas as pd
import numpy as np
import arviz as az

def mcmc_stats(runs, burnin=0, chains=4, prob=0.95, param_string=None):
    traces = runs[burnin:, :]
    n = traces.shape[0] // chains
    k = traces.shape[1]
    alpha = 100 * (1.0 - prob)
    post_mean = np.mean(traces, axis=0)
    post_median = np.median(traces, axis=0)
    post_sd = np.std(traces, axis=0)
    mc_err = [az.mcse(traces[:, i].reshape((n, chains), order='F')).item(0) \
              for i in range(k)]
    ci_lower = np.percentile(traces, 0.5 * alpha, axis=0)
    ci_upper = np.percentile(traces, 100 - 0.5 * alpha, axis=0)
    hpdi = az.hdi(traces, hdi_prob=prob)
    rhat = [az.rhat(traces[:, i].reshape((n, chains), order='F')).item(0) \
            for i in range(k)]
    stats = np.vstack((post_mean, post_median, post_sd, mc_err,
                       ci_lower, ci_upper, hpdi.T, rhat)).T
    stats_string = ['mean', 'median', 'sd', 'error',
                    'CI(lower)', 'CI(upper)',
                    'HPDI(lower)', 'HPDI(upper)', '$\\hat R$']
    if not param_string:
        param_string = [f'$\\beta_{i}$' for i in range(k)]
    return pd.DataFrame(stats, index=param_string, columns=stats_string)