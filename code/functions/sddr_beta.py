import numpy as np
import pandas as pd
import scipy.stats as st

def sddr_beta(runs, b0, A0, burnin=1000, at=None, coef_names=None, tiny=1e-300, return_linear_bf=False):
    k = runs.shape[1] - 1   # sigma^2
    post = runs[burnin:, :k]

    if at is None:
        at = np.zeros(k)
    at = np.asarray(at).reshape(-1)
    if at.size == 1:
        at = np.repeat(at.item(), k)

    # prior: beta_j ~ N(b0_j, var_j), var_j = (A0^{-1})_{jj}
    A0_inv = np.linalg.inv(A0)
    prior_scales = np.sqrt(np.diag(A0_inv))

    prior_logpdf = np.array([
        st.norm.logpdf(at[j], loc=b0[j], scale=prior_scales[j]) for j in range(k)
    ], dtype=float)

    # posterior: KDE 
    post_pdf = np.array([
        float(st.gaussian_kde(post[:, j])(at[j])) for j in range(k)
    ], dtype=float)
    post_logpdf = np.log(np.clip(post_pdf, tiny, None))

    # log Bayes Factor
    logBF01 = post_logpdf - prior_logpdf
    log10BF01 = logBF01 / np.log(10.0)

    out = {
        "param": (coef_names if (coef_names is not None and len(coef_names)==k) else [f"beta_{j}" for j in range(k)]),
        "at": at,
        "prior_logpdf": prior_logpdf,
        "post_logpdf": post_logpdf,
        "logBF01": logBF01,
        "log10BF01": log10BF01,
    }

    if return_linear_bf:
        BF01 = np.exp(np.clip(logBF01, -709, 709))
        BF10 = np.exp(np.clip(-logBF01, -709, 709))
        out.update({"BF01": BF01, "BF10": BF10})

    return pd.DataFrame(out)