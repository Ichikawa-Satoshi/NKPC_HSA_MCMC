import numpy as np
from scipy.stats import norm, gaussian_kde
# posterior sample
import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import norm, invgamma

# SDDR (normal)
def sddr_normal(idata, param, mu, sigma):
    """BF_01 = posterior_density_at_0 / prior_density_at_0"""
    # posterior draws of kappa
    post = np.asarray(idata.posterior[param]).ravel()
    post = post[np.isfinite(post)]
    if post.size < 10:
        return np.nan  # safety
    # posterior density at 0 (KDE)
    kde = gaussian_kde(post)
    post_at0 = float(kde.evaluate([0.0])[0])
    # prior density at 0
    prior_at0 = norm.pdf(0.0, loc=mu, scale=sigma)
    return post_at0 / max(prior_at0, 1e-300)

def plot_prior_posterior(
    idatas,
    params=("kappa", "alpha", "theta"),
    priors=None,
    xlims=None,
    grid=True,
    figsize_per_panel=(3.2, 2.6),
):
    """
    Simple posterior (kde) vs prior (pdf) overlay grid.
    - rows: idatas
    - cols: params
    - HDI: off
    """
    # accept single idata / single param
    if not isinstance(idatas, (list, tuple)):
        idatas = [idatas]
    if isinstance(params, str):
        params = (params,)

    def to_np(x):
        try:
            return np.asarray(x).ravel()
        except Exception:
            return np.asarray(getattr(x, "values", x)).ravel()

    def prior_pdf(prior_dist, x):
        # generic log_prob path (numpyro/pyro etc.)
        if hasattr(prior_dist, "log_prob"):
            try:
                return np.exp(to_np(prior_dist.log_prob(x)))
            except Exception:
                pass

        # small fallback for Normal / InverseGamma
        name = prior_dist.__class__.__name__.lower()
        if "normal" in name:
            mu = getattr(prior_dist, "loc", None)
            sig = getattr(prior_dist, "scale", None)
            if mu is not None and sig is not None:
                mu = float(np.asarray(mu)); sig = float(np.asarray(sig))
                return norm.pdf(x, mu, sig)

        if "inversegamma" in name or "invgamma" in name:
            a = getattr(prior_dist, "concentration", None)
            sc = getattr(prior_dist, "scale", None)
            if a is None:
                a = getattr(prior_dist, "alpha", None)
            if sc is None:
                sc = getattr(prior_dist, "beta", None)
            if a is not None and sc is not None:
                a = float(np.asarray(a)); sc = float(np.asarray(sc))
                return invgamma.pdf(x, a=a, scale=sc)

        return None

    def default_xlim(p, post=None):
        # 1) user override
        if xlims and p in xlims:
            return xlims[p]

        # 2) prior Normal -> mu±5σ
        if p in priors:
            d = priors[p]
            if "normal" in d.__class__.__name__.lower():
                mu = getattr(d, "loc", None)
                sig = getattr(d, "scale", None)
                if mu is not None and sig is not None:
                    mu = float(np.asarray(mu)); sig = float(np.asarray(sig))
                    return (mu - 5*sig, mu + 5*sig)

        # 3) posterior quantiles
        if post is not None and len(post) > 20:
            lo, hi = np.quantile(post, [0.005, 0.995])
            pad = 0.1*(hi-lo) if hi > lo else 1.0
            return (lo-pad, hi+pad)

        return (-5, 5)

    n_rows, n_cols = len(idatas), len(params)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0]*n_cols, figsize_per_panel[1]*n_rows),
        squeeze=False,
        sharey="col"
    )

    for i, idata in enumerate(idatas):
        avail = set(getattr(getattr(idata, "posterior", None), "data_vars", {}).keys())

        for j, p in enumerate(params):
            ax = axes[i, j]

            # posterior samples (for xlim fallback)
            post = None
            if hasattr(idata, "posterior") and p in avail:
                post = to_np(idata.posterior[p])

            xmin, xmax = default_xlim(p, post)

            # posterior (HDI off)
            if p in avail:
                az.plot_posterior(
                    idata,
                    var_names=[p],
                    kind="kde",
                    point_estimate=None,
                    hdi_prob="hide",   # <- HDI off
                    ax=ax
                )
            else:
                ax.text(0.5, 0.5, f"no posterior: {p}", ha="center", va="center", transform=ax.transAxes)
                ax.set_yticks([])

            # prior overlay
            if p in priors:
                x = np.linspace(xmin, xmax, 800)
                y = prior_pdf(priors[p], x)
                if y is not None:
                    ax.plot(x, y, "r--", lw=2)

            ax.set_xlim(xmin, xmax)
            ax.set_xlabel(p)
            if j == 0:
                ax.set_ylabel("Density")

            if grid:
                ax.grid(True, alpha=0.25, linewidth=0.8)

    fig.legend(
        handles=[
            Line2D([0], [0], color="C0", lw=2, label="Posterior"),
            Line2D([0], [0], color="red", lw=2, ls="--", label="Prior"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )
    plt.tight_layout()
    return fig