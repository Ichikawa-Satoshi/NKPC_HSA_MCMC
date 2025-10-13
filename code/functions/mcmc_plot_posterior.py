import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def plot_prior_posterior(
    runs,
    prior_specs,
    burnin=0,
    param_names=None,
    grid=300,
    figsize_scale=1.6,
    kde_bw_method=None,   # 例: 'scott' | 'silverman' | float
):
    runs = np.asarray(runs)
    if runs.ndim != 2:
        raise ValueError("runs must be 2D array (n_draws, n_params).")
    if burnin >= runs.shape[0]:
        raise ValueError("burnin must be smaller than the number of draws.")

    traces = runs[burnin:, :]
    n_draws, k = traces.shape

    #
    if param_names is None:
        if isinstance(prior_specs, dict) and len(prior_specs) == k:
            param_names = list(prior_specs.keys())
        else:
            param_names = [f"param_{i}" for i in range(k)]
    if len(param_names) != k:
        raise ValueError("len(param_names) must match runs.shape[1]")

    # prior 
    def prior_pdf(name):
        spec = prior_specs.get(name, None)
        if spec is None:
            return None

        dist = spec.get("dist", "").lower()
        if dist == "norm":
            loc = spec["loc"]; scale = spec["scale"]
            def f(x): return st.norm.pdf(x, loc=loc, scale=scale)
            support = (-np.inf, np.inf)

        elif dist == "gamma":
            a = spec["a"]; scale = spec["scale"]
            def f(x): return st.gamma.pdf(x, a=a, scale=scale)
            support = (0.0, np.inf)

        elif dist == "inv_gamma":
            a = spec["a"]; scale = spec["scale"]
            def f(x): return st.invgamma.pdf(x, a=a, scale=scale)
            support = (0.0, np.inf)

        elif dist == "lognorm":
            s = spec["s"]; scale = spec["scale"]
            def f(x): return st.lognorm.pdf(x, s=s, scale=scale)
            support = (0.0, np.inf)

        elif dist == "half_norm":
            scale = spec["scale"]
            def f(x): return st.halfnorm.pdf(x, scale=scale)
            support = (0.0, np.inf)

        else:
            return None

        return f, support

    fig, axes = plt.subplots(k, 1, figsize=(8, figsize_scale * k), sharex=False)
    if k == 1:
        axes = np.array([axes])

    for i, name in enumerate(param_names):
        ax = axes[i]
        s = traces[:, i]


        prior_def = prior_pdf(name)
        s_min, s_max = np.min(s), np.max(s)


        if not np.isfinite(s_min) or not np.isfinite(s_max) or (s_max - s_min) < 1e-12:
            s_min, s_max = float(np.mean(s) - 1.0), float(np.mean(s) + 1.0)

        pad = 0.15 * max(1e-12, abs(s_min)) + 0.15 * max(1e-12, abs(s_max))
        x_lo = s_min - pad
        x_hi = s_max + pad

        if prior_def is not None:
            f_prior, support = prior_def
            if support[0] == 0.0:
                x_lo = max(0.0, x_lo)

        x = np.linspace(x_lo, x_hi, grid)

        # Posterior KDE
        try:
            kde = st.gaussian_kde(s, bw_method=kde_bw_method)
            post_y = kde(x)
        except Exception:
            mu, sd = np.mean(s), np.std(s) + 1e-12
            post_y = st.norm.pdf(x, loc=mu, scale=sd)

        # Prior pdf
        if prior_def is not None:
            f_prior, support = prior_def
            prior_y = f_prior(x)
            ax.plot(x, prior_y, label="prior", linewidth=1.8, alpha=0.9)
        else:
            prior_y = None

        ax.plot(x, post_y, label="posterior", linewidth=2.2, alpha=0.9)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=False)
        ax.set_xlim(x_lo, x_hi)
        ymax = np.nanmax([np.nanmax(post_y), np.nanmax(prior_y) if prior_y is not None else 0.0])
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = 1.0
        ax.set_ylim(0, 1.05 * ymax)

        if i == k - 1:
            ax.set_xlabel("value")

    plt.tight_layout()
    plt.show()
    return fig, axes
