from __future__ import annotations

from pathlib import Path


def save_placeholder_figure(out_path: str | Path, message: str) -> None:
    import matplotlib.pyplot as plt

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_posterior_density(idata, var_name: str, out_path: str | Path) -> None:
    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(az, "plot_posterior"):
        az.plot_posterior(idata, var_names=[var_name], hdi_prob=0.95)
    else:
        values = np.asarray(idata.posterior[var_name], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(values, bins=30, density=True, alpha=0.65)
        ax.axvline(np.mean(values), color="black", lw=1.0)
        ax.set_title(f"Posterior density: {var_name}")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(target, dpi=200, bbox_inches="tight")
    plt.close("all")


def save_posterior_predictive_placeholder(out_path: str | Path) -> None:
    save_placeholder_figure(out_path, "Posterior predictive checks are generated after model runs.")


def _posterior_values(idata, var_name: str):
    import numpy as np

    posterior = getattr(idata, "posterior", None)
    if posterior is None or var_name not in posterior:
        return None
    values = np.asarray(posterior[var_name], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return values if values.size else None


def save_prior_posterior_overlay(idata, var_name: str, prior: tuple[float, float], out_path: str | Path) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde, norm

    values = _posterior_values(idata, var_name)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if values is None or values.size < 3 or float(np.std(values, ddof=1)) <= 0.0:
        return False
    mu, sd = prior
    lo = min(float(np.quantile(values, 0.005)), mu - 4.0 * sd)
    hi = max(float(np.quantile(values, 0.995)), mu + 4.0 * sd)
    xs = np.linspace(lo, hi, 400)
    kde = gaussian_kde(values)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, norm.pdf(xs, loc=mu, scale=sd), label="prior", color="0.35", linestyle="--")
    ax.plot(xs, kde(xs), label="posterior", color="C0")
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"Prior vs posterior: {var_name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_kappa_model_comparison(kappa_table, out_path: str | Path) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if kappa_table is None or kappa_table.empty:
        return False
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in kappa_table.iterrows():
        mean = row.get("kappa_mean")
        label = "kappa"
        if pd.isna(mean):
            mean = row.get("kappa_t_overall_mean", row.get("kappa_0_mean"))
            label = "kappa_t avg" if "kappa_t_overall_mean" in row and pd.notna(row.get("kappa_t_overall_mean")) else "kappa_0"
        if pd.notna(mean):
            rows.append((str(row.get("model", row.get("run", ""))), str(row.get("run", "")), label, float(mean)))
    if not rows:
        return False
    fig, ax = plt.subplots(figsize=(7, max(3.0, 0.45 * len(rows))))
    labels = [f"{model}\n{label}" for model, _, label, _ in rows]
    means = [mean for *_, mean in rows]
    ypos = np.arange(len(rows))
    ax.barh(ypos, means, color="C0", alpha=0.75)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("posterior mean")
    ax.set_title("Kappa comparison across models")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_time_varying_path(idata_by_run: dict[str, object], var_name: str, out_path: str | Path) -> bool:
    import matplotlib.pyplot as plt
    import numpy as np

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    plotted = False
    for run, idata in idata_by_run.items():
        posterior = getattr(idata, "posterior", None)
        if posterior is None or var_name not in posterior:
            continue
        arr = np.asarray(posterior[var_name], dtype=float)
        if arr.ndim < 3:
            continue
        path = arr.reshape(-1, arr.shape[-1])
        mean = np.nanmean(path, axis=0)
        lo = np.nanquantile(path, 0.025, axis=0)
        hi = np.nanquantile(path, 0.975, axis=0)
        x = np.arange(mean.size)
        ax.plot(x, mean, label=str(getattr(idata, "attrs", {}).get("model", run)))
        ax.fill_between(x, lo, hi, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"Time-varying coefficient path: {var_name}")
    ax.set_xlabel("time index")
    ax.set_ylabel("posterior mean and 95% interval")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True
