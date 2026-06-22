from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from IPython.display import display


def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)


def prepare_gibbs_sample(
    data: pd.DataFrame,
    *,
    pi_col: str,
    pi_prev_col: str,
    pi_expect_col: str,
    x_col: str,
    x_prev_col: str,
    n_col: str | None = None,
) -> pd.DataFrame:
    cols = [pi_col, pi_prev_col, pi_expect_col, x_col, x_prev_col]
    rename_map = {
        pi_col: "pi",
        pi_prev_col: "pi_prev",
        pi_expect_col: "pi_expect",
        x_col: "x",
        x_prev_col: "x_prev",
    }
    if n_col is not None:
        cols.append(n_col)
        rename_map[n_col] = "N"

    sample = data.loc[:, cols].dropna().copy()
    sample = sample.rename(columns=rename_map)
    sample["DATE"] = pd.to_datetime(sample.index)
    return sample


def format_sample_window(sample: pd.DataFrame) -> str:
    if sample.empty:
        return "empty sample"
    start = pd.Period(sample.index.min(), freq="Q")
    end = pd.Period(sample.index.max(), freq="Q")
    return f"{start}-{end} (T={len(sample)})"


def _selected_models(idata_map, models_to_show=None):
    if models_to_show is None:
        return dict(idata_map)
    return {k: idata_map[k] for k in models_to_show if k in idata_map}


def _posterior_draws(idata, var: str) -> np.ndarray | None:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or var not in posterior:
        return None
    return np.asarray(posterior[var]).reshape(-1)


def _save_tex(df: pd.DataFrame, tex_dir: Path | None, filename: str) -> None:
    if tex_dir is None:
        return
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / filename).write_text(df.to_latex(index=False), encoding="utf-8")


def _sddr_bf01_normal(draws: np.ndarray, mu: float, sigma: float) -> float | None:
    post = np.asarray(draws, dtype=float).reshape(-1)
    post = post[np.isfinite(post)]
    if post.size < 10:
        return None
    sd = float(np.std(post, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    h = 1.06 * sd * (post.size ** (-1.0 / 5.0))
    if not np.isfinite(h) or h <= 0:
        return None
    z = post / h
    post_at0 = float(np.mean(np.exp(-0.5 * z * z)) / (h * np.sqrt(2.0 * np.pi)))
    prior_at0 = float(np.exp(-0.5 * (mu / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi)))
    return post_at0 / max(prior_at0, 1e-300)


def display_hmc_results(
    idata_map,
    prior_specs,
    *,
    models_to_show=None,
    tex_dir: str | Path | None = None,
    params=("alpha", "kappa", "phi_1"),
    title: str = "Gibbs Results",
) -> None:
    tex_path = _as_path(tex_dir)
    selected = _selected_models(idata_map, models_to_show)

    rows = []
    for model_name, idata in selected.items():
        for param in params:
            draws = _posterior_draws(idata, param)
            if draws is None:
                continue
            rows.append(
                {
                    "model": model_name,
                    "param": param,
                    "mean": float(np.nanmean(draws)),
                    "ci_2.5": float(np.nanquantile(draws, 0.025)),
                    "ci_97.5": float(np.nanquantile(draws, 0.975)),
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary_wide = (
            summary.assign(value=summary["mean"].map(lambda x: f"{x:.4f}"))
            .pivot_table(index="model", columns="param", values="value", aggfunc="first")
            .reindex(columns=[p for p in params if p in summary["param"].unique()])
            .reset_index()
            .rename_axis(None, axis=1)
        )
        display(summary_wide)
        _save_tex(summary_wide, tex_path, "summary.tex")

    sddr_rows = []
    for model_name, idata in selected.items():
        row = {"model": model_name}
        has_any = False
        for param in params:
            if param not in prior_specs:
                continue
            draws = _posterior_draws(idata, param)
            if draws is None:
                continue
            bf01 = _sddr_bf01_normal(draws, *prior_specs[param])
            if bf01 is None:
                continue
            row[f"SDDR_BF01_{param}"] = f"{bf01:.4f}"
            has_any = True
        if has_any:
            sddr_rows.append(row)
    sddr = pd.DataFrame(sddr_rows)
    if not sddr.empty:
        _save_tex(sddr, tex_path, "sddr.tex")

    print(title)
    if not summary.empty:
        print(summary)


def display_hmc_posterior_prior(
    idata_map,
    prior_specs,
    *,
    models_to_show=None,
    fig_dir: str | Path | None = None,
    params=("alpha", "kappa", "phi_1"),
    title: str = "Prior vs Posterior",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        plot_params = [p for p in params if p in prior_specs and _posterior_draws(idata, p) is not None]
        if not plot_params:
            continue

        fig, axes = plt.subplots(len(plot_params), 1, figsize=(8.0, 2.5 * len(plot_params)))
        if len(plot_params) == 1:
            axes = [axes]

        for ax, param in zip(axes, plot_params):
            draws = _posterior_draws(idata, param)
            mu, sigma = prior_specs[param]

            lo = min(np.nanmin(draws), mu - 4 * sigma)
            hi = max(np.nanmax(draws), mu + 4 * sigma)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = mu - 4 * sigma, mu + 4 * sigma

            grid = np.linspace(lo, hi, 300)
            ax.plot(grid, norm.pdf(grid, loc=mu, scale=sigma), label="Prior", lw=2)

            finite = draws[np.isfinite(draws)]
            if finite.size >= 2:
                try:
                    kde = gaussian_kde(finite)
                    ax.plot(grid, kde(grid), label="Posterior", lw=2)
                except Exception:
                    ax.hist(finite, bins=min(20, max(5, finite.size)), density=True, alpha=0.35, label="Posterior")
            ax.set_title(f"{model_name}: {param}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")

        fig.suptitle(f"{title}: {model_name}")
        fig.tight_layout()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_prior_posterior.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def save_idata_map(idata_map, out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for model_name, idata in idata_map.items():
        posterior = getattr(idata, "posterior", idata)
        posterior.to_netcdf(out_path / f"{model_name}.nc", mode="w", group="posterior")
