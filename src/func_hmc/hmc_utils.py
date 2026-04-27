from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import re

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, display
from scipy.stats import norm

import numpyro.distributions as dist

from func_show_results import plot_prior_posterior, sddr_normal


def _as_path(path: str | Path | None) -> Path | None:
    return None if path is None else Path(path)


def _selected_models(idata_map: Mapping[str, object], models_to_show: Sequence[str] | None) -> dict[str, object]:
    if models_to_show is None:
        return dict(idata_map)
    return {name: idata_map[name] for name in models_to_show if name in idata_map}


def _save_tex(df: pd.DataFrame, tex_dir: Path | None, filename: str) -> None:
    if tex_dir is None:
        return
    tex_dir.mkdir(parents=True, exist_ok=True)
    (tex_dir / filename).write_text(df.to_latex(index=False), encoding="utf-8")


def save_idata_map(
    idata_map: Mapping[str, object],
    idata_dir: str | Path,
    *,
    overwrite: bool = True,
) -> dict[str, Path]:
    output_dir = Path(idata_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for run_name, idata in idata_map.items():
        path = output_dir / f"{run_name}.nc"
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        idata.to_netcdf(path)
        saved_paths[run_name] = path
    return saved_paths


def _prior_dist_map(prior_specs: Mapping[str, tuple[float, float]], params: Sequence[str]) -> dict[str, object]:
    return {param: dist.Normal(*prior_specs[param]) for param in params if param in prior_specs}


def _extract_timed_posterior_series(
    idata: object,
    prefix: str,
    *,
    time_index: Sequence[object] | None = None,
) -> pd.DataFrame:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    posterior = idata.posterior
    rows = []

    for name in posterior.data_vars:
        match = pattern.match(name)
        if match is None:
            continue
        t = int(match.group(1))
        values = np.asarray(posterior[name]).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "t": t,
                "var": name,
                "mean": float(np.mean(values)),
                "hdi_2.5%": float(np.quantile(values, 0.025)),
                "hdi_97.5%": float(np.quantile(values, 0.975)),
            }
        )

    if not rows:
        if prefix in posterior.data_vars:
            values = np.asarray(posterior[prefix])
            if values.ndim >= 3:
                # Collapse chain/draw and summarize along the last axis.
                flat = values.reshape(-1, values.shape[-1])
                for t in range(flat.shape[1]):
                    series = flat[:, t]
                    series = series[np.isfinite(series)]
                    if series.size == 0:
                        continue
                    rows.append(
                        {
                            "t": t,
                            "var": prefix,
                            "mean": float(np.mean(series)),
                            "hdi_2.5%": float(np.quantile(series, 0.025)),
                            "hdi_97.5%": float(np.quantile(series, 0.975)),
                        }
                    )
            elif values.ndim == 2:
                # Scalar parameter (chain, draw). Do NOT interpret draws as time.
                flat = values.reshape(-1)
                flat = flat[np.isfinite(flat)]
                if flat.size:
                    rows.append(
                        {
                            "t": 0,
                            "var": prefix,
                            "mean": float(np.mean(flat)),
                            "hdi_2.5%": float(np.quantile(flat, 0.025)),
                            "hdi_97.5%": float(np.quantile(flat, 0.975)),
                        }
                    )
            elif values.ndim == 1:
                flat = values.reshape(-1)
                flat = flat[np.isfinite(flat)]
                if flat.size:
                    rows.append(
                        {
                            "t": 0,
                            "var": prefix,
                            "mean": float(np.mean(flat)),
                            "hdi_2.5%": float(np.quantile(flat, 0.025)),
                            "hdi_97.5%": float(np.quantile(flat, 0.975)),
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["t", "var", "mean", "hdi_2.5%", "hdi_97.5%"])

    out = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    if time_index is not None:
        time_index = list(time_index)
        out = out[out["t"] < len(time_index)].copy()
        out["time"] = [time_index[t] for t in out["t"]]
    else:
        out["time"] = out["t"]
    return out


def _extract_timed_sample_series(
    samples: Mapping[str, object],
    prefix: str,
    *,
    time_index: Sequence[object] | None = None,
) -> pd.DataFrame:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    rows = []

    for name, value in samples.items():
        match = pattern.match(name)
        if match is None:
            continue
        t = int(match.group(1))
        values = np.asarray(value)
        values = values.reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "t": t,
                "var": name,
                "mean": float(np.mean(values)),
                "hdi_2.5%": float(np.quantile(values, 0.025)),
                "hdi_97.5%": float(np.quantile(values, 0.975)),
            }
        )

    if not rows:
        # If samples contains a scalar param exactly equal to prefix, summarize it once.
        if prefix in samples:
            values = np.asarray(samples[prefix]).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size:
                rows.append(
                    {
                        "t": 0,
                        "var": prefix,
                        "mean": float(np.mean(values)),
                        "hdi_2.5%": float(np.quantile(values, 0.025)),
                        "hdi_97.5%": float(np.quantile(values, 0.975)),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["t", "var", "mean", "hdi_2.5%", "hdi_97.5%"])

    out = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    if time_index is not None:
        time_index = list(time_index)
        out = out[out["t"] < len(time_index)].copy()
        out["time"] = [time_index[t] for t in out["t"]]
    else:
        out["time"] = out["t"]
    return out


def _posterior_draws(idata: object, name: str) -> np.ndarray | None:
    if name not in idata.posterior:
        return None
    values = np.asarray(idata.posterior[name])
    if values.ndim <= 1:
        return values.reshape(-1)
    return values.reshape(-1, *values.shape[2:])


def _summarize_draw_matrix(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws = np.asarray(draws)
    mean = np.mean(draws, axis=0)
    lo = np.quantile(draws, 0.025, axis=0)
    hi = np.quantile(draws, 0.975, axis=0)
    return mean, lo, hi


def _assemble_full_trend_draws(idata: object) -> np.ndarray | None:
    bar_n0 = _posterior_draws(idata, "bar_N_0")
    nbar = _posterior_draws(idata, "Nbar")

    if nbar is not None:
        nbar = np.asarray(nbar)
        if nbar.ndim == 1:
            nbar = nbar.reshape(-1, 1)
        elif nbar.ndim >= 2:
            nbar = nbar.reshape(-1, nbar.shape[-1])

        if bar_n0 is not None:
            bar_n0 = np.asarray(bar_n0).reshape(-1, 1)
            if bar_n0.shape[0] == nbar.shape[0]:
                return np.concatenate([bar_n0, nbar], axis=1)
        return nbar

    if bar_n0 is None:
        return None

    timed = [np.asarray(bar_n0).reshape(-1, 1)]

    nbar_vars = []
    for name in idata.posterior.data_vars:
        match = re.match(r"^Nbar_(\d+)$", name)
        if match is not None:
            nbar_vars.append((int(match.group(1)), name))
    if not nbar_vars:
        return np.asarray(bar_n0).reshape(-1, 1)

    for _, name in sorted(nbar_vars, key=lambda x: x[0]):
        timed.append(np.asarray(_posterior_draws(idata, name)).reshape(-1, 1))

    return np.concatenate(timed, axis=1)


def build_posterior_summary_table(
    idata_map: Mapping[str, object],
    *,
    models_to_show: Sequence[str] | None = None,
    params: Sequence[str] = ("alpha", "kappa", "phi_1"),
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for model_name, idata in _selected_models(idata_map, models_to_show).items():
        present = set(idata.posterior.data_vars)
        selected = [param for param in params if param in present]
        if not selected:
            continue
        s = az.summary(idata, var_names=selected, hdi_prob=hdi_prob)
        s = s.reset_index(names="param")
        s.insert(0, "model", model_name)
        rows.append(s[["model", "param", "mean", "hdi_2.5%", "hdi_97.5%"]])

    if not rows:
        return pd.DataFrame(columns=["model", "param", "mean", "hdi_2.5%", "hdi_97.5%"])
    return pd.concat(rows, ignore_index=True)


def build_sddr_table(
    idata_map: Mapping[str, object],
    prior_specs: Mapping[str, tuple[float, float]],
    *,
    models_to_show: Sequence[str] | None = None,
    params: Sequence[str] = ("alpha", "kappa", "phi_1"),
) -> pd.DataFrame:
    rows = []
    for model_name, idata in _selected_models(idata_map, models_to_show).items():
        row = {"model": model_name}
        for param in params:
            if param not in prior_specs:
                row[f"SDDR_BF01_{param}"] = np.nan
                continue
            try:
                mu, sigma = prior_specs[param]
                row[f"SDDR_BF01_{param}"] = sddr_normal(idata, param, mu, sigma)
            except Exception:
                row[f"SDDR_BF01_{param}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_prior_posterior_figure(
    idata: object,
    prior_specs: Mapping[str, tuple[float, float]],
    params: Sequence[str],
    *,
    title: str | None = None,
    figsize_per_panel: tuple[float, float] = (3.2, 2.6),
    xlims: Mapping[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    prior_dists = _prior_dist_map(prior_specs, params)
    fig = plot_prior_posterior(
        idatas=[idata],
        params=params,
        priors=prior_dists,
        figsize_per_panel=figsize_per_panel,
        xlims=xlims,
    )
    if title is not None:
        fig.suptitle(title, y=1.02)
    return fig


def build_convergence_figure(
    idata: object,
    params: Sequence[str],
    *,
    title: str | None = None,
) -> plt.Figure:
    trace_axes = az.plot_trace(idata, var_names=list(params), compact=False)
    trace_array = np.asarray(trace_axes)
    fig_trace = trace_array.ravel()[0].figure if trace_array.size else trace_axes.figure

    if title is not None:
        fig_trace.suptitle(f"{title} - Trace", y=1.02)
    return fig_trace


def build_time_varying_coefficients_figure(
    idata: object,
    *,
    prefixes: Sequence[str] = ("kappa", "theta"),
    time_index: Sequence[object] | None = None,
    title: str | None = None,
    figsize_per_panel: tuple[float, float] = (11.0, 3.0),
) -> plt.Figure | None:
    series = [
        (prefix, _extract_timed_posterior_series(idata, prefix, time_index=time_index))
        for prefix in prefixes
    ]
    series = [(prefix, df) for prefix, df in series if not df.empty]
    if not series:
        return None

    fig, axes = plt.subplots(
        len(series),
        1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(series)),
        squeeze=False,
        sharex=True,
    )

    for ax, (prefix, df) in zip(axes[:, 0], series):
        ax.plot(df["time"], df["mean"], color="C0", lw=2, label="Posterior mean")
        ax.fill_between(df["time"], df["hdi_2.5%"], df["hdi_97.5%"], color="C0", alpha=0.2, label="95% HDI")
        ax.set_ylabel(prefix)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="upper right")

    axes[-1, 0].set_xlabel("Time")
    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def build_time_varying_coefficients_figure_from_samples(
    samples: Mapping[str, object],
    *,
    prefixes: Sequence[str] = ("kappa", "theta"),
    time_index: Sequence[object] | None = None,
    title: str | None = None,
    figsize_per_panel: tuple[float, float] = (11.0, 3.0),
) -> plt.Figure | None:
    series = [
        (prefix, _extract_timed_sample_series(samples, prefix, time_index=time_index))
        for prefix in prefixes
    ]
    series = [(prefix, df) for prefix, df in series if not df.empty]
    if not series:
        return None

    fig, axes = plt.subplots(
        len(series),
        1,
        figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(series)),
        squeeze=False,
        sharex=True,
    )

    for ax, (prefix, df) in zip(axes[:, 0], series):
        ax.plot(df["time"], df["mean"], color="C0", lw=2, label="Posterior mean")
        ax.fill_between(df["time"], df["hdi_2.5%"], df["hdi_97.5%"], color="C0", alpha=0.2, label="95% HDI")
        ax.set_ylabel(prefix)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="upper right")

    axes[-1, 0].set_xlabel("Time")
    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def build_decomposition_figure(
    idata: object,
    observed: Sequence[float] | pd.Series,
    *,
    time_index: Sequence[object] | None = None,
    trend_prefix: str = "Nbar",
    cycle_prefix: str = "Nhat",
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> plt.Figure | None:
    trend = _extract_timed_posterior_series(idata, trend_prefix, time_index=time_index)
    cycle = _extract_timed_posterior_series(idata, cycle_prefix, time_index=time_index)
    if trend.empty and cycle.empty:
        return None

    if time_index is not None:
        x = list(time_index)
    else:
        x = list(range(len(observed)))

    obs = np.asarray(observed).reshape(-1)
    n = min(len(x), len(obs))
    x = x[:n]
    obs = obs[:n]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(x, obs, color="black", lw=1.5, label="Observed N")
    if not trend.empty:
        axes[0].plot(trend["time"], trend["mean"], color="C1", lw=2, label="Posterior mean Nbar")
        axes[0].fill_between(trend["time"], trend["hdi_2.5%"], trend["hdi_97.5%"], color="C1", alpha=0.2)
    axes[0].set_ylabel("Nbar")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    if not cycle.empty:
        axes[1].plot(cycle["time"], cycle["mean"], color="C2", lw=2, label="Posterior mean Nhat")
        axes[1].fill_between(cycle["time"], cycle["hdi_2.5%"], cycle["hdi_97.5%"], color="C2", alpha=0.2)
    axes[1].axhline(0.0, color="black", lw=1, alpha=0.7)
    axes[1].set_ylabel("Nhat")
    axes[1].set_xlabel("Time")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25)

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def build_decomposition_figure_from_samples(
    samples: Mapping[str, object],
    observed: Sequence[float] | pd.Series,
    *,
    time_index: Sequence[object] | None = None,
    trend_prefix: str = "Nbar",
    cycle_prefix: str = "Nhat",
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> plt.Figure | None:
    trend = _extract_timed_sample_series(samples, trend_prefix, time_index=time_index)
    cycle = _extract_timed_sample_series(samples, cycle_prefix, time_index=time_index)
    if trend.empty and cycle.empty:
        return None

    if time_index is not None:
        x = list(time_index)
    else:
        x = list(range(len(observed)))

    obs = np.asarray(observed).reshape(-1)
    n = min(len(x), len(obs))
    x = x[:n]
    obs = obs[:n]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(x, obs, color="black", lw=1.5, label="Observed N")
    if not trend.empty:
        axes[0].plot(trend["time"], trend["mean"], color="C1", lw=2, label="Posterior mean Nbar")
        axes[0].fill_between(trend["time"], trend["hdi_2.5%"], trend["hdi_97.5%"], color="C1", alpha=0.2)
    axes[0].set_ylabel("Nbar")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    if not cycle.empty:
        axes[1].plot(cycle["time"], cycle["mean"], color="C2", lw=2, label="Posterior mean Nhat")
        axes[1].fill_between(cycle["time"], cycle["hdi_2.5%"], cycle["hdi_97.5%"], color="C2", alpha=0.2)
    axes[1].axhline(0.0, color="black", lw=1, alpha=0.7)
    axes[1].set_ylabel("Nhat")
    axes[1].set_xlabel("Time")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25)

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def display_hmc_results(
    idata_map: Mapping[str, object],
    prior_specs: Mapping[str, tuple[float, float]],
    *,
    models_to_show: Sequence[str] | None = None,
    tex_dir: str | Path | None = None,
    params: Sequence[str] = ("alpha", "kappa", "phi_1"),
    title: str = "HMC Results",
) -> None:
    tex_path = _as_path(tex_dir)
    selected = _selected_models(idata_map, models_to_show)

    display(Markdown(f"### {title}"))

    summary = build_posterior_summary_table(selected, models_to_show=None, params=params)
    if summary.empty:
        print("No parameters found for summary.")
    else:
        summary_wide = (
            summary.assign(value=summary["mean"].map(lambda x: f"{x:.4f}"))
            .pivot_table(index="model", columns="param", values="value", aggfunc="first")
            .reindex(columns=params)
            .reset_index()
            .rename_axis(None, axis=1)
        )
        display(HTML("<h3>Summary</h3>"))
        display(summary_wide.style.hide(axis="index"))
        _save_tex(summary_wide, tex_path, "summary.tex")

    sddr = build_sddr_table(selected, prior_specs, models_to_show=None, params=params)
    if not sddr.empty:
        for col in sddr.columns:
            if col.startswith("SDDR_BF01_"):
                sddr[col] = sddr[col].map(lambda v: f"{v:.4f}" if pd.notnull(v) else "")
        display(HTML("<h3>SDDR: Bayes Factors (BF01)</h3>"))
        display(sddr.style.hide(axis="index"))
        _save_tex(sddr, tex_path, "sddr.tex")


def display_hmc_posterior_prior(
    idata_map: Mapping[str, object],
    prior_specs: Mapping[str, tuple[float, float]],
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    params: Sequence[str] = ("alpha", "kappa", "phi_1"),
    title: str = "Prior vs Posterior",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        posterior_vars = set(getattr(idata, "posterior").data_vars)
        missing_posterior = [p for p in params if p not in posterior_vars]
        missing_prior = [p for p in params if p not in prior_specs]
        plot_params = [p for p in params if (p in posterior_vars) and (p in prior_specs)]

        if missing_posterior or missing_prior:
            msg = [f"**{title}: {model_name}**"]
            if missing_posterior:
                msg.append(f"- Missing in posterior: `{', '.join(missing_posterior)}`")
            if missing_prior:
                msg.append(f"- Missing in prior specs: `{', '.join(missing_prior)}`")
            if not plot_params:
                msg.append("- Nothing to plot (no parameters with both prior and posterior).")
            display(Markdown("\n".join(msg)))

        if not plot_params:
            continue

        fig = build_prior_posterior_figure(idata, prior_specs, plot_params, title=f"{title}: {model_name}")
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_prior_posterior.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def display_hmc_convergence(
    idata_map: Mapping[str, object],
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    params: Sequence[str] = ("alpha", "kappa", "phi_1"),
    title: str = "Convergence",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        fig_trace = build_convergence_figure(idata, params, title=f"{title}: {model_name}")
        display(fig_trace)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig_trace.savefig(fig_path / f"{model_name.lower()}_trace.png", dpi=300, bbox_inches="tight")
        plt.close(fig_trace)


def display_hmc_time_varying_coefficients(
    idata_map: Mapping[str, object],
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    time_index: Sequence[object] | None = None,
    prefixes: Sequence[str] = ("kappa", "theta"),
    title: str = "Time-varying coefficients",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        fig = build_time_varying_coefficients_figure(
            idata,
            prefixes=prefixes,
            time_index=time_index,
            title=f"{title}: {model_name}",
        )
        if fig is None:
            continue
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_time_varying_coefficients.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def display_hmc_time_varying_coefficients_from_samples(
    samples_map: Mapping[str, Mapping[str, object]],
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    time_index: Sequence[object] | None = None,
    prefixes: Sequence[str] = ("kappa", "theta"),
    title: str = "Time-varying coefficients",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(samples_map, models_to_show)

    for model_name, samples in selected.items():
        fig = build_time_varying_coefficients_figure_from_samples(
            samples,
            prefixes=prefixes,
            time_index=time_index,
            title=f"{title}: {model_name}",
        )
        if fig is None:
            continue
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_time_varying_coefficients.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def display_hmc_decomposition(
    idata_map: Mapping[str, object],
    observed: Sequence[float] | pd.Series,
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    time_index: Sequence[object] | None = None,
    trend_prefix: str = "Nbar",
    cycle_prefix: str = "Nhat",
    title: str = "Decomposition",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        fig = build_decomposition_figure(
            idata,
            observed,
            time_index=time_index,
            trend_prefix=trend_prefix,
            cycle_prefix=cycle_prefix,
            title=f"{title}: {model_name}",
        )
        if fig is None:
            continue
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_decomposition.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_hsa_full_time_varying_coefficients_figure(
    idata: object,
    *,
    time_index: Sequence[object] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> plt.Figure | None:
    trend_draws = _assemble_full_trend_draws(idata)
    if trend_draws is None or trend_draws.shape[1] <= 1:
        return None

    kappa0 = _posterior_draws(idata, "kappa_0")
    delta = _posterior_draws(idata, "delta")
    theta0 = _posterior_draws(idata, "theta_0")
    gamma = _posterior_draws(idata, "gamma")
    if kappa0 is None or delta is None or theta0 is None or gamma is None:
        return None

    # Use t >= 1 because the model defines the time-varying coefficients from the first state update.
    kappa_draws = kappa0[:, None] + delta[:, None] * trend_draws[:, 1:]
    theta_draws = theta0[:, None] + gamma[:, None] * trend_draws[:, 1:]

    kappa_mean, kappa_lo, kappa_hi = _summarize_draw_matrix(kappa_draws)
    theta_mean, theta_lo, theta_hi = _summarize_draw_matrix(theta_draws)

    if time_index is not None:
        x = list(time_index)[1 : 1 + kappa_draws.shape[1]]
    else:
        x = list(range(1, 1 + kappa_draws.shape[1]))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(x, kappa_mean, color="C0", lw=2, label="Posterior mean")
    axes[0].fill_between(x, kappa_lo, kappa_hi, color="C0", alpha=0.2, label="95% HDI")
    axes[0].set_ylabel("kappa_t")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(x, theta_mean, color="C1", lw=2, label="Posterior mean")
    axes[1].fill_between(x, theta_lo, theta_hi, color="C1", alpha=0.2, label="95% HDI")
    axes[1].set_ylabel("theta_t")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def display_hsa_full_time_varying_coefficients(
    idata_map: Mapping[str, object],
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    time_index: Sequence[object] | None = None,
    title: str = "Time-varying coefficients",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        fig = build_hsa_full_time_varying_coefficients_figure(
            idata,
            time_index=time_index,
            title=f"{title}: {model_name}",
        )
        if fig is None:
            continue
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_time_varying_coefficients.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_hsa_full_decomposition_figure(
    idata: object,
    observed: Sequence[float] | pd.Series,
    *,
    time_index: Sequence[object] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 6.0),
) -> plt.Figure | None:
    trend_draws = _assemble_full_trend_draws(idata)
    if trend_draws is None:
        return None

    trend_mean, trend_lo, trend_hi = _summarize_draw_matrix(trend_draws)
    cycle_draws = np.asarray(observed).reshape(-1)[None, :] - trend_draws
    cycle_mean, cycle_lo, cycle_hi = _summarize_draw_matrix(cycle_draws)

    if time_index is not None:
        x = list(time_index)[: trend_mean.shape[0]]
    else:
        x = list(range(trend_mean.shape[0]))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(x, np.asarray(observed).reshape(-1)[: len(x)], color="black", lw=1.5, label="Observed N")
    axes[0].plot(x, trend_mean[: len(x)], color="C1", lw=2, label="Posterior mean Nbar")
    axes[0].fill_between(x, trend_lo[: len(x)], trend_hi[: len(x)], color="C1", alpha=0.2)
    axes[0].set_ylabel("Nbar")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, cycle_mean[: len(x)], color="C2", lw=2, label="Posterior mean Nhat")
    axes[1].fill_between(x, cycle_lo[: len(x)], cycle_hi[: len(x)], color="C2", alpha=0.2)
    axes[1].axhline(0.0, color="black", lw=1, alpha=0.7)
    axes[1].set_ylabel("Nhat")
    axes[1].set_xlabel("Time")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.25)

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def display_hsa_full_decomposition(
    idata_map: Mapping[str, object],
    observed: Sequence[float] | pd.Series,
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    time_index: Sequence[object] | None = None,
    title: str = "N decomposition",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        fig = build_hsa_full_decomposition_figure(
            idata,
            observed,
            time_index=time_index,
            title=f"{title}: {model_name}",
        )
        if fig is None:
            continue
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_decomposition.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def build_hsa_steady_time_varying_kappa_figure(
    idata: object,
    *,
    time_index: Sequence[object] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (11.0, 4.0),
) -> plt.Figure | None:
    kappa0 = _posterior_draws(idata, "kappa_0")
    delta = _posterior_draws(idata, "delta")
    nbar = _assemble_full_trend_draws(idata)
    if kappa0 is None or delta is None or nbar is None or nbar.shape[1] <= 1:
        return None

    kappa_draws = kappa0[:, None] + delta[:, None] * nbar[:, 1:]
    kappa_mean, kappa_lo, kappa_hi = _summarize_draw_matrix(kappa_draws)

    if time_index is not None:
        x = list(time_index)[1 : 1 + kappa_draws.shape[1]]
    else:
        x = list(range(1, 1 + kappa_draws.shape[1]))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(x, kappa_mean, color="C0", lw=2, label="Posterior mean")
    ax.fill_between(x, kappa_lo, kappa_hi, color="C0", alpha=0.2, label="95% HDI")
    ax.set_ylabel("kappa_t")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    if title is not None:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def display_hsa_steady_time_varying_kappa(
    idata_map: Mapping[str, object],
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    time_index: Sequence[object] | None = None,
    title: str = "Time-varying kappa",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(idata_map, models_to_show)

    for model_name, idata in selected.items():
        fig = build_hsa_steady_time_varying_kappa_figure(
            idata,
            time_index=time_index,
            title=f"{title}: {model_name}",
        )
        if fig is None:
            continue
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_time_varying_kappa.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def display_hmc_decomposition_from_samples(
    samples_map: Mapping[str, Mapping[str, object]],
    observed: Sequence[float] | pd.Series,
    *,
    models_to_show: Sequence[str] | None = None,
    fig_dir: str | Path | None = None,
    time_index: Sequence[object] | None = None,
    trend_prefix: str = "Nbar",
    cycle_prefix: str = "Nhat",
    title: str = "Decomposition",
) -> None:
    fig_path = _as_path(fig_dir)
    selected = _selected_models(samples_map, models_to_show)

    for model_name, samples in selected.items():
        fig = build_decomposition_figure_from_samples(
            samples,
            observed,
            time_index=time_index,
            trend_prefix=trend_prefix,
            cycle_prefix=cycle_prefix,
            title=f"{title}: {model_name}",
        )
        if fig is None:
            continue
        display(fig)
        plt.show()
        if fig_path is not None:
            fig_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path / f"{model_name.lower()}_decomposition.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
