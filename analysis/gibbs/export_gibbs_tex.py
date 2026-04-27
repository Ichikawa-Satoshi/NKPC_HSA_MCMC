from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, invgamma, norm


REPO_ROOT = Path(__file__).resolve().parents[2]
IDATA_ROOT = REPO_ROOT / "results" / "idata"
TABLE_ROOT = REPO_ROOT / "results" / "tex" / "table"
RESULTS_ROOT = REPO_ROOT / "results" / "tex"
FIG_ROOT = REPO_ROOT / "results" / "fig"

GAP_SPECS: tuple[tuple[str, str, str], ...] = (
    ("output_gap_BN", "GDPC1 (BN)", "output_gap_bn"),
    ("unemp_gap", "Unemployment Gap", "unemp_gap"),
    ("markup_BN_inv", "Inverse of Markup (BN)", "markup_bn_inv"),
    ("markup_inv", "Inverse of Markup", "markup_inv"),
)

CORR_SPECS: tuple[tuple[str, str, str], ...] = (
    ("orth", "Uncorr.", "uncorr"),
    ("corr", "Corr.", "corr"),
)

PERIOD_SPECS: tuple[dict[str, str | int], ...] = (
    {
        "label": "1982_2012",
        "title": "Section 1: Gustavo N (1982-2013)",
        "start_year": 1982,
        "end_year": 2012,
    },
    {
        "label": "1988_2017",
        "title": "Section 2: TINC N (1988-2017)",
        "start_year": 1988,
        "end_year": 2017,
    },
)


@dataclass(frozen=True)
class CellSpec:
    mean: float | None
    ci_lo: float | None
    ci_hi: float | None
    sddr_bf01: float | None = None


@dataclass(frozen=True)
class PlotParamSpec:
    label: str
    var: str
    prior: tuple[str, float, float]


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]


def _summarize_draws(draws: np.ndarray) -> tuple[float, float, float]:
    d = _finite(draws)
    if d.size == 0:
        raise ValueError("No finite draws.")
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _format_num(x: float, *, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def _format_sddr(bf01: float, *, digits: int = 4) -> str:
    return f"({_format_num(bf01, digits=digits)})"


def _format_cell(cell: CellSpec, *, include_sddr: bool, digits: int = 4) -> str:
    if cell.mean is None:
        return "--"
    lines = [_format_num(cell.mean, digits=digits)]
    if include_sddr and cell.sddr_bf01 is not None and np.isfinite(cell.sddr_bf01):
        lines.append(_format_sddr(float(cell.sddr_bf01), digits=digits))
    if len(lines) == 1:
        return lines[0]
    inner = r" \\ ".join(lines)
    return r"\begin{tabular}[c]{@{}c@{}}" + inner + r"\end{tabular}"


def _load_posterior_ds(nc_path: Path):
    import xarray as xr

    for engine in ("h5netcdf", "netcdf4", None):
        try:
            return xr.open_dataset(nc_path, group="posterior", engine=engine)
        except Exception:
            pass
    return xr.open_dataset(nc_path, group="posterior")


def _posterior_values(posterior_ds, var: str) -> np.ndarray | None:
    if posterior_ds is None or var not in posterior_ds:
        return None
    return np.asarray(posterior_ds[var])


def _posterior_draws(posterior_ds, var: str) -> np.ndarray | None:
    values = _posterior_values(posterior_ds, var)
    if values is None:
        return None
    return values.reshape(-1)


def _prior_pdf(grid: np.ndarray, prior: tuple[str, float, float]) -> np.ndarray:
    family, p1, p2 = prior
    if family == "normal":
        return norm.pdf(grid, loc=p1, scale=p2)
    if family == "invgamma_variance_sqrt":
        # The sampler's prior is on variance. For sigma=sqrt(v), transform
        # p_sigma(s) = p_v(s^2) * 2s.
        out = np.zeros_like(grid, dtype=float)
        positive = grid > 0
        out[positive] = invgamma.pdf(grid[positive] ** 2, a=p1, scale=p2) * 2.0 * grid[positive]
        return out
    raise ValueError(f"Unknown prior family: {family}")


def _plot_model_posterior_prior(
    *,
    posterior_ds,
    model_name: str,
    params: list[PlotParamSpec],
    out_path: Path,
) -> None:
    available = [p for p in params if _posterior_draws(posterior_ds, p.var) is not None]
    if not available:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.8 * n_rows), squeeze=False)
    axes_flat = axes.reshape(-1)

    for ax, spec in zip(axes_flat, available):
        draws = _finite(_posterior_draws(posterior_ds, spec.var))
        family, p1, p2 = spec.prior
        if family == "normal":
            prior_lo, prior_hi = p1 - 4.0 * p2, p1 + 4.0 * p2
        else:
            if p1 < 0.1 and p2 < 0.1:
                # Very diffuse IG(0.001, 0.001) priors have effectively
                # infinite upper quantiles, so use the posterior scale for
                # plotting while still drawing the exact transformed density.
                prior_lo = 0.0
                prior_hi = float(np.nanquantile(draws, 0.995) * 1.5) if draws.size else 1.0
            else:
                prior_lo, prior_hi = invgamma.ppf([0.001, 0.999], a=p1, scale=p2)
                prior_lo, prior_hi = np.sqrt(max(prior_lo, 0.0)), np.sqrt(prior_hi)

        if draws.size:
            draw_lo, draw_hi = np.nanquantile(draws, [0.005, 0.995])
            lo = float(min(draw_lo, prior_lo))
            hi = float(max(draw_hi, prior_hi))
        else:
            lo, hi = float(prior_lo), float(prior_hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            center = p1 if family == "normal" else 1.0
            lo, hi = center - 1.0, center + 1.0
        if family != "normal":
            lo = max(0.0, lo)

        grid = np.linspace(lo, hi, 400)
        ax.plot(grid, _prior_pdf(grid, spec.prior), label="Prior", lw=2.0, color="#385F71")

        if draws.size >= 2 and np.nanstd(draws) > 1e-12:
            try:
                kde = gaussian_kde(draws)
                ax.plot(grid, kde(grid), label="Posterior", lw=2.0, color="#D1495B")
            except Exception:
                ax.hist(draws, bins=min(25, max(5, draws.size)), density=True, alpha=0.35, label="Posterior")
        elif draws.size:
            ax.axvline(float(np.nanmean(draws)), color="#D1495B", lw=2.0, label="Posterior mean")

        ax.set_title(spec.label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    for ax in axes_flat[len(available):]:
        ax.axis("off")

    fig.suptitle(model_name.replace("_", " "), fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _generate_family_posterior_prior_plots(
    *,
    family_dir: str,
    infl: str,
    model_paths: dict[tuple[str, str], Path],
    params: list[PlotParamSpec],
    model_prefix: str,
    fig_subdir: str = "",
) -> None:
    infl_lower = infl.lower()
    fig_dir = FIG_ROOT / family_dir / fig_subdir / infl_lower if fig_subdir else FIG_ROOT / family_dir / infl_lower
    for path in sorted(fig_dir.glob("*.png")):
        path.unlink()
    for (gap, corr_key), path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing idata file: {path}")
        posterior_ds = _load_posterior_ds(path)
        corr_infix = "orth_" if corr_key == "orth" else ""
        out_name = f"{infl_lower}_{model_prefix}_{corr_infix}{gap.lower()}_prior_posterior.png"
        _plot_model_posterior_prior(
            posterior_ds=posterior_ds,
            model_name=path.stem,
            params=params,
            out_path=fig_dir / out_name,
        )


def _time_axis(n_time: int) -> np.ndarray:
    return np.arange(n_time, dtype=float)


def _time_tick_labels(n_time: int, *, start_year: int, end_year: int) -> tuple[np.ndarray, list[str]]:
    years = np.arange(start_year, end_year + 1, 5)
    ticks = (years - start_year) * 4
    ticks = ticks[ticks < n_time]
    return ticks.astype(float), [str(int(y)) for y in years[: len(ticks)]]


def _flatten_time_draws(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim < 2:
        raise ValueError("Expected time-varying posterior draws.")
    return arr.reshape(-1, arr.shape[-1])


def _summarize_time_path(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.asarray(paths, dtype=float)
    return (
        np.nanmean(finite, axis=0),
        np.nanquantile(finite, 0.025, axis=0),
        np.nanquantile(finite, 0.975, axis=0),
    )


def _plot_time_path_band(
    *,
    paths: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    mean, lo, hi = _summarize_time_path(paths)
    x = _time_axis(mean.size)
    ticks, labels = _time_tick_labels(mean.size, start_year=start_year, end_year=end_year)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.fill_between(x, lo, hi, color="#D1495B", alpha=0.22, label="95% credible interval")
    ax.plot(x, mean, color="#D1495B", lw=2.0, label="Posterior mean")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.55)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _generate_gibbs_steady_kappa_plots(*, infl: str, period_label: str, start_year: int, end_year: int) -> None:
    infl = infl.upper()
    idata_dir = IDATA_ROOT / f"gibbs_hsa_steady_{period_label}_{infl.lower()}"
    for gap, _, _ in GAP_SPECS:
        for corr_label, model_suffix in (("uncorr", f"{infl.lower()}_hsa_{gap.lower()}_{period_label}_uncorr"), ("corr", f"{infl.lower()}_hsa_{gap.lower()}_{period_label}_corr")):
            posterior_ds = _load_posterior_ds(idata_dir / f"{model_suffix}.nc")
            kappa_t = _posterior_values(posterior_ds, "kappa_t")
            if kappa_t is None:
                continue
            paths = _flatten_time_draws(kappa_t)
            out_path = FIG_ROOT / "gibbs_hsa_steady" / period_label / infl.lower() / gap / corr_label / "kappa_t.png"
            _plot_time_path_band(
                paths=paths,
                title=f"{infl} HSA steady {gap} ({corr_label}): kappa_t",
                ylabel=r"$\kappa_t$",
                out_path=out_path,
                start_year=start_year,
                end_year=end_year,
            )


def _generate_gibbs_dynamic_theta_plots(*, infl: str, period_label: str, start_year: int, end_year: int) -> None:
    infl = infl.upper()
    idata_dir = IDATA_ROOT / f"gibbs_hsa_dynamic_{period_label}_{infl.lower()}"
    for gap, _, _ in GAP_SPECS:
        for corr_label, model_suffix in (("uncorr", f"{infl.lower()}_hsa_{gap.lower()}_{period_label}_uncorr"), ("corr", f"{infl.lower()}_hsa_{gap.lower()}_{period_label}_corr")):
            posterior_ds = _load_posterior_ds(idata_dir / f"{model_suffix}.nc")
            theta = _posterior_draws(posterior_ds, "theta")
            nhat = _posterior_values(posterior_ds, "Nhat")
            if theta is None or nhat is None:
                continue
            theta_draws = np.asarray(theta, dtype=float).reshape(-1)
            nhat_draws = _flatten_time_draws(nhat)
            n_draws = min(theta_draws.size, nhat_draws.shape[0])
            paths = theta_draws[:n_draws, None] * nhat_draws[:n_draws, :]
            out_path = FIG_ROOT / "gibbs_hsa_dynamic" / period_label / infl.lower() / gap / corr_label / "theta_t.png"
            _plot_time_path_band(
                paths=paths,
                title=f"{infl} HSA dynamic {gap} ({corr_label}): theta x Nhat",
                ylabel=r"$\theta \hat{N}_t$",
                out_path=out_path,
                start_year=start_year,
                end_year=end_year,
            )


def _posterior_prior_frame(
    *,
    frame_title: str,
    fig_root: str,
    infl_lower: str,
    model_prefix: str,
    corr_key: str,
    fig_subdir: str = "",
) -> str:
    corr_infix = "orth_" if corr_key == "orth" else ""
    subdir = f"{fig_subdir}/" if fig_subdir else ""
    images = []
    for _, _, slug in GAP_SPECS:
        filename = f"{infl_lower}_{model_prefix}_{corr_infix}{slug}_prior_posterior.png"
        images.append(
            rf"\includegraphics[width=0.48\textwidth,height=0.35\textheight,keepaspectratio]{{\detokenize{{../../fig/{fig_root}/{subdir}{infl_lower}/{filename}}}}}"
        )
    return rf"""
\begin{{frame}}
  \frametitle{{{frame_title}}}
  \centering
  \setlength{{\tabcolsep}}{{1pt}}
  \renewcommand{{\arraystretch}}{{1.1}}
  \begin{{tabular}}{{cc}}
    {images[0]} & {images[1]} \\
    {images[2]} & {images[3]} \\
  \end{{tabular}}
\end{{frame}}
"""


def _time_varying_frames(
    *,
    frame_title: str,
    fig_root: str,
    infl_lower: str,
    filename: str,
    fig_subdir: str = "",
) -> str:
    subdir = f"{fig_subdir}/" if fig_subdir else ""
    frames = []
    for _, corr_title, corr_dir in CORR_SPECS:
        images = []
        for gap, _, _ in GAP_SPECS:
            images.append(
                rf"\includegraphics[width=0.48\textwidth,height=0.35\textheight,keepaspectratio]{{\detokenize{{../../fig/{fig_root}/{subdir}{infl_lower}/{gap}/{corr_dir}/{filename}}}}}"
            )
        frames.append(
            rf"""
\begin{{frame}}
  \frametitle{{{frame_title} ({corr_title})}}
  \centering
  \setlength{{\tabcolsep}}{{1pt}}
  \renewcommand{{\arraystretch}}{{1.1}}
  \begin{{tabular}}{{cc}}
    {images[0]} & {images[1]} \\
    {images[2]} & {images[3]} \\
  \end{{tabular}}
\end{{frame}}
"""
        )
    return "\n".join(frames)


def _sddr_bf01_normal(posterior_ds, *, var: str, mu: float, sigma: float) -> float | None:
    try:
        draws = _posterior_draws(posterior_ds, var)
        if draws is None:
            return None
        post = _finite(draws)
        if post.size < 10:
            return None
        n = post.size
        sd = float(np.std(post, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            return None
        h = 1.06 * sd * (n ** (-1.0 / 5.0))
        if not np.isfinite(h) or h <= 0:
            return None
        z = post / h
        post_at0 = float(np.mean(np.exp(-0.5 * z * z)) / (h * np.sqrt(2.0 * np.pi)))
        prior_at0 = float(
            np.exp(-0.5 * ((0.0 - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        )
        return post_at0 / max(prior_at0, 1e-300)
    except Exception:
        return None


def _cell_for_var(
    posterior_ds,
    *,
    var: str,
    prior_mu_sigma: tuple[float, float] | None = None,
) -> CellSpec:
    draws = _posterior_draws(posterior_ds, var)
    if draws is None:
        return CellSpec(mean=None, ci_lo=None, ci_hi=None, sddr_bf01=None)
    mean, lo, hi = _summarize_draws(draws)
    bf01 = None
    if prior_mu_sigma is not None:
        mu, sigma = prior_mu_sigma
        bf01 = _sddr_bf01_normal(posterior_ds, var=var, mu=mu, sigma=sigma)
    return CellSpec(mean=mean, ci_lo=lo, ci_hi=hi, sddr_bf01=bf01)


def _render_table(
    *,
    model_paths: dict[tuple[str, str], Path],
    rows: list[tuple[str, str]],
    prior_specs: dict[str, tuple[float, float]],
    out_path: Path,
    digits: int = 4,
) -> None:
    posterior = {}
    for key, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing idata file: {path}")
        posterior[key] = _load_posterior_ds(path)

    def cell(col_key: tuple[str, str], var: str) -> str:
        prior = prior_specs.get(var)
        include_sddr = prior is not None
        c = _cell_for_var(posterior[col_key], var=var, prior_mu_sigma=prior)
        return _format_cell(c, include_sddr=include_sddr, digits=digits)

    col_order = [(gap, corr_key) for gap, _, _ in GAP_SPECS for corr_key, _, _ in CORR_SPECS]
    gap_header = " & ".join(
        [rf"\multicolumn{{2}}{{c}}{{{gap_label}}}" for _, gap_label, _ in GAP_SPECS]
    )
    corr_header = " & ".join([corr_label for _ in GAP_SPECS for _, corr_label, _ in CORR_SPECS])
    colspec = "c|" + "|".join(["cc"] * len(GAP_SPECS))

    lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        rf"Output gap & {gap_header} \\",
        r"\midrule",
        rf"Parameter & {corr_header} \\",
        r"\midrule",
    ]
    for label, var in rows:
        lines.append(
            " & ".join(
                [label]
                + [cell(col_key, var) for col_key in col_order]
            )
            + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_model_summary_tables(
    *,
    model_paths: dict[tuple[str, str], Path],
    summary_vars: list[str],
    sddr_vars: list[str],
    prior_specs: dict[str, tuple[float, float]],
    out_dir: Path,
    digits: int = 4,
) -> None:
    posterior = {}
    for key, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing idata file: {path}")
        posterior[key] = _load_posterior_ds(path)

    col_order = [(gap, corr_key) for gap, _, _ in GAP_SPECS for corr_key, _, _ in CORR_SPECS]

    def model_name(col_key: tuple[str, str]) -> str:
        return model_paths[col_key].stem.upper()

    def summary_value(col_key: tuple[str, str], var: str) -> str:
        draws = _posterior_draws(posterior[col_key], var)
        if draws is None:
            return "--"
        mean, _, _ = _summarize_draws(draws)
        return _format_num(mean, digits=digits)

    def sddr_value(col_key: tuple[str, str], var: str) -> str:
        prior = prior_specs.get(var)
        if prior is None:
            return "--"
        bf01 = _sddr_bf01_normal(posterior[col_key], var=var, mu=prior[0], sigma=prior[1])
        if bf01 is None or not np.isfinite(bf01):
            return "--"
        return _format_num(float(bf01), digits=digits)

    summary_lines = [
        r"\begin{tabular}{" + "l" * (len(summary_vars) + 1) + r"}",
        r"\toprule",
        "model & " + " & ".join(summary_vars) + r" \\",
        r"\midrule",
    ]
    for col_key in col_order:
        summary_lines.append(
            " & ".join([model_name(col_key)] + [summary_value(col_key, var) for var in summary_vars]) + r" \\"
        )
    summary_lines += [r"\bottomrule", r"\end{tabular}"]

    sddr_lines = [
        r"\begin{tabular}{" + "l" * (len(sddr_vars) + 1) + r"}",
        r"\toprule",
        "model & " + " & ".join([f"SDDR_BF01_{var}" for var in sddr_vars]) + r" \\",
        r"\midrule",
    ]
    for col_key in col_order:
        sddr_lines.append(
            " & ".join([model_name(col_key)] + [sddr_value(col_key, var) for var in sddr_vars]) + r" \\"
        )
    sddr_lines += [r"\bottomrule", r"\end{tabular}"]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.tex").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (out_dir / "sddr.tex").write_text("\n".join(sddr_lines) + "\n", encoding="utf-8")


def render_ces_table(*, infl: str, digits: int = 4) -> None:
    infl = infl.upper()
    idata_dir = IDATA_ROOT / f"gibbs_ces_{infl.lower()}"
    period_label = "1988_2017"
    model_paths = {
        (gap, corr_key): idata_dir / f"{infl.lower()}_ces_{gap.lower()}_{period_label}_{'uncorr' if corr_key == 'orth' else 'corr'}.nc"
        for gap, _, _ in GAP_SPECS
        for corr_key, _, _ in CORR_SPECS
    }
    plot_params = [
        PlotParamSpec(r"$\alpha$", "alpha", ("normal", 0.5, 0.2)),
        PlotParamSpec(r"$\kappa$", "kappa", ("normal", 0.1, 0.2)),
        PlotParamSpec(r"$\phi_1$", "phi_1", ("normal", 0.7, 0.2)),
        PlotParamSpec(r"$\lambda_{ez}$", "lambda_ez", ("normal", 0.0, 0.5)),
        PlotParamSpec(r"$\sigma_e$", "sigma_e", ("invgamma_variance_sqrt", 2.0, 2.0)),
        PlotParamSpec(r"$\sigma_\zeta$", "sigma_zeta", ("invgamma_variance_sqrt", 0.001, 0.001)),
    ]
    _generate_family_posterior_prior_plots(
        family_dir="gibbs_ces",
        infl=infl,
        model_paths=model_paths,
        params=plot_params,
        model_prefix="ces",
    )
    rows = [
        (r"$\alpha$", "alpha"),
        (r"$\kappa$", "kappa"),
        (r"$\phi_1$", "phi_1"),
        (r"$\sigma_e$", "sigma_e"),
        (r"$\sigma_\zeta$", "sigma_zeta"),
    ]
    prior_specs = {"alpha": (0.5, 0.2), "kappa": (0.1, 0.2), "phi_1": (0.7, 0.2)}
    out_path = TABLE_ROOT / "gibbs_ces" / f"table_gibbs_ces_{infl.lower()}.tex"
    _render_table(model_paths=model_paths, rows=rows, prior_specs=prior_specs, out_path=out_path, digits=digits)


def render_hsa_dynamic_table(*, infl: str, digits: int = 4) -> None:
    infl = infl.upper()
    plot_params = [
        PlotParamSpec(r"$\alpha$", "alpha", ("normal", 0.5, 0.2)),
        PlotParamSpec(r"$\kappa$", "kappa", ("normal", 0.1, 0.2)),
        PlotParamSpec(r"$\theta$", "theta", ("normal", 0.1, 0.2)),
        PlotParamSpec(r"$\phi_1$", "phi_1", ("normal", 0.7, 0.2)),
        PlotParamSpec(r"$\lambda_{ez}$", "lambda_ez", ("normal", 0.0, 0.5)),
        PlotParamSpec(r"$\rho_1$", "rho_1", ("normal", 0.2, 0.2)),
        PlotParamSpec(r"$\rho_2$", "rho_2", ("normal", 0.2, 0.2)),
        PlotParamSpec(r"$n$", "n", ("normal", 0.0, 0.1)),
        PlotParamSpec(r"$\sigma_e$", "sigma_e", ("invgamma_variance_sqrt", 2.0, 2.0)),
        PlotParamSpec(r"$\sigma_\zeta$", "sigma_zeta", ("invgamma_variance_sqrt", 0.001, 0.001)),
        PlotParamSpec(r"$\sigma_u$", "sigma_u", ("invgamma_variance_sqrt", 2.0, 2.0)),
        PlotParamSpec(r"$\sigma_\epsilon$", "sigma_eps", ("invgamma_variance_sqrt", 2.0, 2.0)),
    ]
    rows = [
        (r"$\alpha$", "alpha"),
        (r"$\kappa$", "kappa"),
        (r"$\phi_1$", "phi_1"),
        (r"$\theta$", "theta"),
        (r"$\rho_1$", "rho_1"),
        (r"$\rho_2$", "rho_2"),
        (r"$\sigma_e$", "sigma_e"),
        (r"$\sigma_\zeta$", "sigma_zeta"),
        (r"$\sigma_u$", "sigma_u"),
        (r"$\sigma_\epsilon$", "sigma_eps"),
    ]
    prior_specs = {
        "alpha": (0.5, 0.2),
        "kappa": (0.1, 0.2),
        "phi_1": (0.7, 0.2),
        "theta": (0.1, 0.2),
        "rho_1": (0.2, 0.2),
        "rho_2": (0.2, 0.2),
    }
    for period in PERIOD_SPECS:
        period_label = str(period["label"])
        start_year = int(period["start_year"])
        end_year = int(period["end_year"])
        idata_dir = IDATA_ROOT / f"gibbs_hsa_dynamic_{period_label}_{infl.lower()}"
        model_paths = {
            (gap, corr_key): idata_dir / f"{infl.lower()}_hsa_{gap.lower()}_{period_label}_{'uncorr' if corr_key == 'orth' else 'corr'}.nc"
            for gap, _, _ in GAP_SPECS
            for corr_key, _, _ in CORR_SPECS
        }
        _generate_family_posterior_prior_plots(
            family_dir="gibbs_hsa_dynamic",
            infl=infl,
            model_paths=model_paths,
            params=plot_params,
            model_prefix="hsa",
            fig_subdir=period_label,
        )
        _generate_gibbs_dynamic_theta_plots(infl=infl, period_label=period_label, start_year=start_year, end_year=end_year)
        out_dir = TABLE_ROOT / "gibbs_hsa_dynamic" / period_label / infl.lower()
        out_path = out_dir / f"table_gibbs_hsa_dynamic_{infl.lower()}.tex"
        _render_table(model_paths=model_paths, rows=rows, prior_specs=prior_specs, out_path=out_path, digits=digits)
        _render_model_summary_tables(
            model_paths=model_paths,
            summary_vars=["alpha", "kappa", "theta", "phi_1", "rho_1", "rho_2", "rho", "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps"],
            sddr_vars=["alpha", "kappa", "theta", "phi_1", "rho_1", "rho_2"],
            prior_specs=prior_specs,
            out_dir=out_dir,
            digits=digits,
        )


def render_hsa_steady_table(*, infl: str, digits: int = 4) -> None:
    infl = infl.upper()
    plot_params = [
        PlotParamSpec(r"$\alpha$", "alpha", ("normal", 0.5, 0.2)),
        PlotParamSpec(r"$\kappa_0$", "kappa_0", ("normal", 0.1, 0.2)),
        PlotParamSpec(r"$\delta$", "delta", ("normal", 0.1, 0.2)),
        PlotParamSpec(r"$\phi_1$", "phi_1", ("normal", 0.7, 0.2)),
        PlotParamSpec(r"$\lambda_{ez}$", "lambda_ez", ("normal", 0.0, 0.5)),
        PlotParamSpec(r"$\rho_1$", "rho_1", ("normal", 0.2, 0.2)),
        PlotParamSpec(r"$\rho_2$", "rho_2", ("normal", 0.2, 0.2)),
        PlotParamSpec(r"$n$", "n", ("normal", 0.0, 0.1)),
        PlotParamSpec(r"$\sigma_e$", "sigma_e", ("invgamma_variance_sqrt", 2.0, 2.0)),
        PlotParamSpec(r"$\sigma_\zeta$", "sigma_zeta", ("invgamma_variance_sqrt", 0.001, 0.001)),
        PlotParamSpec(r"$\sigma_u$", "sigma_u", ("invgamma_variance_sqrt", 2.0, 2.0)),
        PlotParamSpec(r"$\sigma_\epsilon$", "sigma_eps", ("invgamma_variance_sqrt", 2.0, 2.0)),
    ]
    rows = [
        (r"$\alpha$", "alpha"),
        (r"$\kappa_0$", "kappa_0"),
        (r"$\delta$", "delta"),
        (r"$\phi_1$", "phi_1"),
        (r"$\rho_1$", "rho_1"),
        (r"$\rho_2$", "rho_2"),
        (r"$\sigma_e$", "sigma_e"),
        (r"$\sigma_\zeta$", "sigma_zeta"),
        (r"$\sigma_u$", "sigma_u"),
        (r"$\sigma_\epsilon$", "sigma_eps"),
    ]
    prior_specs = {
        "alpha": (0.5, 0.2),
        "kappa_0": (0.1, 0.2),
        "delta": (0.1, 0.2),
        "phi_1": (0.7, 0.2),
        "rho_1": (0.2, 0.2),
        "rho_2": (0.2, 0.2),
    }
    for period in PERIOD_SPECS:
        period_label = str(period["label"])
        start_year = int(period["start_year"])
        end_year = int(period["end_year"])
        idata_dir = IDATA_ROOT / f"gibbs_hsa_steady_{period_label}_{infl.lower()}"
        model_paths = {
            (gap, corr_key): idata_dir / f"{infl.lower()}_hsa_{gap.lower()}_{period_label}_{'uncorr' if corr_key == 'orth' else 'corr'}.nc"
            for gap, _, _ in GAP_SPECS
            for corr_key, _, _ in CORR_SPECS
        }
        _generate_family_posterior_prior_plots(
            family_dir="gibbs_hsa_steady",
            infl=infl,
            model_paths=model_paths,
            params=plot_params,
            model_prefix="hsa",
            fig_subdir=period_label,
        )
        _generate_gibbs_steady_kappa_plots(infl=infl, period_label=period_label, start_year=start_year, end_year=end_year)
        out_dir = TABLE_ROOT / "gibbs_hsa_steady" / period_label / infl.lower()
        out_path = out_dir / f"table_gibbs_hsa_steady_{infl.lower()}.tex"
        _render_table(model_paths=model_paths, rows=rows, prior_specs=prior_specs, out_path=out_path, digits=digits)
        _render_model_summary_tables(
            model_paths=model_paths,
            summary_vars=["alpha", "kappa_0", "delta", "phi_1", "rho_1", "rho_2", "rho", "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps"],
            sddr_vars=["alpha", "kappa_0", "delta", "phi_1", "rho_1", "rho_2"],
            prior_specs=prior_specs,
            out_dir=out_dir,
            digits=digits,
        )


def _period_hsa_section(*, period_label: str, period_title: str, infl_lower: str) -> str:
    dynamic_uncorr = _posterior_prior_frame(
        frame_title="HSA (Dynamic Effect): Posterior vs Prior (Uncorr.)",
        fig_root="gibbs_hsa_dynamic",
        infl_lower=infl_lower,
        model_prefix="hsa",
        corr_key="orth",
        fig_subdir=period_label,
    )
    dynamic_corr = _posterior_prior_frame(
        frame_title="HSA (Dynamic Effect): Posterior vs Prior (Corr.)",
        fig_root="gibbs_hsa_dynamic",
        infl_lower=infl_lower,
        model_prefix="hsa",
        corr_key="corr",
        fig_subdir=period_label,
    )
    dynamic_paths = _time_varying_frames(
        frame_title=r"HSA (Dynamic Effect): Estimated $\theta \hat{N}_t$",
        fig_root="gibbs_hsa_dynamic",
        infl_lower=infl_lower,
        filename="theta_t.png",
        fig_subdir=period_label,
    )
    steady_uncorr = _posterior_prior_frame(
        frame_title="HSA (Steady State Effect): Posterior vs Prior (Uncorr.)",
        fig_root="gibbs_hsa_steady",
        infl_lower=infl_lower,
        model_prefix="hsa",
        corr_key="orth",
        fig_subdir=period_label,
    )
    steady_corr = _posterior_prior_frame(
        frame_title="HSA (Steady State Effect): Posterior vs Prior (Corr.)",
        fig_root="gibbs_hsa_steady",
        infl_lower=infl_lower,
        model_prefix="hsa",
        corr_key="corr",
        fig_subdir=period_label,
    )
    steady_paths = _time_varying_frames(
        frame_title=r"HSA (Steady State Effect): Estimated $\kappa_t$",
        fig_root="gibbs_hsa_steady",
        infl_lower=infl_lower,
        filename="kappa_t.png",
        fig_subdir=period_label,
    )
    return rf"""
\section{{{period_title}}}

\begin{{frame}}
  \centering
  \Huge{{{period_title}}}
\end{{frame}}

\begin{{frame}}
  \frametitle{{HSA (Dynamic Effect): Posterior Estimates}}
  \begin{{table}}
    \centering
    \tiny
    \resizebox{{0.92\textwidth}}{{!}}{{\input{{../table/gibbs_hsa_dynamic/{period_label}/{infl_lower}/table_gibbs_hsa_dynamic_{infl_lower}.tex}}}}
  \end{{table}}
\end{{frame}}

{dynamic_uncorr}

{dynamic_corr}

{dynamic_paths}

\begin{{frame}}
  \frametitle{{HSA (Steady State Effect): Posterior Estimates}}
  \begin{{table}}
    \centering
    \tiny
    \resizebox{{0.92\textwidth}}{{!}}{{\input{{../table/gibbs_hsa_steady/{period_label}/{infl_lower}/table_gibbs_hsa_steady_{infl_lower}.tex}}}}
  \end{{table}}
\end{{frame}}

{steady_uncorr}

{steady_corr}

{steady_paths}
"""


def _build_results_tex(infl: str) -> str:
    infl_lower = infl.lower()
    inflation_expect = "SPF, 1-year ahead CPI forecast." if infl == "CPI" else "SPF, 1-year ahead GDP deflator forecast."
    ces_uncorr = _posterior_prior_frame(
        frame_title="CES: Posterior vs Prior (Uncorr.)",
        fig_root="gibbs_ces",
        infl_lower=infl_lower,
        model_prefix="ces",
        corr_key="orth",
    )
    ces_corr = _posterior_prior_frame(
        frame_title="CES: Posterior vs Prior (Corr.)",
        fig_root="gibbs_ces",
        infl_lower=infl_lower,
        model_prefix="ces",
        corr_key="corr",
    )
    hsa_sections = "\n".join(
        _period_hsa_section(
            period_label=str(period["label"]),
            period_title=str(period["title"]),
            infl_lower=infl_lower,
        )
        for period in PERIOD_SPECS
    )
    return rf"""\documentclass{{beamer}}
\usetheme{{Madrid}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\usepackage{{graphicx}}

\title{{Gibbs Sampling Results Summary}}
\author{{Ippei Fujiwara (Keio University), James Morley (Sydney University), \ Satoshi Ichikawa (Keio University)}}
\date{{\today}}

\begin{{document}}
\begin{{frame}}
  \titlepage
\end{{frame}}

\begin{{frame}}
  Data:
  \begin{{itemize}}
    \item US quarterly data.
    \item Inflation: {infl}
    \item Output gap (1): GDPC1 (BN)
    \item Output gap (2): Unemployment gap
    \item Output gap (3): Inverse of the markup (BN)
    \item Output gap (4): Inverse of the markup (original series)
    \item Competition measure changes by section.
    \item Inflation Expectations: {inflation_expect}
    \item Gibbs run: $n_{{iter}}=12000$, burn-in $=4000$, thin $=5$.
  \end{{itemize}}
\end{{frame}}

\begin{{frame}}
  \frametitle{{CES: Equation}}
  \begin{{equation*}}
    \begin{{aligned}}
      \pi_t &= \alpha \pi_{{t-1}} + (1-\alpha)\mathbb{{E}}_t \pi_{{t+1}} + \kappa x_t + e_t \\
      x_t &= \phi_1 x_{{t-1}} + \zeta_t
    \end{{aligned}}
  \end{{equation*}}
\end{{frame}}

\begin{{frame}}
  \frametitle{{CES: Posterior Estimates}}
  \begin{{table}}
    \centering
    \scriptsize
    \resizebox{{0.92\textwidth}}{{!}}{{\input{{../table/gibbs_ces/table_gibbs_ces_{infl_lower}.tex}}}}
  \end{{table}}
\end{{frame}}

\begin{{frame}}
  \frametitle{{Chib Marginal Likelihood (Joint)}}
  \begin{{table}}
    \centering
    \tiny
    \resizebox{{0.92\textwidth}}{{!}}{{\input{{../table/gibbs_marginal_likelihood/gibbs_chib_marginal_likelihood_{infl_lower}.tex}}}}
  \end{{table}}
  \vspace{{1mm}}
  \tiny{{Note: joint log marginal likelihood includes the output-gap law of motion. Rankings are within each inflation and output-gap measure.}}
\end{{frame}}

\begin{{frame}}
  \frametitle{{Chib Marginal Likelihood (Conditional NKPC)}}
  \begin{{table}}
    \centering
    \tiny
    \resizebox{{0.92\textwidth}}{{!}}{{\input{{../table/gibbs_marginal_likelihood/gibbs_chib_conditional_marginal_likelihood_{infl_lower}.tex}}}}
  \end{{table}}
  \vspace{{1mm}}
  \tiny{{Note: conditional log marginal likelihood compares the NKPC inflation equation, conditioning on output-gap measures and conditioning HSA on observed competition. Rankings are within each inflation and output-gap measure.}}
\end{{frame}}

{ces_uncorr}

{ces_corr}

{hsa_sections}

\end{{document}}
"""

def write_results_tex(infl: str) -> Path:
    out_dir = RESULTS_ROOT / f"results_gibbs_{infl.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_gibbs_{infl.lower()}.tex"
    out_path.write_text(_build_results_tex(infl), encoding="utf-8")
    return out_path


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["latexmk", "-g", "-pdf", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )


def main() -> None:
    for infl in ("CPI", "PPI"):
        render_ces_table(infl=infl)
        render_hsa_dynamic_table(infl=infl)
        render_hsa_steady_table(infl=infl)
        tex_path = write_results_tex(infl)
        compile_tex(tex_path)
        print(f"written and compiled: {tex_path}")


if __name__ == "__main__":
    main()
