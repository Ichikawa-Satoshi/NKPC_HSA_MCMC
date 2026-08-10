"""Tables and figures for one ``each_result`` cell.

Everything here reads only what a run already saved -- ``posterior.nc``,
``priors.json``, ``data_spec.json``, ``metadata.json`` -- so a cell can be
rebuilt without re-estimating.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nkpc_hsa.reporting.each_result import Cell, RunRef
from nkpc_hsa.reporting.tables import parameter_unit

# Order in which coefficients appear in the tables. Models carry a subset.
COEFFICIENT_ORDER = (
    "alpha",
    "kappa",
    "kappa_0",
    "delta",
    "theta",
    "theta_0",
    "gamma",
    "lambda_ez",
    "phi_1",
    "rho_1",
    "rho_2",
    "n",
    "lambda_E",
    "n_E",
    "rho_E1",
    "rho_E2",
    "rho_NE",
)
SCALE_PARAMETERS = (
    "sigma_e",
    "sigma_eta",
    "sigma_zeta",
    "sigma_u",
    "sigma_eps",
    "sigma_N",
    "sigma_uE",
    "sigma_epsE",
    "sigma_E",
    "rho",
)
# Posterior scale parameter -> the (a, b) inverse-gamma prior on its variance.
VARIANCE_PRIOR_KEYS = {
    "sigma_u": ("a_u", "b_u"),
    "sigma_eps": ("a_eps", "b_eps"),
    "sigma_N": ("a_N", "b_N"),
    "sigma_e": ("a_e", "b_e"),
    "sigma_zeta": ("a_z", "b_z"),
    "sigma_E": ("a_E", "b_E"),
    "sigma_epsE": ("a_epsE", "b_epsE"),
}
MODEL_LABELS = {
    "ces": "CES",
    "hsa_steady": "HSA steady",
    "hsa_dynamic": "HSA dynamic",
    "hsa_const_theta": "HSA const-$\\theta$",
    "hsa_full": "HSA full",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def tex_escape(text: str) -> str:
    """Escape LaTeX specials in prose and identifiers.

    These tables are written with ``escape=False`` because the parameter column
    is deliberate maths, so anything that came from a column name or an error
    message has to be escaped here instead.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(text))


def tex_code(text: str) -> str:
    return f"\\texttt{{{tex_escape(text)}}}"


@dataclass
class LoadedRun:
    """A run with its posterior in memory."""

    ref: RunRef
    posterior: "az.InferenceData"
    priors: dict

    @property
    def model(self) -> str:
        return self.ref.model

    @property
    def prior(self) -> str:
        return self.ref.prior

    def scalars(self, name: str) -> np.ndarray | None:
        ds = self.posterior.posterior
        if name not in ds or ds[name].ndim != 2:
            return None
        values = np.asarray(ds[name]).reshape(-1)
        values = values[np.isfinite(values)]
        return values if values.size else None

    def path(self, name: str) -> np.ndarray | None:
        """Return draws x T for a time-varying quantity."""
        ds = self.posterior.posterior
        if name not in ds or ds[name].ndim != 3:
            return None
        values = np.asarray(ds[name])
        return values.reshape(-1, values.shape[-1])

    def time_index(self, length: int) -> pd.PeriodIndex | np.ndarray:
        start = self.ref.metadata.get("sample_start")
        if not start:
            return np.arange(length)
        try:
            return pd.period_range(start=pd.Timestamp(str(start)), periods=length, freq="Q")
        except (ValueError, TypeError):
            return np.arange(length)


def load_cell(cell: Cell) -> list[LoadedRun]:
    loaded: list[LoadedRun] = []
    for ref in cell.sorted_runs():
        try:
            idata = az.from_netcdf(ref.path / "posterior.nc")
        except (OSError, ValueError):
            continue
        priors_path = ref.path / "priors.json"
        try:
            priors = json.loads(priors_path.read_text()) if priors_path.exists() else {}
        except ValueError:
            priors = {}
        loaded.append(LoadedRun(ref=ref, posterior=idata, priors=priors))
    return loaded


# ---------------------------------------------------------------- tables


def _format_estimate(values: np.ndarray) -> str:
    """Mean over a 95% interval on a second line, so a column stays narrow.

    One column per model already makes these tables wide; keeping the interval
    beside the mean pushed them off the page.
    """
    mean = float(np.mean(values))
    low, high = np.quantile(values, [0.025, 0.975])
    return f"\\makecell[r]{{{mean:.4f}\\\\\\scriptsize [{low:.4f}, {high:.4f}]}}"


DESIGN_LABELS = {
    "annual_q4": "Q4",
    "quarterly_interpolated": "PCHIP",
    "quarterly_observed": "obs",
}


def design_label(frequency: str) -> str:
    return DESIGN_LABELS.get(frequency, frequency)


def column_label(run: LoadedRun, *, with_design: bool) -> str:
    """A run's column heading. The observation design has to appear whenever a
    cell holds more than one, or the mixed-frequency and interpolated estimates
    of the same model silently overwrite each other."""
    base = model_label(run.model)
    return f"{base} ({design_label(run.ref.frequency)})" if with_design else base


def coefficient_table(runs: list[LoadedRun], prior: str, frequency: str | None = None) -> pd.DataFrame:
    """Posterior mean and 95% interval, one column per model.

    Split by observation design as well as prior: a cell holding both the Q4 and
    the PCHIP design would otherwise need ten model columns and could not be read
    at any font size. Units live in their own table for the same reason.
    """
    selected = [
        run
        for run in runs
        if run.prior == prior and (frequency is None or run.ref.frequency == frequency)
    ]
    if not selected:
        return pd.DataFrame()
    names = [name for name in COEFFICIENT_ORDER + SCALE_PARAMETERS if any(run.scalars(name) is not None for run in selected)]
    rows = []
    for name in names:
        row: dict[str, str] = {"Parameter": _tex_parameter(name)}
        for run in selected:
            values = run.scalars(name)
            row[column_label(run, with_design=frequency is None)] = "" if values is None else _format_estimate(values)
        rows.append(row)
    return pd.DataFrame(rows)


def parameter_units_table(runs: list[LoadedRun]) -> pd.DataFrame:
    """The units behind the coefficient tables, kept out of them to save width."""
    names = [
        name
        for name in COEFFICIENT_ORDER + SCALE_PARAMETERS
        if any(run.scalars(name) is not None for run in runs)
    ]
    return pd.DataFrame(
        [{"Parameter": _tex_parameter(name), "Unit": tex_escape(parameter_unit(name))} for name in names]
    )


def prior_comparison_table(runs: list[LoadedRun], parameters: tuple[str, ...]) -> pd.DataFrame:
    """Headline parameters across priors, one row per (model, parameter)."""
    priors = sorted({run.prior for run in runs})
    if len(priors) < 2:
        return pd.DataFrame()
    rows = []
    keys = dict.fromkeys((run.model, run.ref.frequency) for run in runs)
    with_design = len({frequency for _, frequency in keys}) > 1
    for model, frequency in keys:
        for name in parameters:
            row: dict[str, str] = {
                "Model": f"{model_label(model)} ({design_label(frequency)})" if with_design else model_label(model),
                "Parameter": _tex_parameter(name),
            }
            present = False
            for prior in priors:
                match = next(
                    (r for r in runs if r.model == model and r.ref.frequency == frequency and r.prior == prior),
                    None,
                )
                values = None if match is None else match.scalars(name)
                if values is not None:
                    present = True
                row[prior.capitalize()] = "" if values is None else _format_estimate(values)
            if present:
                rows.append(row)
    return pd.DataFrame(rows)


def data_description_table(cell: Cell, runs: list[LoadedRun]) -> pd.DataFrame:
    """What this cell is estimated on."""
    if not runs:
        return pd.DataFrame()
    ref = runs[0].ref
    spec = ref.data_spec
    competition = ref.metadata.get("competition_measurement", {}) or {}
    rows = [
        ("Inflation", f"{cell.price_label} ({tex_code(spec.get('pi_col', ''))})"),
        ("Expected inflation", tex_code(spec.get("pi_expect_col", ""))),
        ("Activity", f"{cell.slack_label} ({tex_code(spec.get('x_col', ''))})"),
        ("Competition", f"{cell.competition_label} ({tex_code(spec.get('n_col', ''))})"),
    ]
    if cell.establishment_label is not None:
        rows.append(("Establishments", f"{cell.establishment_label} ({tex_code(spec.get('e_col', ''))})"))
    rows.extend(
        [
            ("Sample", tex_escape(f"{ref.sample[0]} to {ref.sample[1]}")),
            ("Observations", "" if ref.n_obs is None else str(ref.n_obs)),
            ("Competition design", tex_code(competition.get("frequency", ref.frequency))),
            (
                "Competition observed",
                tex_escape(f"{competition.get('finite_N_obs_count', 'n/a')} of {ref.n_obs} quarters"),
            ),
            ("Competition transform", tex_code(ref.metadata.get("n_transform", ""))),
            ("Models", ", ".join(model_label(m) for m in cell.models())),
            ("Priors", tex_escape(", ".join(cell.priors()))),
            (
                "MCMC",
                tex_escape(
                    f"{ref.metadata.get('n_iter')} iterations, {ref.metadata.get('burn')} burn-in, "
                    f"thin {ref.metadata.get('thin')}, {ref.metadata.get('chains')} chains"
                ),
            ),
        ]
    )
    return pd.DataFrame(rows, columns=["Item", "Value"])


def convergence_table(runs: list[LoadedRun]) -> pd.DataFrame:
    """Worst R-hat and smallest bulk ESS per run, so a cell states its own reliability."""
    rows = []
    for run in runs:
        try:
            summary = az.summary(run.posterior, kind="diagnostics")
        except (ValueError, KeyError, TypeError):
            continue
        if summary.empty:
            continue
        # az.summary can return object-dtype columns, so coerce before reducing.
        rhat = pd.to_numeric(summary.get("r_hat"), errors="coerce").dropna()
        ess = pd.to_numeric(summary.get("ess_bulk"), errors="coerce").dropna()
        rows.append(
            {
                "Model": model_label(run.model),
                "Design": design_label(run.ref.frequency),
                "Prior": run.prior,
                "Max $\\hat{R}$": f"{float(rhat.max()):.3f}" if len(rhat) else "",
                "Min bulk ESS": f"{float(ess.min()):.0f}" if len(ess) else "",
            }
        )
    return pd.DataFrame(rows)


def _tex_parameter(name: str) -> str:
    greek = {
        "alpha": "$\\alpha$",
        "kappa": "$\\kappa$",
        "kappa_0": "$\\kappa_0$",
        "delta": "$\\delta$",
        "theta": "$\\theta$",
        "theta_0": "$\\theta_0$",
        "gamma": "$\\gamma$",
        "lambda_ez": "$\\lambda_{e\\zeta}$",
        "lambda_E": "$\\lambda_E$",
        "phi_1": "$\\phi_1$",
        "rho_1": "$\\rho_1$",
        "rho_2": "$\\rho_2$",
        "rho_E1": "$\\rho_{E1}$",
        "rho_E2": "$\\rho_{E2}$",
        "rho_NE": "$\\rho_{NE}$",
        "rho": "$\\rho$",
        "n": "$n$",
        "n_E": "$n_E$",
    }
    if name in greek:
        return greek[name]
    if name.startswith("sigma_"):
        return f"$\\sigma_{{{name[6:].replace('_', '')}}}$"
    return name.replace("_", "\\_")


# ---------------------------------------------------------------- figures


def _prior_density(name: str, priors: dict, grid: np.ndarray) -> np.ndarray | None:
    """Normal priors for coefficients, inverse-gamma-implied for scale parameters."""
    spec = priors.get(name)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        mean, sd = float(spec[0]), float(spec[1])
        if sd > 0:
            return np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
        return None
    keys = VARIANCE_PRIOR_KEYS.get(name)
    if keys is None:
        return None
    a, b = priors.get(keys[0]), priors.get(keys[1])
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a <= 0 or b <= 0:
        return None
    # sigma^2 ~ IG(a, b)  =>  p(sigma) = 2 sigma * IG(sigma^2; a, b)
    positive = grid > 0
    density = np.zeros_like(grid)
    s2 = grid[positive] ** 2
    from scipy.special import gammaln

    log_ig = a * np.log(b) - gammaln(a) - (a + 1.0) * np.log(s2) - b / s2
    density[positive] = 2.0 * grid[positive] * np.exp(log_ig)
    return density


def plot_prior_posterior(run: LoadedRun, path: Path) -> Path | None:
    """One panel per scalar parameter: posterior histogram with the prior overlaid."""
    names = [name for name in COEFFICIENT_ORDER + SCALE_PARAMETERS if run.scalars(name) is not None]
    if not names:
        return None
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.4 * nrows), squeeze=False)
    for index, name in enumerate(names):
        ax = axes[index // ncols][index % ncols]
        values = run.scalars(name)
        ax.hist(values, bins=30, density=True, color="#4477aa", alpha=0.65, edgecolor="none")
        lo, hi = float(np.min(values)), float(np.max(values))
        pad = 0.5 * (hi - lo) + 1e-9
        grid = np.linspace(lo - pad, hi + pad, 400)
        density = _prior_density(name, run.priors, grid)
        if density is not None and np.any(np.isfinite(density)):
            ax.plot(grid, density, color="#cc3311", lw=1.4, label="prior")
            ax.legend(fontsize=6, frameon=False)
        ax.set_title(_tex_parameter(name), fontsize=9)
        ax.tick_params(labelsize=7)
    for index in range(len(names), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")
    fig.suptitle(f"{model_label(run.model)} -- {run.prior} prior", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def select_for_paths(
    runs: list[LoadedRun], *, frequency: str, prior: str = "baseline"
) -> list[LoadedRun]:
    """Runs for one time-series figure: a single design, a single prior.

    Overlaying both observation designs put two different estimates of the same
    model on one axis, and overlaying priors tripled the lines; each figure now
    carries one line per model.
    """
    same_design = [run for run in runs if run.ref.frequency == frequency]
    at_prior = [run for run in same_design if run.prior == prior]
    if at_prior:
        return at_prior
    # A cell estimated only under weak/tight still deserves a figure.
    fallback = sorted({run.prior for run in same_design})
    return [run for run in same_design if run.prior == fallback[0]] if fallback else []


def plot_path_across_models(runs: list[LoadedRun], variable: str, path: Path, *, ylabel: str) -> Path | None:
    """Posterior median and 90% band of a time-varying quantity, one line per model."""
    selected = [run for run in runs if run.path(variable) is not None]
    if not selected:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = plt.get_cmap("tab10")
    for index, run in enumerate(selected):
        draws = run.path(variable)
        median = np.nanmedian(draws, axis=0)
        low, high = np.nanquantile(draws, [0.05, 0.95], axis=0)
        periods = run.time_index(median.size)
        x = periods.to_timestamp() if isinstance(periods, pd.PeriodIndex) else periods
        color = colors(index % 10)
        ax.plot(x, median, color=color, lw=1.5, label=model_label(run.model))
        ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_competition_decomposition(runs: list[LoadedRun], path: Path) -> Path | None:
    """Trend and cycle of the competition state, one panel per model.

    A joint N/E run adds its establishment states to the same panel, which is
    the only place the two decompositions can be compared period by period.
    """
    selected = [run for run in runs if run.path("Nbar") is not None]
    if not selected:
        return None
    nrows = len(selected)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.2, 2.4 * nrows), squeeze=False, sharex=True)
    for index, run in enumerate(selected):
        ax = axes[index][0]
        series = [("Nbar", "$\\bar{N}$", "#004488"), ("Nhat", "$\\hat{N}$", "#bb5566")]
        if run.path("Ebar") is not None:
            series += [("Ebar", "$\\bar{E}$", "#228833"), ("Ehat", "$\\hat{E}$", "#ccbb44")]
        for name, label, color in series:
            draws = run.path(name)
            if draws is None:
                continue
            median = np.nanmedian(draws, axis=0)
            low, high = np.nanquantile(draws, [0.05, 0.95], axis=0)
            periods = run.time_index(median.size)
            x = periods.to_timestamp() if isinstance(periods, pd.PeriodIndex) else periods
            ax.plot(x, median, color=color, lw=1.4, label=label)
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
        ax.set_title(model_label(run.model), fontsize=9)
        ax.legend(fontsize=7, frameon=False, ncol=4)
        ax.tick_params(labelsize=8)
    fig.supylabel("ten-log-point deviation from the sample mean", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
