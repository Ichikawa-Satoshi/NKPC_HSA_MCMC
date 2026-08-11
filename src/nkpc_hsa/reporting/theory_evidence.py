"""Restriction-report evidence derived from already-validated theory runs.

``theory_report`` owns provenance: which runs are current, comparable, and
converged. This module owns everything the restriction report needs *beyond*
the restriction taxonomy itself -- the data description, the prior and sampling
design, the cross-equation restriction check, conditional in-sample fit,
convergence, and the prior-against-posterior figures.

Two rules hold throughout:

* Nothing here reads ``results/runs`` or any historical artifact. Every number
  comes from a run directory that ``theory_report`` has already signed off, or
  from the processed data that run recorded in its own ``data_spec.json``.
* Nothing here re-estimates. The fit section is therefore a *conditional*
  in-sample diagnostic evaluated at the stored latent paths, not a marginal
  likelihood; the report says so where the table appears.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nkpc_hsa.theory_models import THEORY_MODELS, theory_model_definition


RunSet = Mapping[str, tuple[Path, Mapping[str, Any], Any]]

# Posterior name of the competition *cycle* the inflation equation loads on.
# F0 has no Nbar state, so its single deviation state plays that role.
CYCLE_VARIABLE = {"hsa_f0": "N_deviation"}
DEFAULT_CYCLE_VARIABLE = "Nhat"

# Colour is fixed per model, not per panel. Matplotlib's per-axes cycle would
# hand the first colour to whichever model happens to appear in that panel, so
# F0's colour would silently label U wherever F0 has no such coefficient.
# Stored but not sampled: fixed under this design, a deterministic transform of
# a sampled coefficient, or sampler telemetry. Excluded from convergence scans.
DERIVED_OR_TELEMETRY = frozenset(
    {"zeta0", "mu0", "d_kappa_d_logN", "admissibility_violation"}
)

MODEL_COLORS = {
    "hsa_f0": "tab:blue",
    "hsa_u": "tab:orange",
    "hsa_r1": "tab:green",
    "hsa_r2": "tab:red",
    "hsa_r3": "tab:purple",
}

GAUSSIAN_PRIOR_LABELS = (
    ("alpha", r"$\alpha$", "inflation inertia"),
    ("kappa_0", r"$\kappa_0$", "slope at the reference"),
    ("theta_0", r"$\theta_0$", "entry coefficient"),
    ("delta", r"$\kappa_N$", "slope response to $\\bar N$ (U only; derived in R1--R3)"),
    ("gamma", r"$\gamma$", "entry response to $\\bar N$"),
    ("phi_1", r"$\phi_1$", "activity AR(1)"),
    ("rho_1", r"$\rho_1$", "competition cycle AR(2)"),
    ("rho_2", r"$\rho_2$", "competition cycle AR(2)"),
    ("n", r"$n$", "reference drift"),
)

INVERSE_GAMMA_PRIOR_LABELS = (
    ("a_e", "b_e", r"$\sigma_\eta^2$", "inflation equation"),
    ("a_z", "b_z", r"$\sigma_\zeta^2$", "activity innovation"),
    ("a_u", "b_u", r"$\sigma_u^2$", "competition cycle innovation"),
    ("a_eps", "b_eps", r"$\sigma_\epsilon^2$", "reference innovation"),
    ("a_N", "b_N", r"$\sigma_N^2$", "competition measurement"),
)

PRIOR_POSTERIOR_PARAMETERS = (
    ("alpha", "alpha", r"$\alpha$"),
    ("kappa_0", "kappa_0", r"$\kappa_0$"),
    ("theta_0", "theta_0", r"$\theta_0$"),
    ("delta", "kappa_N_empirical", r"$\kappa_N$"),
    ("gamma", "gamma", r"$\gamma$"),
    ("phi_1", "phi_1", r"$\phi_1$"),
)


def _verbatim(text: Any) -> str:
    """Render a code-like identifier for a fragment written with ``escape=False``.

    Config identifiers such as ``exact_qoq_percent_change_nonannualized`` are a
    single unbreakable box in ``\\texttt``, which overflows any sane column, so
    a break opportunity is offered after each underscore.
    """
    rendered = str(text).replace("\\", "").replace("%", "\\%").replace("&", "\\&")
    rendered = rendered.replace("_", "\\_\\allowbreak{}")
    return "\\texttt{" + rendered + "}"


def _hierarchy(metadata: Mapping[str, Any]) -> str:
    return str(metadata.get("model_hierarchy", metadata.get("model", "?")))


def _flat(idata, name: str) -> np.ndarray:
    """Return draws as ``(draw, ...)`` with the chain axis folded in."""
    values = np.asarray(idata.posterior[name], dtype=float)
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def _quantile_row(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    flat = flat[np.isfinite(flat)]
    return {
        "Mean": float(np.mean(flat)),
        "SD": float(np.std(flat, ddof=1)),
        "5%": float(np.quantile(flat, 0.05)),
        "95%": float(np.quantile(flat, 0.95)),
    }


def free_coefficient_count(slug: str) -> int:
    """Coefficients drawn in the inflation-equation block, restriction included.

    R1 and R2 sample one fewer coefficient than U because $\\kappa_N$ is a
    deterministic function of $\\theta_0$; R2 restricts the same count to a
    smaller region. R3 drops $\\gamma$ as well.
    """
    definition = theory_model_definition(slug)
    count = 3  # alpha, kappa_0, theta_0
    if not definition.cross_equation_restriction and definition.moving_reference:
        count += 1  # kappa_N sampled independently
    if definition.gamma_restriction not in {"not_applicable", "gamma = 0"}:
        count += 1  # gamma
    return count + 1  # lambda_ez


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_run_model_data(run_dir: Path) -> tuple[pd.Index | None, dict[str, np.ndarray]]:
    """Rebuild the exact estimation sample a run used, from its own data spec."""
    from nkpc_hsa.inference.wrappers import _coerce_model_data, model_sample_index
    from nkpc_hsa.paths import project_path

    spec = json.loads((Path(run_dir) / "data_spec.json").read_text(encoding="utf-8"))
    processed = project_path("data", "processed", "model_ready.csv")
    frame = pd.read_csv(processed, parse_dates=["DATE"]).set_index("DATE")
    return model_sample_index(frame, spec), _coerce_model_data(frame, data_spec=spec)


def data_summary_table(runs: RunSet) -> pd.DataFrame:
    """Series-by-series description of the one data cell every run shares."""
    if not runs:
        return pd.DataFrame()
    run_dir, metadata, _ = next(iter(runs.values()))
    spec = json.loads((run_dir / "data_spec.json").read_text(encoding="utf-8"))
    _, model_data = load_run_model_data(run_dir)
    rows = [
        (
            "Inflation $\\pi_t$",
            spec.get("pi_col"),
            metadata.get("inflation_transformation"),
            "pi",
        ),
        (
            "Lagged inflation $\\pi_{t-1}$",
            spec.get("pi_prev_col"),
            "one-quarter lag of the same transformation",
            "pi_prev",
        ),
        (
            "Expected inflation $E_t\\pi_{t+1}$",
            spec.get("pi_expect_col"),
            spec.get("expectation_transformation"),
            "pi_expect",
        ),
        (
            "Real marginal cost $x_t$",
            spec.get("x_col"),
            spec.get("activity_transformation"),
            "x",
        ),
        (
            "Competition proxy $N_t$",
            spec.get("n_col"),
            f"{metadata.get('n_transform')}, observed {metadata.get('competition_measurement_frequency')}",
            "N",
        ),
    ]
    out = []
    for label, column, transformation, key in rows:
        values = np.asarray(model_data.get(key, []), dtype=float)
        finite = values[np.isfinite(values)]
        out.append(
            {
                "Series": label,
                "Column": _verbatim(column),
                "Transformation": _verbatim(transformation),
                "Mean": float(np.mean(finite)) if finite.size else np.nan,
                "SD": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
                "Min": float(np.min(finite)) if finite.size else np.nan,
                "Max": float(np.max(finite)) if finite.size else np.nan,
            }
        )
    return pd.DataFrame(out)


def plot_data_series(runs: RunSet, output_path: Path) -> Path | None:
    """The four observed series behind every restriction run."""
    import matplotlib.pyplot as plt

    if not runs:
        return None
    run_dir, metadata, _ = next(iter(runs.values()))
    index, model_data = load_run_model_data(run_dir)
    x_axis = index if index is not None else np.arange(len(model_data["pi"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.6), sharex=True)
    axes[0].plot(x_axis, model_data["pi"], lw=1.5, label="$\\pi_t$")
    axes[0].plot(x_axis, model_data["pi_expect"], lw=1.3, label="$E_t\\pi_{t+1}$")
    axes[0].set_ylabel("percent")
    axes[0].set_title("Inflation and expected inflation")

    axes[1].plot(x_axis, model_data["x"], lw=1.5, color="tab:red")
    axes[1].axhline(0.0, color="black", lw=0.8, alpha=0.4)
    axes[1].set_ylabel("percent")
    axes[1].set_title("Real marginal cost proxy $x_t$")

    axes[2].plot(x_axis, model_data["N"], lw=1.5, color="tab:green")
    axes[2].set_ylabel("raw column units")
    axes[2].set_title(
        f"Competition proxy {metadata.get('competition_proxy')} "
        f"({metadata.get('competition_measurement_frequency')} observation)"
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Priors and sampling design
# ---------------------------------------------------------------------------


def _shared_priors(runs: RunSet) -> dict[str, Any]:
    priors: dict[str, Any] | None = None
    for run_dir, metadata, _ in runs.values():
        path = run_dir / "priors.json"
        if not path.exists():
            continue
        current = json.loads(path.read_text(encoding="utf-8"))
        if priors is None:
            priors = current
        elif current != priors:
            raise ValueError(
                f"Theory runs do not share one prior specification; {metadata.get('model')} differs."
            )
    return priors or {}


def priors_table(runs: RunSet) -> pd.DataFrame:
    priors = _shared_priors(runs)
    rows = []
    for key, label, meaning in GAUSSIAN_PRIOR_LABELS:
        entry = priors.get(key)
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        rows.append(
            {
                "Parameter": label,
                "Role": meaning,
                "Prior": f"N({float(entry[0]):.3g}, {float(entry[1]):.3g}$^2$)",
            }
        )
    for a_key, b_key, label, meaning in INVERSE_GAMMA_PRIOR_LABELS:
        if a_key not in priors or b_key not in priors:
            continue
        rows.append(
            {
                "Parameter": label,
                "Role": meaning,
                "Prior": f"IG({float(priors[a_key]):.3g}, {float(priors[b_key]):.3g})",
            }
        )
    return pd.DataFrame(rows)


SAMPLER_LABELS = {
    "fixed_reference_ffbs": "fixed-reference FFBS",
    "particle_gibbs": "particle Gibbs",
    "joint_ffbs": "joint FFBS",
    "linear_truncated_gaussian_coordinate_gibbs": "truncated Gaussian",
    "positive_gaussian_rejection": "positive Gaussian",
}


def sampling_design_table(runs: RunSet) -> pd.DataFrame:
    rows = []
    for slug in THEORY_MODELS:
        if slug not in runs:
            continue
        _, metadata, idata = runs[slug]
        stored = int(idata.posterior.sizes["chain"]) * int(idata.posterior.sizes["draw"])
        rows.append(
            {
                "Model": _hierarchy(metadata),
                "State sampler": SAMPLER_LABELS.get(
                    str(metadata.get("sampler_type", "")), str(metadata.get("sampler_type", "")).replace("_", " ")
                ),
                "Coefficient block": SAMPLER_LABELS.get(
                    str(metadata.get("parameter_sampler", "")),
                    str(metadata.get("parameter_sampler", "")).replace("_", " "),
                ),
                "Free coefs": free_coefficient_count(slug),
                "Iterations": int(metadata.get("n_iter", 0)),
                "Burn": int(metadata.get("burn", 0)),
                "Thin": int(metadata.get("thin", 0)),
                "Chains": int(metadata.get("chains", 0)),
                "Stored draws": stored,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conditional in-sample fit
# ---------------------------------------------------------------------------


def _residual_draws(slug: str, metadata: Mapping[str, Any], idata, model_data: Mapping[str, np.ndarray]):
    """Inflation-equation residuals at every stored draw.

    ``eta_t = (pi_t - Epi_t) - alpha a_t - kappa_t x_t + theta_t c_t - lambda zeta_t``

    with ``c_t`` the competition cycle state (``Nhat``, or F0's single
    deviation state) and ``kappa_t``/``theta_t`` the stored physical paths, so
    the arithmetic matches the sampler's own regression exactly.
    """
    pi = np.asarray(model_data["pi"], dtype=float)
    pi_prev = np.asarray(model_data["pi_prev"], dtype=float)
    pi_expect = np.asarray(model_data["pi_expect"], dtype=float)
    x = np.asarray(model_data["x"], dtype=float)
    x_prev = np.asarray(model_data["x_prev"], dtype=float)

    y = pi - pi_expect
    a_t = pi_prev - pi_expect

    alpha = _flat(idata, "alpha")[:, None]
    lambda_ez = _flat(idata, "lambda_ez")[:, None]
    phi_1 = _flat(idata, "phi_1")[:, None]
    kappa_t = _flat(idata, "kappa_t")
    theta_t = _flat(idata, "theta_t")
    cycle = _flat(idata, CYCLE_VARIABLE.get(slug, DEFAULT_CYCLE_VARIABLE))

    if kappa_t.shape[1] != y.size:
        raise ValueError(
            f"{metadata.get('model')} stored {kappa_t.shape[1]} periods but its data spec yields {y.size}."
        )
    zeta = x[None, :] - phi_1 * x_prev[None, :]
    eta = y[None, :] - alpha * a_t[None, :] - kappa_t * x[None, :] + theta_t * cycle - lambda_ez * zeta

    sigma_e2 = _flat(idata, "sigma_e") ** 2
    sigma_zeta2 = _flat(idata, "sigma_zeta") ** 2
    sigma_eta2 = np.clip(sigma_e2 - _flat(idata, "lambda_ez") ** 2 * sigma_zeta2, 1e-12, None)
    return eta, sigma_eta2


def conditional_fit_table(runs: RunSet) -> pd.DataFrame:
    """In-sample fit of the inflation equation, conditional on the latent paths.

    This is deliberately not a marginal likelihood: the latent competition path
    is held at its stored draw, so models with a richer state block are not
    charged for that richness. It answers "does imposing the restriction move
    the fitted inflation equation", not "which model does the data prefer".
    """
    rows = []
    for slug in THEORY_MODELS:
        if slug not in runs:
            continue
        run_dir, metadata, idata = runs[slug]
        _, model_data = load_run_model_data(run_dir)
        eta, sigma_eta2 = _residual_draws(slug, metadata, idata, model_data)
        T = eta.shape[1]

        rmse = np.sqrt(np.mean(eta ** 2, axis=1))
        loglik = -0.5 * T * np.log(2.0 * np.pi * sigma_eta2) - 0.5 * np.sum(eta ** 2, axis=1) / sigma_eta2

        eta_bar = np.mean(eta, axis=0)
        sigma_bar = float(np.mean(sigma_eta2))
        loglik_at_mean = float(
            -0.5 * T * np.log(2.0 * np.pi * sigma_bar) - 0.5 * float(np.sum(eta_bar ** 2)) / sigma_bar
        )
        deviance_mean = float(np.mean(-2.0 * loglik))
        p_d = deviance_mean - (-2.0 * loglik_at_mean)
        rows.append(
            {
                "Model": _hierarchy(metadata),
                "Free coefs": free_coefficient_count(slug),
                "RMSE": float(np.mean(rmse)),
                "sigma_eta": float(np.mean(np.sqrt(sigma_eta2))),
                "Mean log-lik": float(np.mean(loglik)),
                "p_D": p_d,
                "Conditional DIC": deviance_mean + p_d,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The cross-equation restriction itself
# ---------------------------------------------------------------------------


def cross_restriction_table(runs: RunSet) -> pd.DataFrame:
    """How far the unrestricted posterior sits from $100\\kappa_N=b_x\\zeta_0\\theta_0$.

    Evaluated inside U, where $\\kappa_N$ and $\\theta_0$ are separately
    sampled. R1 is listed for reference: it satisfies the equality by
    construction, so its row shows where the restriction places the pair.
    """
    rows = []
    for slug in ("hsa_u", "hsa_r1", "hsa_r2", "hsa_r3"):
        if slug not in runs:
            continue
        _, metadata, idata = runs[slug]
        if "kappa_N_empirical" not in idata.posterior or "theta_0" not in idata.posterior:
            continue
        zeta0 = float(metadata.get("zeta0", 6.0))
        b_x = float(metadata.get("marginal_cost_loading", 1.0))
        kappa_n = _flat(idata, "kappa_N_empirical").reshape(-1)
        theta_0 = _flat(idata, "theta_0").reshape(-1)
        implied = b_x * zeta0 * theta_0 / 100.0
        gap = kappa_n - implied
        positive_theta = theta_0 > 1e-9
        implied_zeta = np.full_like(theta_0, np.nan)
        implied_zeta[positive_theta] = 100.0 * kappa_n[positive_theta] / (b_x * theta_0[positive_theta])
        finite_zeta = implied_zeta[np.isfinite(implied_zeta)]
        rows.append(
            {
                "Model": _hierarchy(metadata),
                "kappa_N mean": float(np.mean(kappa_n)),
                "Restriction value": float(np.mean(implied)),
                "Gap mean": float(np.mean(gap)),
                "Gap 5%": float(np.quantile(gap, 0.05)),
                "Gap 95%": float(np.quantile(gap, 0.95)),
                "P(gap>0)": float(np.mean(gap > 0.0)),
                # Only the median and the share above one are reported: with
                # theta_0 near zero the implied-zeta quantiles run to +-30 and
                # would suggest a precision the statistic does not have.
                "Implied zeta median": float(np.median(finite_zeta)) if finite_zeta.size else np.nan,
                "P(zeta>1)": float(np.mean(finite_zeta > 1.0)) if finite_zeta.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_cross_restriction(runs: RunSet, output_path: Path) -> Path | None:
    """U's $(\\theta_0,\\kappa_N)$ cloud against the line the restriction imposes."""
    import matplotlib.pyplot as plt

    if "hsa_u" not in runs:
        return None
    _, metadata, idata = runs["hsa_u"]
    if "kappa_N_empirical" not in idata.posterior or "theta_0" not in idata.posterior:
        return None
    zeta0 = float(metadata.get("zeta0", 6.0))
    b_x = float(metadata.get("marginal_cost_loading", 1.0))
    theta_0 = _flat(idata, "theta_0").reshape(-1)
    kappa_n = _flat(idata, "kappa_N_empirical").reshape(-1)
    step = max(1, theta_0.size // 6000)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.scatter(theta_0[::step], kappa_n[::step], s=5, alpha=0.18, color="tab:blue", label="U posterior draws")
    grid = np.linspace(0.0, float(np.quantile(theta_0, 0.995)), 200)
    ax.plot(
        grid,
        b_x * zeta0 * grid / 100.0,
        color="black",
        lw=1.8,
        label=f"$100\\kappa_N=b_x\\zeta_0\\theta_0$ ($\\zeta_0={zeta0:g}$, $b_x={b_x:g}$)",
    )
    # R1 and R2 can sit almost on top of each other; distinct markers and
    # decreasing z-order keep an overlap visible as an overlap.
    for order, (slug, marker) in enumerate((("hsa_r1", "o"), ("hsa_r2", "s"), ("hsa_r3", "D"))):
        if slug not in runs:
            continue
        _, restricted_meta, restricted = runs[slug]
        if "kappa_N_empirical" not in restricted.posterior:
            continue
        rt = _flat(restricted, "theta_0").reshape(-1)
        rk = _flat(restricted, "kappa_N_empirical").reshape(-1)
        ax.errorbar(
            [float(np.mean(rt))],
            [float(np.mean(rk))],
            xerr=[[float(np.mean(rt) - np.quantile(rt, 0.05))], [float(np.quantile(rt, 0.95) - np.mean(rt))]],
            yerr=[[float(np.mean(rk) - np.quantile(rk, 0.05))], [float(np.quantile(rk, 0.95) - np.mean(rk))]],
            fmt=marker,
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.8,
            color=MODEL_COLORS.get(slug),
            capsize=3,
            zorder=5 - order,
            label=f"{_hierarchy(restricted_meta)} posterior (on the line)",
        )
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\kappa_N$")
    ax.set_title("The cross-equation restriction against the unrestricted posterior")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Competition decomposition and convergence
# ---------------------------------------------------------------------------


def plot_competition_decomposition(runs: RunSet, output_path: Path) -> Path | None:
    """Reference level and cycle, for the models that separate the two."""
    import matplotlib.pyplot as plt

    available = [
        slug for slug in THEORY_MODELS
        if slug in runs and "Nbar" in runs[slug][2].posterior and "Nhat" in runs[slug][2].posterior
    ]
    if not available:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
    for slug in available:
        run_dir, metadata, idata = runs[slug]
        index, _ = load_run_model_data(run_dir)
        for ax, name in zip(axes, ("Nbar", "Nhat")):
            paths = _flat(idata, name)
            x_axis = index if index is not None and len(index) == paths.shape[1] else np.arange(paths.shape[1])
            center = np.median(paths, axis=0)
            lo, hi = np.quantile(paths, [0.05, 0.95], axis=0)
            color = MODEL_COLORS.get(slug)
            ax.plot(x_axis, center, lw=1.5, color=color, label=_hierarchy(metadata))
            ax.fill_between(x_axis, lo, hi, color=color, alpha=0.08)
    axes[0].set_title(r"Moving reference $\bar N_t$ (ten log points from $N_0$)")
    axes[1].set_title(r"Competition cycle $\hat N_t$")
    for ax in axes:
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.4)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=min(4, len(available)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


_CONVERGENCE_CACHE: dict[tuple[str, ...], list[dict[str, Any]]] = {}


def _convergence_records(runs: RunSet) -> list[dict[str, Any]]:
    """Numeric convergence records, computed once per run set.

    The ESS scan over the latent paths is the slow part of the build, and both
    the table and the result macros need it, so the result is cached on the run
    directories rather than recomputed.
    """
    import arviz as az

    key = tuple(str(runs[slug][0]) for slug in sorted(runs))
    if key in _CONVERGENCE_CACHE:
        return _CONVERGENCE_CACHE[key]

    rows = []
    for slug in THEORY_MODELS:
        if slug not in runs:
            continue
        _, metadata, idata = runs[slug]
        posterior = idata.posterior
        # Constants (zeta0 under a fixed-zeta design), deterministic transforms,
        # and sampler telemetry are not sampled parameters; scanning them would
        # report a NaN R-hat as if a parameter had failed.
        scalars = [
            name for name in posterior.data_vars
            if posterior[name].ndim == 2 and name not in DERIVED_OR_TELEMETRY
            and not name.startswith("pg_")
        ]
        paths = [name for name in posterior.data_vars if posterior[name].ndim > 2]
        summary = az.summary(idata, var_names=scalars, kind="diagnostics")
        rows.append(
            {
                "Model": _hierarchy(metadata),
                "scalar_ess": float(summary["ess_bulk"].min()),
                "scalar_rhat": float(summary["r_hat"].max()),
                "worst_scalar": str(summary["ess_bulk"].idxmin()),
                "path_ess": (
                    min(float(np.nanmin(np.asarray(az.ess(idata, var_names=[name])[name]))) for name in paths)
                    if paths else np.nan
                ),
                "path_rhat": (
                    max(float(np.nanmax(np.asarray(az.rhat(idata, var_names=[name])[name]))) for name in paths)
                    if paths else np.nan
                ),
                "status": str(metadata.get("convergence_status", "pending_diagnostics")),
            }
        )
    _CONVERGENCE_CACHE[key] = rows
    return rows


def convergence_table(runs: RunSet) -> pd.DataFrame:
    """Worst-case R-hat and ESS, reported separately for scalars and paths."""
    def _count(value: float) -> str:
        return "--" if not np.isfinite(value) else f"{value:.0f}"

    def _ratio(value: float) -> str:
        return "--" if not np.isfinite(value) else f"{value:.3f}"

    return pd.DataFrame(
        [
            {
                "Model": record["Model"],
                "Scalar min ESS": _count(record["scalar_ess"]),
                "Scalar max R-hat": _ratio(record["scalar_rhat"]),
                "Worst scalar": record["worst_scalar"],
                "Path min ESS": _count(record["path_ess"]),
                "Path max R-hat": _ratio(record["path_rhat"]),
                "Status": record["status"],
            }
            for record in _convergence_records(runs)
        ]
    )


# ---------------------------------------------------------------------------
# Result macros
# ---------------------------------------------------------------------------

# LaTeX control sequences cannot contain digits, so the hierarchy labels and
# subscripts are spelled out.
_MACRO_MODEL = {"F0": "FZero", "U": "U", "R1": "ROne", "R2": "RTwo", "R3": "RThree"}
_MACRO_PARAMETER = {
    "kappa_0": "KappaZero",
    "kappa_N_empirical": "KappaN",
    "d_kappa_d_logN": "DKappaDLogN",
    "theta_0": "ThetaZero",
    "gamma": "Gamma",
}


def _as_quarter(value: Any) -> str:
    try:
        return str(pd.Period(pd.Timestamp(str(value)), freq="Q"))
    except Exception:
        return str(value)


def _macro(name: str, value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        rendered = str(int(value))
    elif isinstance(value, float) and not np.isfinite(value):
        rendered = "n/a"
    elif isinstance(value, float):
        rendered = f"{value:.{digits}f}"
    else:
        rendered = str(value)
    return "\\newcommand{\\" + name + "}{" + rendered + "}\n"


def write_result_macros(runs: RunSet, path: Path) -> Path:
    """Emit every number the report's prose quotes, so none is typed by hand."""
    lines: list[str] = [
        "% Generated by nkpc_hsa.reporting.theory_evidence.write_result_macros.\n",
        "% Do not edit; do not type a number into the report body by hand.\n",
    ]
    if runs:
        _, baseline, _ = next(iter(runs.values()))
        lines += [
            _macro("TheorySampleStart", _as_quarter(baseline.get("sample_start"))),
            _macro("TheorySampleEnd", _as_quarter(baseline.get("sample_end"))),
            _macro("TheoryObservationCount", int(baseline.get("n_obs", 0))),
            _macro("TheoryZetaZero", float(baseline.get("zeta0", np.nan)), digits=2),
            _macro("TheoryMuZero", float(baseline.get("mu0", np.nan)), digits=2),
            _macro("TheoryMarginalCostLoading", float(baseline.get("marginal_cost_loading", np.nan)), digits=2),
            _macro("TheoryNZeroAnchor", float(baseline.get("N0_anchor") or np.nan), digits=3),
            _macro("TheoryPriorSpec", str(baseline.get("prior_spec", ""))),
            _macro("TheoryCodeRevision", str(baseline.get("code_revision", ""))[:12]),
        ]

    for slug in THEORY_MODELS:
        if slug not in runs:
            continue
        _, metadata, idata = runs[slug]
        token = _MACRO_MODEL.get(_hierarchy(metadata))
        if token is None:
            continue
        for parameter, parameter_token in _MACRO_PARAMETER.items():
            if parameter not in idata.posterior:
                continue
            values = _flat(idata, parameter).reshape(-1)
            stem = f"Theory{token}{parameter_token}"
            lines += [
                _macro(stem, float(np.mean(values))),
                _macro(stem + "Lo", float(np.quantile(values, 0.025))),
                _macro(stem + "Hi", float(np.quantile(values, 0.975))),
            ]
        lines.append(_macro(f"Theory{token}StoredDraws", int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])))
        sampling = dict(metadata.get("admissibility_sampling", {}) or {})
        proposals = max(1, int(metadata.get("chains", 1)) * int(metadata.get("n_iter", 0)))
        lines += [
            _macro(
                f"Theory{token}StateRejectPercent",
                100.0 * float(sampling.get("state_path_rejection_share", 0.0) or 0.0),
                digits=1,
            ),
            _macro(
                f"Theory{token}ProbePercent",
                100.0 * float(sampling.get("coefficient_rejections", 0) or 0) / proposals,
                digits=1,
            ),
        ]

    cross = cross_restriction_table(runs)
    for _, row in cross.iterrows():
        token = _MACRO_MODEL.get(str(row["Model"]))
        if token is None:
            continue
        lines += [
            _macro(f"Theory{token}RestrictionValue", float(row["Restriction value"])),
            _macro(f"Theory{token}GapMean", float(row["Gap mean"])),
            _macro(f"Theory{token}GapLo", float(row["Gap 5%"])),
            _macro(f"Theory{token}GapHi", float(row["Gap 95%"])),
            _macro(f"Theory{token}ProbGapPositive", float(row["P(gap>0)"]), digits=2),
            _macro(f"Theory{token}ImpliedZetaMedian", float(row["Implied zeta median"]), digits=2),
            _macro(f"Theory{token}ProbZetaAboveOne", float(row["P(zeta>1)"]), digits=2),
        ]

    fit = conditional_fit_table(runs)
    for _, row in fit.iterrows():
        token = _MACRO_MODEL.get(str(row["Model"]))
        if token is None:
            continue
        lines += [
            _macro(f"Theory{token}Rmse", float(row["RMSE"]), digits=3),
            _macro(f"Theory{token}SigmaEta", float(row["sigma_eta"]), digits=3),
            _macro(f"Theory{token}ConditionalDIC", float(row["Conditional DIC"]), digits=1),
            _macro(f"Theory{token}FreeCoefficients", int(row["Free coefs"])),
        ]

    convergence = _convergence_records(runs)
    if convergence:
        lines += [
            _macro("TheoryWorstScalarESS", float(min(r["scalar_ess"] for r in convergence)), digits=0),
            _macro("TheoryWorstScalarRhat", float(max(r["scalar_rhat"] for r in convergence)), digits=3),
            _macro("TheoryWorstPathESS", float(np.nanmin([r["path_ess"] for r in convergence])), digits=0),
            _macro("TheoryWorstPathRhat", float(np.nanmax([r["path_rhat"] for r in convergence])), digits=3),
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    return path


def plot_prior_vs_posterior(runs: RunSet, output_path: Path) -> Path | None:
    """Every restriction-relevant coefficient against the prior it was drawn under."""
    import matplotlib.pyplot as plt

    priors = _shared_priors(runs)
    available = [slug for slug in THEORY_MODELS if slug in runs]
    if not available or not priors:
        return None
    panels = [
        (prior_key, posterior_key, label)
        for prior_key, posterior_key, label in PRIOR_POSTERIOR_PARAMETERS
        if isinstance(priors.get(prior_key), (list, tuple))
        and any(posterior_key in runs[slug][2].posterior for slug in available)
    ]
    if not panels:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = 3
    rows_count = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(11.4, 3.1 * rows_count))
    flat_axes = np.atleast_1d(axes).reshape(-1)
    for ax, (prior_key, posterior_key, label) in zip(flat_axes, panels):
        mean, sd = float(priors[prior_key][0]), float(priors[prior_key][1])
        drawn: list[np.ndarray] = []
        for slug in available:
            _, metadata, idata = runs[slug]
            if posterior_key not in idata.posterior:
                continue
            values = _flat(idata, posterior_key).reshape(-1)
            drawn.append(values)
            ax.hist(
                values,
                bins=70,
                density=True,
                histtype="step",
                lw=1.4,
                color=MODEL_COLORS.get(slug),
                label=_hierarchy(metadata),
            )
        span = np.concatenate(drawn) if drawn else np.array([mean - 3 * sd, mean + 3 * sd])
        lo = min(float(np.quantile(span, 0.001)), mean - 3.0 * sd)
        hi = max(float(np.quantile(span, 0.999)), mean + 3.0 * sd)
        grid = np.linspace(lo, hi, 400)
        density = np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))
        ax.plot(grid, density, color="black", lw=1.6, ls="--", label="prior")
        ax.set_title(label)
        ax.set_yticks([])
        ax.grid(alpha=0.2)
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=MODEL_COLORS.get(slug), lw=1.4, label=_hierarchy(runs[slug][1]))
        for slug in available
    ] + [Line2D([], [], color="black", lw=1.6, ls="--", label="prior")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), frameon=False)
    fig.suptitle("Prior against posterior, every model and restriction-relevant coefficient")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path
