from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, norm

from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, model_sample_index
from nkpc_hsa.reporting.cpi_ppi_spec import (
    annual_q4_run_keys,
    INFLATION_SPECS,
    MODEL_LABELS,
    MODEL_ORDER,
    PRIMARY_SPECS,
    PRIOR_ORDER,
    report_run_keys,
)


OUT_TABLES = RESULTS_DIR / "tables"
OUT_FIGURES = RESULTS_DIR / "figures"
RHAT_LIMIT = 1.01
ESS_LIMIT = 400.0
# Minimum effective number of draws contributing to the Rao-Blackwellised
# Savage-Dickey ordinate before it is preferred over the kernel estimate.
RB_MIN_EFFECTIVE_DRAWS = 50.0

# ---------------------------------------------------------------------------
# Convergence diagnostics groups.
#
# The report's acceptance rule is max Rhat <= 1.01 and min bulk ESS >= 400.
# It is applied to three *separately reported* groups so that a marginally
# elevated Rhat on a raw latent-state level does not silently reclassify an
# otherwise well-mixed coefficient block, and so that the report can never
# claim "converged" on the strength of a coefficient-only check.
#
#   scalar  -> every stored scalar parameter (coefficients, AR terms, drift,
#              simultaneity loading and every estimated variance)
#   state   -> the raw latent firm-count paths
#   derived -> the economically relevant time-varying coefficient paths
#
# The dagger mark in the coefficient tables remains the *scalar* rule; the
# state and derived rules are reported alongside it and drive the explicit
# "joint" column.
# ---------------------------------------------------------------------------
SCALAR_PARAMETERS = [
    "alpha",
    "kappa",
    "kappa_0",
    "delta",
    "theta",
    "theta_0",
    "gamma",
    "rho_1",
    "rho_2",
    "n",
    "phi_1",
    "lambda_ez",
    "sigma_e",
    "sigma_eta",
    "sigma_zeta",
    "sigma_u",
    "sigma_eps",
    "sigma_N",
]
STATE_PATH_PARAMETERS = ["Nbar", "Nhat"]
DERIVED_PATH_PARAMETERS = ["kappa_t", "theta_t"]

SAMPLER_LABELS = {
    "particle_gibbs": "Particle Gibbs",
    "joint_ffbs": "joint FFBS",
    "alternating_ffbs": "alternating FFBS",
    "ffbs": "FFBS",
    "conjugate": "conjugate Gibbs",
}


def _run_key(path: Path, idata) -> tuple[str, str]:
    attrs = getattr(idata, "attrs", {})
    return str(attrs.get("run_id", "")), path.parent.name


_FALLBACK_STATE_SAMPLER = {
    "ces": "conjugate",
    "hsa_steady": "joint_ffbs",
    "hsa_dynamic": "joint_ffbs",
    "hsa_const_theta": "alternating_ffbs",
    "hsa_full": "alternating_ffbs",
}


def _resolve_state_sampler(run_dir: Path, model: str) -> str:
    """Sampler that drew the latent-state block, for table notes and the manifest.

    Newer samplers declare ``state_sampler`` in their model metadata. Runs saved
    before that declaration existed fall back to the per-model default, which is
    what those runs actually used.
    """
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        for chain in (meta.get("extra", {}) or {}).get("chains", []) or []:
            declared = (chain.get("model_metadata", {}) or {}).get("state_sampler")
            if declared:
                return str(declared)
    return _FALLBACK_STATE_SAMPLER.get(model, "unknown")


def _load_runs(
    runs_dir: Path,
    *,
    min_iter: int,
    competition_frequency: str,
) -> dict[tuple[str, str, str], tuple[Path, object]]:
    selected: dict[tuple[str, str, str], tuple[Path, object]] = {}
    for posterior in sorted(runs_dir.glob("*/posterior.nc")):
        idata = az.from_netcdf(posterior)
        attrs = getattr(idata, "attrs", {})
        if str(attrs.get("estimation_revision", "")) != ESTIMATION_REVISION:
            continue
        if int(attrs.get("n_iter", 0) or 0) < min_iter:
            continue
        if str(attrs.get("period", "full") or "full") != "full":
            continue
        if str(attrs.get("constraint_spec", "unrestricted") or "unrestricted") != "unrestricted":
            continue
        if str(attrs.get("competition_measurement_frequency", "quarterly_interpolated")) != competition_frequency:
            continue
        if str(attrs.get("n_transform", "")) != "log100_centered10":
            continue
        model = str(attrs.get("model", ""))
        data_spec = str(attrs.get("data_spec", ""))
        prior = str(attrs.get("prior_spec", "baseline") or "baseline")
        if model not in MODEL_ORDER:
            continue
        priors_path = posterior.parent / "priors.json"
        if priors_path.exists():
            idata.attrs["run_priors"] = json.loads(priors_path.read_text(encoding="utf-8"))
        idata.attrs["state_sampler"] = _resolve_state_sampler(posterior.parent, model)
        spec_path = posterior.parent / "data_spec.json"
        if spec_path.exists():
            idata.attrs["run_data_spec"] = json.loads(spec_path.read_text(encoding="utf-8"))
        key = (model, data_spec, prior)
        current = selected.get(key)
        if current is None or _run_key(posterior, idata) >= _run_key(current[0], current[1]):
            selected[key] = (posterior, idata)
    return selected


def _sampler_label(idata) -> str:
    sampler = str(getattr(idata, "attrs", {}).get("state_sampler", "unknown"))
    return SAMPLER_LABELS.get(sampler, sampler)


def load_report_runs(
    *,
    runs_dir: Path | None = None,
    min_iter: int = 1,
    competition_frequency: str = "quarterly_interpolated",
    verbose: bool = False,
) -> dict[tuple[str, str, str], tuple[Path, object]]:
    """The single entry point every report artifact must use to obtain its runs.

    Assembles the authoritative run-set for one observation design from the
    production run directory. ``hsa_full`` is estimated by Particle Gibbs in that
    directory (``run_model`` dispatches to the Particle-Gibbs sampler), so no
    out-of-band merge is needed: an earlier revision routed the PCHIP
    ``hsa_full`` cells through ``results/evidence/runs`` because
    Particle Gibbs was only reachable via a monkeypatch. Any script that builds a
    report table calls this rather than ``_load_runs`` directly.
    """
    runs_dir = RESULTS_DIR / "runs" if runs_dir is None else runs_dir
    runs = _load_runs(runs_dir, min_iter=min_iter, competition_frequency=competition_frequency)
    if verbose:
        samplers = sorted({_sampler_label(d) for (m, _, _), (_, d) in runs.items() if m == "hsa_full"})
        print(f"  {competition_frequency}: hsa_full sampler = {' / '.join(samplers) or 'n/a'}")
    return runs


def assert_expected_sampler(runs, *, model: str, expected: str, label: str) -> None:
    """Fail the build if a model's cells were not drawn by the intended sampler.

    ``hsa_full`` is non-linear-Gaussian in the joint state, so it must be
    Particle Gibbs; a cell silently falling back to the superseded alternating
    FFBS would make the reported numbers incomparable across the table.
    """
    offenders = sorted(
        "/".join(key)
        for key, (_, idata) in runs.items()
        if key[0] == model and str(idata.attrs.get("state_sampler", "")) != expected
    )
    if offenders:
        raise SystemExit(
            f"{label}: {len(offenders)} {model} cell(s) are not {expected} "
            f"({', '.join(offenders[:6])}). Re-run scripts/rerun_hsa_full_particle_gibbs.py."
        )


def assert_single_sampler_per_cell(*run_sets: dict[tuple[str, str, str], tuple[Path, object]]) -> None:
    """Fail loudly if one (model, spec, prior) cell is reported under two samplers.

    This is the guard against the failure mode where one table is regenerated
    from the Particle-Gibbs runs while another still shows the alternating-FFBS
    numbers for the same cell.
    """
    seen: dict[tuple[str, str, str], set[str]] = {}
    for runs in run_sets:
        for key, (_, idata) in runs.items():
            seen.setdefault(key, set()).add(str(idata.attrs.get("state_sampler", "unknown")))
    clashes = {key: samplers for key, samplers in seen.items() if len(samplers) > 1}
    if clashes:
        detail = "; ".join(f"{'/'.join(k)}: {sorted(v)}" for k, v in sorted(clashes.items()))
        raise SystemExit(f"Same cell reported under multiple samplers: {detail}")


def _draws(idata, parameter: str) -> np.ndarray | None:
    if parameter not in idata.posterior:
        return None
    values = np.asarray(idata.posterior[parameter], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return values if values.size else None


def _summary(idata, parameter: str) -> dict[str, float] | None:
    values = _draws(idata, parameter)
    if values is None:
        return None
    return {
        "mean": float(np.mean(values)),
        "lo": float(np.quantile(values, 0.025)),
        "hi": float(np.quantile(values, 0.975)),
        "p_gt_0": float(np.mean(values > 0.0)),
    }


def _path_summary(idata, parameter: str) -> dict[str, float] | None:
    if parameter not in idata.posterior:
        return None
    values = np.asarray(idata.posterior[parameter], dtype=float)
    if values.ndim < 3:
        return None
    paths = values.reshape(-1, values.shape[-1])
    return {
        "start": float(np.nanmean(paths[:, 0])),
        "end": float(np.nanmean(paths[:, -1])),
    }


_MODEL_READY: pd.DataFrame | None = None

# Coefficient-block layout of each model's Gaussian Gibbs update. The tuple is the
# ordered regressor names; `_design_columns` turns them into the actual X matrix.
_BETA_BLOCK: dict[str, tuple[str, ...]] = {
    "hsa_steady": ("a", "x", "x_Nbar"),
    "hsa_const_theta": ("a", "x", "x_Nbar", "neg_Nhat"),
    "hsa_full": ("a", "x", "x_Nbar", "neg_Nhat", "neg_Nhat_Nbar"),
}
_BETA_PRIOR_KEYS: dict[str, tuple[str, ...]] = {
    "hsa_steady": ("alpha", "kappa_0", "delta"),
    "hsa_const_theta": ("alpha", "kappa_0", "delta", "theta"),
    "hsa_full": ("alpha", "kappa_0", "delta", "theta_0", "gamma"),
}


def _model_ready() -> pd.DataFrame:
    global _MODEL_READY
    if _MODEL_READY is None:
        _MODEL_READY = pd.read_csv(
            DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]
        ).set_index("DATE")
    return _MODEL_READY


def _run_series(idata) -> dict[str, np.ndarray] | None:
    """The estimation sample for this run, rebuilt from its saved data spec."""
    spec = getattr(idata, "attrs", {}).get("run_data_spec")
    if not isinstance(spec, dict):
        return None
    cols = {
        "pi": spec.get("pi_col"), "pi_prev": spec.get("pi_prev_col"),
        "pi_expect": spec.get("pi_expect_col"), "x": spec.get("x_col"),
        "x_prev": spec.get("x_prev_col"), "N": spec.get("n_col"),
    }
    if any(c is None for c in cols.values()):
        return None
    data = _model_ready()
    needed = [c for c in cols.values() if c in data.columns]
    if len(needed) < 6:
        return None
    sample_index = model_sample_index(data, spec)
    if sample_index is None:
        return None
    sample = data.loc[sample_index, needed]
    expected = int(getattr(idata, "attrs", {}).get("n_obs", 0) or 0)
    if expected and len(sample) != expected:
        raise ValueError(
            f"Saved posterior has T={expected}, but its saved data specification "
            f"selects T={len(sample)} from model_ready.csv. Re-estimate the cell."
        )
    return {
        "y": (sample[cols["pi"]] - sample[cols["pi_expect"]]).to_numpy(float),
        "a": (sample[cols["pi_prev"]] - sample[cols["pi_expect"]]).to_numpy(float),
        "x": sample[cols["x"]].to_numpy(float),
        "x_prev": sample[cols["x_prev"]].to_numpy(float),
    }


def _sigma_eta2_draws(idata) -> np.ndarray | None:
    """Inflation-equation residual variance, however the model stored it."""
    if "sigma_eta" in idata.posterior:
        return np.asarray(idata.posterior["sigma_eta"], dtype=float).reshape(-1) ** 2
    if "sigma_e" in idata.posterior and "sigma_zeta" in idata.posterior and "lambda_ez" in idata.posterior:
        se = np.asarray(idata.posterior["sigma_e"], dtype=float).reshape(-1) ** 2
        sz = np.asarray(idata.posterior["sigma_zeta"], dtype=float).reshape(-1) ** 2
        lam = np.asarray(idata.posterior["lambda_ez"], dtype=float).reshape(-1)
        return np.maximum(se - lam**2 * sz, 1e-12)
    return None


def _rao_blackwell_terms(idata, parameter: str) -> np.ndarray | None:
    """Per-draw mixture components of the Rao-Blackwellised ordinate at zero.

    Split out from ``_rao_blackwell_ordinate`` so the report can also describe
    how concentrated the average is, which is what justifies preferring it to a
    kernel estimate. Returns ``None`` whenever the model or the run is missing
    something the conditional needs.
    """
    return _rao_blackwell_ordinate(idata, parameter, _return_terms=True)


def _rao_blackwell_ordinate(idata, parameter: str, _return_terms: bool = False):
    """Posterior density of ``parameter`` at zero, without density estimation.

    The Gibbs update for the inflation coefficients is an exact Gaussian
    conditional, so the marginal posterior ordinate is the Rao-Blackwellised
    average of that conditional over the retained draws of everything else:

        p(delta = 0 | y) = E[ p(delta = 0 | psi, y) ],   psi ~ p(psi | y)

    Each term is an exact normal density, so the estimator stays accurate where a
    kernel density estimate does not: zero sits several posterior standard
    deviations outside the sampled range of delta, and a Gaussian KDE there is
    extrapolating its own tail rather than measuring the posterior.
    """
    model = str(getattr(idata, "attrs", {}).get("model", ""))
    block = _BETA_BLOCK.get(model)
    names = _BETA_PRIOR_KEYS.get(model)
    if block is None or names is None or parameter not in names:
        return None
    series = _run_series(idata)
    if series is None:
        return None
    priors = getattr(idata, "attrs", {}).get("run_priors", {})
    if not isinstance(priors, dict):
        return None

    prior_mean, prior_var = [], []
    for name in names:
        spec = priors.get(name)
        if spec is None and name == "theta":
            spec = priors.get("theta_0")
        if spec is None:
            return None
        mean, sd = (float(spec["mean"]), float(spec["sd"])) if isinstance(spec, dict) else (float(spec[0]), float(spec[1]))
        prior_mean.append(mean)
        prior_var.append(sd**2)
    prior_mean = np.asarray(prior_mean)
    V0_inv = np.diag(1.0 / np.asarray(prior_var))
    index = names.index(parameter)

    post = idata.posterior
    need = {"Nbar", "Nhat", "lambda_ez", "phi_1"}
    if not need.issubset(set(post.data_vars)):
        return None
    T = series["y"].size
    Nbar = np.asarray(post["Nbar"], dtype=float).reshape(-1, T)
    Nhat = np.asarray(post["Nhat"], dtype=float).reshape(-1, T)
    lam = np.asarray(post["lambda_ez"], dtype=float).reshape(-1)
    phi = np.asarray(post["phi_1"], dtype=float).reshape(-1)
    sigma_eta2 = _sigma_eta2_draws(idata)
    if sigma_eta2 is None or len(sigma_eta2) != len(lam):
        return None

    y, a, x, x_prev = series["y"], series["a"], series["x"], series["x_prev"]
    terms = np.empty(len(lam))
    for s in range(len(lam)):
        nb, nh = Nbar[s], Nhat[s]
        cols = {
            "a": a, "x": x, "x_Nbar": x * nb,
            "neg_Nhat": -nh, "neg_Nhat_Nbar": -(nh * nb),
        }
        X = np.column_stack([cols[c] for c in block])
        y_adj = y - lam[s] * (x - phi[s] * x_prev)
        V1 = np.linalg.inv(X.T @ X / sigma_eta2[s] + V0_inv)
        b1 = V1 @ (X.T @ y_adj / sigma_eta2[s] + V0_inv @ prior_mean)
        terms[s] = float(norm.pdf(0.0, loc=b1[index], scale=np.sqrt(V1[index, index])))

    if _return_terms:
        return terms
    total = float(terms.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    # Reliability guard. The average is only meaningful if many draws contribute:
    # if a handful of mixture components carry it, the estimator has the same
    # far-tail problem as a kernel density estimate and should not be trusted.
    weights = terms / total
    effective = 1.0 / float(np.sum(weights**2))
    if effective < RB_MIN_EFFECTIVE_DRAWS:
        return None
    return total / len(terms)


def _bf10(idata, parameter: str) -> float | None:
    """Savage-Dickey Bayes factor against ``parameter`` = 0.

    The prior ordinate is exact. The posterior ordinate is Rao-Blackwellised
    where the model's coefficient block allows it, and falls back to a Gaussian
    kernel density estimate otherwise -- see ``_rao_blackwell_ordinate`` for why
    the KDE is unreliable at this evaluation point.
    """
    values = _draws(idata, parameter)
    if values is None or values.size < 20 or np.std(values, ddof=1) <= 0:
        return None
    priors = idata.attrs.get("run_priors", {})
    prior = priors.get(parameter)
    if prior is None:
        return None
    if isinstance(prior, dict):
        prior_mean, prior_sd = float(prior["mean"]), float(prior["sd"])
    else:
        prior_mean, prior_sd = float(prior[0]), float(prior[1])
    posterior_at_zero = _rao_blackwell_ordinate(idata, parameter)
    if posterior_at_zero is None or not np.isfinite(posterior_at_zero) or posterior_at_zero <= 0.0:
        posterior_at_zero = float(gaussian_kde(values)([0.0])[0])
    prior_at_zero = float(norm.pdf(0.0, loc=prior_mean, scale=prior_sd))
    bf01 = posterior_at_zero / max(prior_at_zero, 1e-300)
    return float(1.0 / max(bf01, 1e-300))


def _fmt(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "--"
    return f"{summary['mean']:+.3f} [{summary['lo']:+.3f}, {summary['hi']:+.3f}]"


def _fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    if value >= 1000:
        return ">999"
    return f"{value:.{digits}f}"


def _group_diagnostics(idata, names: list[str]) -> dict[str, float | bool | str | None]:
    """max Rhat / min bulk ESS over one group of stored quantities.

    Variables that are constant across draws (for example a coefficient a model
    restricts to zero) produce a non-finite Rhat; they carry no mixing
    information and are skipped rather than poisoning the max/min.
    """
    max_rhat = -np.inf
    min_ess = np.inf
    worst_rhat_name: str | None = None
    worst_ess_name: str | None = None
    checked: list[str] = []
    for name in names:
        if name not in idata.posterior:
            continue
        values = np.asarray(idata.posterior[name], dtype=float)
        if not np.isfinite(values).any() or float(np.nanstd(values)) <= 0.0:
            continue
        rhat = float(np.nanmax(np.asarray(az.rhat(idata.posterior[name]), dtype=float)))
        ess = float(np.nanmin(np.asarray(az.ess(idata.posterior[name], method="bulk"), dtype=float)))
        checked.append(name)
        if np.isfinite(rhat) and rhat > max_rhat:
            max_rhat, worst_rhat_name = rhat, name
        if np.isfinite(ess) and ess < min_ess:
            min_ess, worst_ess_name = ess, name
    if not checked:
        return {
            "max_rhat": np.nan,
            "min_ess": np.nan,
            "converged": None,
            "worst_rhat": None,
            "worst_ess": None,
            "n_checked": 0,
        }
    converged = bool(
        np.isfinite(max_rhat)
        and np.isfinite(min_ess)
        and max_rhat <= RHAT_LIMIT
        and min_ess >= ESS_LIMIT
    )
    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "converged": converged,
        "worst_rhat": worst_rhat_name,
        "worst_ess": worst_ess_name,
        "n_checked": len(checked),
    }


def _diagnostics(idata) -> dict[str, object]:
    """Grouped convergence diagnostics.

    Three groups are computed and reported separately:

    ``scalar``  every stored scalar parameter -- coefficients, the AR(2) block,
                the trend drift ``n``, the simultaneity loading and every
                estimated variance.
    ``state``   the raw latent firm-count paths ``Nbar_t`` and ``Nhat_t``.
    ``derived`` the economically relevant time-varying coefficient paths
                ``kappa_t`` and (where estimated) ``theta_t``.

    ``converged`` is the *scalar* rule and is what the dagger mark in the
    coefficient tables means. ``state_converged`` / ``derived_converged`` /
    ``joint_converged`` are exposed so the report can state which of the three
    it is asserting rather than implying all of them.
    """
    scalar = _group_diagnostics(idata, SCALAR_PARAMETERS)
    state = _group_diagnostics(idata, STATE_PATH_PARAMETERS)
    derived = _group_diagnostics(idata, DERIVED_PATH_PARAMETERS)
    groups = [scalar["converged"], state["converged"], derived["converged"]]
    present = [flag for flag in groups if flag is not None]
    return {
        "scalar": scalar,
        "state": state,
        "derived": derived,
        # Backwards-compatible keys: the dagger rule stays coefficient-based.
        "max_rhat": scalar["max_rhat"],
        "min_ess": scalar["min_ess"],
        "converged": bool(scalar["converged"]),
        "state_converged": state["converged"],
        "derived_converged": derived["converged"],
        "joint_converged": bool(present) and all(present),
        "has_states": state["converged"] is not None,
    }


def _marked(value: str, converged: bool) -> str:
    if value == "--":
        return value
    return value if converged else value + r"\textsuperscript{$\dagger$}"


def _conv_status(diagnostics: dict[str, object], *, japanese: bool = False) -> str:
    """Short status string distinguishing coefficient from joint convergence.

    ``japanese`` is retained only so older call sites keep working; the report is
    English-only and the tables are written in English at source.
    """
    watch = "要注意" if japanese else "watch"
    if not bool(diagnostics["converged"]):
        return watch
    if diagnostics["has_states"] and not bool(diagnostics["joint_converged"]):
        return "OK (coef)"
    return "OK"


def _has_cjk(text: str) -> bool:
    return any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" for ch in text)


# Coefficient names as the reader meets them in the equations. Tables used to
# emit the bare identifier inside math mode -- "$kappa$" is k*a*p*p*a, five
# italic variables multiplied together, which TeX typesets without complaint. The
# body text said kappa_t = kappa_0 + delta*Nbar_t in symbols while the tables
# under it said "kappa" and "delta" in words.
_SYMBOL = {
    "kappa": r"$\kappa$",
    "kappa_0": r"$\kappa_0$",
    "kappa_t": r"$\kappa_t$",
    "delta": r"$\delta$",
    "theta": r"$\theta$",
    "theta_0": r"$\theta_0$",
    "theta_t": r"$\theta_t$",
    "gamma": r"$\gamma$",
    "alpha": r"$\alpha$",
    "n": "$n$",
}
# Column headings, which are phrases rather than bare names.
_COLUMN_SYMBOL = {
    "delta": r"$\delta$",
    "gamma": r"$\gamma$",
    "BF10": r"$\mathrm{BF}_{10}$",
    "BF10(delta)": r"$\mathrm{BF}_{10}(\delta)$",
    "kappa path": r"$\kappa_t$ path",
    "CES kappa": r"CES $\kappa$",
    "HSA kappa": r"HSA $\kappa$",
    "max Rhat": r"max $\hat R$",
    "state max Rhat": r"state max $\hat R$",
    "path max Rhat": r"path max $\hat R$",
}


def _symbol(name: str) -> str:
    """LaTeX for a coefficient name, for use inside a table cell."""
    return _SYMBOL.get(name, name)


def _write_latex(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    display = df.reindex(columns=columns).copy()
    display.columns = [_COLUMN_SYMBOL.get(c, c) for c in display.columns]
    display.to_latex(
        OUT_TABLES / f"{name}.tex",
        index=False,
        escape=False,
        na_rep="--",
        column_format="l" * len(columns),
    )
    df.to_csv(OUT_TABLES / f"{name}.csv", index=False)
    # The report is English-only and there is no longer a translation pass, so a
    # Japanese label reaching a table would land straight in the PDF.
    written = (OUT_TABLES / f"{name}.tex").read_text(encoding="utf-8")
    if _has_cjk(written):
        raise SystemExit(f"{name}.tex contains CJK text; tables must be written in English.")


def build_hsa_steady_activity_table(runs) -> pd.DataFrame:
    rows = []
    for inflation, activity_specs in INFLATION_SPECS.items():
        for activity, data_spec in activity_specs.items():
            item = runs.get(("hsa_steady", data_spec, "baseline"))
            if item is None:
                continue
            _, idata = item
            diagnostics = _diagnostics(idata)
            delta = _summary(idata, "delta")
            path = _path_summary(idata, "kappa_t") or {}
            rows.append(
                {
                    "inflation": inflation,
                    "activity": activity,
                    "delta": _marked(_fmt(delta), bool(diagnostics["converged"])),
                    "BF10": _fmt_num(_bf10(idata, "delta")),
                    "kappa_start": path.get("start", np.nan),
                    "kappa_end": path.get("end", np.nan),
                    "delta_mean": np.nan if delta is None else delta["mean"],
                    "delta_lo": np.nan if delta is None else delta["lo"],
                    "delta_hi": np.nan if delta is None else delta["hi"],
                    "converged": bool(diagnostics["converged"]),
                }
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["kappa path"] = table.apply(
            lambda r: _marked(
                f"{r['kappa_start']:+.3f} $\\rightarrow$ {r['kappa_end']:+.3f}",
                bool(r["converged"]),
            ),
            axis=1,
        )
    _write_latex(table, "hsa_steady_by_activity", ["inflation", "activity", "delta", "BF10", "kappa path"])
    return table


def build_model_table(runs) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for inflation, data_spec in PRIMARY_SPECS.items():
            item = runs.get((model, data_spec, "baseline"))
            if item is None:
                continue
            _, idata = item
            diagnostics = _diagnostics(idata)
            slope_name = "kappa" if model in {"ces", "hsa_dynamic"} else "kappa_0"
            entry_name = "theta" if model in {"hsa_dynamic", "hsa_const_theta"} else "theta_0"
            delta = _summary(idata, "delta")
            gamma = _summary(idata, "gamma")
            entry = _summary(idata, entry_name)
            path = _path_summary(idata, "kappa_t")
            rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "inflation": inflation,
                    "slope": _fmt(_summary(idata, slope_name)),
                    "delta": _fmt(delta),
                    "BF10(delta)": _fmt_num(_bf10(idata, "delta")),
                    "entry": _fmt(entry),
                    "gamma": _fmt(gamma),
                    "kappa path": "--" if path is None else f"{path['start']:+.3f} $\\rightarrow$ {path['end']:+.3f}",
                    "diagnostics": _conv_status(diagnostics),
                }
            )
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "unemployment_by_model",
        # No convergence column: the dagger on each value already carries the
        # coefficient rule, and the group-by-group tables carry the rest. A third
        # restatement in every coefficient table only invited reading "OK" as a
        # claim about the whole run.
        ["model", "inflation", "slope", "delta", "BF10(delta)", "entry", "gamma", "kappa path"],
    )
    return table


def build_output_gap_model_tables(runs) -> pd.DataFrame:
    """Write one complete model-comparison table for each output-gap definition."""
    rows: list[dict[str, object]] = []
    for activity in ["HP output gap", "BN output gap"]:
        for model in MODEL_ORDER:
            for inflation, activity_specs in INFLATION_SPECS.items():
                data_spec = activity_specs[activity]
                item = runs.get((model, data_spec, "baseline"))
                if item is None:
                    continue
                _, idata = item
                diagnostics = _diagnostics(idata)
                slope_name = "kappa" if model in {"ces", "hsa_dynamic"} else "kappa_0"
                entry_name = "theta" if model in {"hsa_dynamic", "hsa_const_theta"} else "theta_0"
                path = _path_summary(idata, "kappa_t")
                rows.append(
                    {
                        "activity": activity,
                        "model": MODEL_LABELS[model],
                        "inflation": inflation,
                        "slope": _marked(_fmt(_summary(idata, slope_name)), bool(diagnostics["converged"])),
                        "delta": _marked(_fmt(_summary(idata, "delta")), bool(diagnostics["converged"])),
                        "BF10(delta)": _fmt_num(_bf10(idata, "delta")),
                        "entry": _marked(_fmt(_summary(idata, entry_name)), bool(diagnostics["converged"])),
                        "gamma": _marked(_fmt(_summary(idata, "gamma")), bool(diagnostics["converged"])),
                        "kappa path": "--" if path is None else f"{path['start']:+.3f} $\\rightarrow$ {path['end']:+.3f}",
                        "diagnostics": _conv_status(diagnostics),
                    }
                )
    table = pd.DataFrame(rows)
    columns = ["model", "inflation", "slope", "delta", "BF10(delta)", "entry", "gamma", "kappa path"]
    for activity, filename in [("HP output gap", "output_gap_hp_by_model"), ("BN output gap", "output_gap_bn_by_model")]:
        _write_latex(table.loc[table["activity"] == activity], filename, columns)
    return table


def _paired_difference(lhs: np.ndarray, rhs: np.ndarray, *, seed: int = 81731) -> np.ndarray:
    """Monte Carlo draws for two independently estimated posterior quantities."""
    n = min(lhs.size, rhs.size)
    rng = np.random.default_rng(seed)
    return lhs[:n] - rhs[rng.permutation(rhs.size)[:n]]


def _summarize_draws(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "mean": float(np.mean(finite)),
        "lo": float(np.quantile(finite, 0.025)),
        "hi": float(np.quantile(finite, 0.975)),
        "p_lt_0": float(np.mean(finite < 0.0)),
    }


def _hsa_implied_ovb(idata, data_spec: dict[str, object], data: pd.DataFrame) -> np.ndarray:
    """Slope bias from omitting -theta*Nhat, conditional on a_t and zeta_t."""
    cols = {
        "pi": str(data_spec["pi_col"]),
        "pi_prev": str(data_spec["pi_prev_col"]),
        "pi_expect": str(data_spec["pi_expect_col"]),
        "x": str(data_spec["x_col"]),
        "x_prev": str(data_spec["x_prev_col"]),
        "N": str(data_spec["n_col"]),
    }
    sample_index = model_sample_index(data, data_spec)
    if sample_index is None:
        raise ValueError("Could not reconstruct the estimation sample for the OVB diagnostic.")
    sample = data.loc[sample_index, list(cols.values())]
    x = sample[cols["x"]].to_numpy(dtype=float)
    x_prev = sample[cols["x_prev"]].to_numpy(dtype=float)
    a = (sample[cols["pi_prev"]] - sample[cols["pi_expect"]]).to_numpy(dtype=float)
    nhat = np.asarray(idata.posterior["Nhat"], dtype=float).reshape(-1, len(x))
    theta = _draws(idata, "theta")
    phi = _draws(idata, "phi_1")
    if theta is None or phi is None or nhat.shape[0] != theta.size or theta.size != phi.size:
        raise ValueError("HSA dynamic posterior arrays are not aligned for the OVB diagnostic.")
    out = np.empty(theta.size)
    for s in range(theta.size):
        zeta = x - phi[s] * x_prev
        controls = np.column_stack([a, zeta])
        x_resid = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
        omitted = -theta[s] * nhat[s]
        omitted_resid = omitted - controls @ np.linalg.lstsq(controls, omitted, rcond=None)[0]
        denom = float(x_resid @ x_resid)
        out[s] = float(x_resid @ omitted_resid / denom) if denom > 0 else np.nan
    return out


def build_ces_hsa_bias_table(runs, *, command_prefix: str = "") -> pd.DataFrame:
    """Compare naive CES and HSA-dynamic slopes and the HSA-implied OVB."""
    config = load_model_config(ROOT / "configs" / "models.yaml")
    specs = configured_data_specs(config, list(config.get("data_specs", {})))
    data = pd.read_csv(DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    comparisons = [
        ("Inverse markup (theory)", "inv_markup"),
        ("Headline CPI / negative unemployment gap", PRIMARY_SPECS["Headline CPI"]),
        ("Core CPI / negative unemployment gap", PRIMARY_SPECS["Core CPI"]),
        ("PPI / negative unemployment gap", PRIMARY_SPECS["PPI"]),
    ]
    rows: list[dict[str, object]] = []
    direct_macros: list[str] = ["% Generated by scripts/12_build_cpi_ppi_report.py; do not edit by hand."]
    for label, data_spec_name in comparisons:
        ces_item = runs.get(("ces", data_spec_name, "baseline"))
        hsa_item = runs.get(("hsa_dynamic", data_spec_name, "baseline"))
        if ces_item is None or hsa_item is None:
            continue
        ces_idata, hsa_idata = ces_item[1], hsa_item[1]
        ces_draws = _draws(ces_idata, "kappa")
        hsa_draws = _draws(hsa_idata, "kappa")
        if ces_draws is None or hsa_draws is None:
            continue
        difference = _summarize_draws(_paired_difference(ces_draws, hsa_draws))
        ovb = _summarize_draws(_hsa_implied_ovb(hsa_idata, specs[data_spec_name], data))
        converged = bool(_diagnostics(ces_idata)["converged"] and _diagnostics(hsa_idata)["converged"])
        rows.append(
            {
                "specification": label,
                "CES kappa": _marked(_fmt(_summary(ces_idata, "kappa")), converged),
                "HSA kappa": _marked(_fmt(_summary(hsa_idata, "kappa")), converged),
                "CES-HSA": _marked(_fmt(difference), converged),
                "Pr(CES-HSA<0)": f"{difference['p_lt_0']:.3f}",
                "HSA-implied OVB": _marked(_fmt(ovb), converged),
                "Pr(OVB<0)": f"{ovb['p_lt_0']:.3f}",
                "diagnostics": "OK" if converged else r"watch$^{\dagger}$",
            }
        )
        if data_spec_name == "inv_markup":
            direct_macros.extend(
                [
                    rf"\providecommand{{\{command_prefix}BiasDirectCES}}{{{_fmt(_summary(ces_idata, 'kappa'))}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectHSA}}{{{_fmt(_summary(hsa_idata, 'kappa'))}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectDifference}}{{{_fmt(difference)}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectProbability}}{{{difference['p_lt_0']:.3f}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectOVB}}{{{_fmt(ovb)}}}",
                    rf"\providecommand{{\{command_prefix}BiasDirectOVBProbability}}{{{ovb['p_lt_0']:.3f}}}",
                ]
            )
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "ces_hsa_kappa_bias",
        ["specification", "CES kappa", "HSA kappa", "CES-HSA", "Pr(CES-HSA<0)", "HSA-implied OVB", "Pr(OVB<0)"],
    )
    (OUT_TABLES / "bias_macros.tex").write_text("\n".join(direct_macros) + "\n", encoding="utf-8")
    return table


def build_prior_table(runs) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for inflation, data_spec in PRIMARY_SPECS.items():
            for prior in PRIOR_ORDER:
                item = runs.get((model, data_spec, prior))
                if item is None:
                    continue
                _, idata = item
                diagnostics = _diagnostics(idata)
                parameter = "kappa" if model in {"ces", "hsa_dynamic"} else "delta"
                rows.append(
                    {
                        "model": MODEL_LABELS[model],
                        "inflation": inflation,
                        "prior": prior,
                        "parameter": _symbol(parameter),
                        "posterior": _marked(_fmt(_summary(idata, parameter)), bool(diagnostics["converged"])),
                        "BF10": _fmt_num(_bf10(idata, parameter)),
                        "converged": bool(diagnostics["converged"]),
                    }
                )
    table = pd.DataFrame(rows)
    _write_latex(table, "prior_sensitivity_by_model", ["model", "inflation", "prior", "parameter", "posterior", "BF10"])
    if not table.empty:
        compact = table.pivot(index=["model", "inflation", "parameter"], columns="prior", values="posterior").reset_index()
        for prior in PRIOR_ORDER:
            if prior not in compact:
                compact[prior] = "--"
        _write_latex(
            compact,
            "prior_sensitivity_compact",
            ["model", "inflation", "parameter", "baseline", "weak", "tight"],
        )
        steady = compact[compact["model"] == MODEL_LABELS["hsa_steady"]].copy()
        _write_latex(
            steady,
            "prior_sensitivity_hsa_steady",
            ["inflation", "parameter", "baseline", "weak", "tight"],
        )
    return table


def build_run_manifest(runs) -> pd.DataFrame:
    rows = []
    for (model, data_spec, prior), (path, idata) in sorted(runs.items()):
        attrs = getattr(idata, "attrs", {})
        diagnostics = _diagnostics(idata)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "data spec": r"\texttt{" + data_spec.replace("_", r"\_") + "}",
                "prior": prior,
                "N frequency": str(
                    attrs.get("competition_measurement_frequency", "quarterly_interpolated")
                ).replace("quarterly_interpolated", "PCHIP quarterly").replace("annual_q4", "annual Q4"),
                "T": int(attrs.get("n_obs", 0) or 0),
                "run id": r"\texttt{" + str(attrs.get("run_id", path.parent.name)).replace("_", r"\_") + "}",
                "state sampler": _sampler_label(idata),
                "max Rhat": diagnostics["max_rhat"],
                "min bulk ESS": diagnostics["min_ess"],
                "state max Rhat": diagnostics["state"]["max_rhat"],
                "state min ESS": diagnostics["state"]["min_ess"],
                "path max Rhat": diagnostics["derived"]["max_rhat"],
                "path min ESS": diagnostics["derived"]["min_ess"],
                "converged": bool(diagnostics["converged"]),
                "state converged": diagnostics["state_converged"],
                "joint converged": bool(diagnostics["joint_converged"]),
                "has states": bool(diagnostics["has_states"]),
            }
        )
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "report_run_manifest",
        [
            "model", "data spec", "prior", "N frequency", "T", "run id", "state sampler",
            "max Rhat", "min bulk ESS", "state max Rhat", "state min ESS",
        ],
    )
    if not table.empty:
        summary = (
            table.groupby("model", sort=False)
            .agg(
                runs=("model", "size"),
                warnings=("converged", lambda x: int((~x).sum())),
                **{
                    "state warnings": ("state converged", lambda x: int((x == False).sum())),  # noqa: E712
                    "joint warnings": ("joint converged", lambda x: int((~x).sum())),
                    "sampler": ("state sampler", lambda x: "/".join(sorted(set(x)))),
                    "max Rhat": ("max Rhat", "max"),
                    "min bulk ESS": ("min bulk ESS", "min"),
                },
            )
            .reset_index()
        )
        summary["max Rhat"] = summary["max Rhat"].map(lambda x: f"{x:.3f}")
        summary["min bulk ESS"] = summary["min bulk ESS"].map(lambda x: f"{x:.0f}")
        _write_latex(
            summary,
            "convergence_summary",
            ["model", "sampler", "runs", "warnings", "state warnings", "joint warnings", "max Rhat", "min bulk ESS"],
        )

        warnings = table.loc[~table["converged"]].copy()
        warnings = warnings.sort_values(["max Rhat", "min bulk ESS"], ascending=[False, True])
        warning_view = warnings.head(15).copy()
        warning_view["max Rhat"] = warning_view["max Rhat"].map(lambda x: f"{x:.3f}")
        warning_view["min bulk ESS"] = warning_view["min bulk ESS"].map(lambda x: f"{x:.0f}")
        _write_latex(
            warning_view,
            "convergence_warnings",
            ["model", "data spec", "prior", "max Rhat", "min bulk ESS"],
        )
    return table


def build_frequency_outputs(
    runs,
    *,
    tables_dir: Path,
    figures_dir: Path,
    command_prefix: str,
) -> None:
    """Build the same report artifacts for one competition-frequency design."""
    global OUT_TABLES, OUT_FIGURES
    OUT_TABLES = tables_dir
    OUT_FIGURES = figures_dir
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    hsa_table = build_hsa_steady_activity_table(runs)
    build_model_table(runs)
    build_output_gap_model_tables(runs)
    build_ces_hsa_bias_table(runs, command_prefix=command_prefix)
    build_prior_table(runs)
    build_diagnostics_table(runs)
    build_primary_parameter_state_diagnostics(runs)
    build_group_convergence_diagnostics(runs)
    build_run_manifest(runs)
    build_delta_grid_table(runs)
    write_result_macros(runs, command_prefix=command_prefix)
    save_delta_forest(hsa_table)
    save_output_gap_delta_forest(hsa_table)
    save_additional_parameter_forests(runs)
    save_kappa_path_comparison(runs)
    save_slope_prior_vs_posterior(runs)
    save_prior_vs_posterior_grid(
        runs,
        design_label="mixed-frequency" if command_prefix == "Annual" else "interpolated",
    )
    save_main_competition_decomposition(runs)


def write_result_macros(
    runs,
    *,
    command_prefix: str = "",
    filename: str = "result_macros.tex",
) -> Path:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    diagnostics = [_diagnostics(idata) for _, idata in runs.values()]
    warning_count = sum(not bool(item["converged"]) for item in diagnostics)
    state_warning_count = sum(item["state_converged"] is False for item in diagnostics)
    joint_warning_count = sum(not bool(item["joint_converged"]) for item in diagnostics)
    full_samplers = sorted(
        {_sampler_label(idata) for (model, _, _), (_, idata) in runs.items() if model == "hsa_full"}
    )
    const_theta_samplers = sorted(
        {_sampler_label(idata) for (model, _, _), (_, idata) in runs.items() if model == "hsa_const_theta"}
    )
    lines = [
        "% Generated by scripts/12_build_cpi_ppi_report.py; do not edit by hand.",
        rf"\providecommand{{\{command_prefix}ReportEstimationRevision}}{{\texttt{{{ESTIMATION_REVISION}}}}}",
        rf"\providecommand{{\{command_prefix}ReportRunCount}}{{{len(runs)}}}",
        rf"\providecommand{{\{command_prefix}ReportWarningCount}}{{{warning_count}}}",
        rf"\providecommand{{\{command_prefix}ReportStateWarningCount}}{{{state_warning_count}}}",
        rf"\providecommand{{\{command_prefix}ReportJointWarningCount}}{{{joint_warning_count}}}",
        rf"\providecommand{{\{command_prefix}HsaFullSampler}}{{{' / '.join(full_samplers) or 'n/a'}}}",
        rf"\providecommand{{\{command_prefix}HsaConstThetaSampler}}{{{' / '.join(const_theta_samplers) or 'n/a'}}}",
    ]
    lines.extend(_sample_metadata_macros(runs, command_prefix))
    prefixes = {"Headline CPI": "HeadlineUnemp", "Core CPI": "CoreUnemp", "PPI": "PPIUnemp"}
    for inflation, data_spec in PRIMARY_SPECS.items():
        prefix = prefixes[inflation]
        item = runs.get(("hsa_steady", data_spec, "baseline"))
        if item is None:
            continue
        _, idata = item
        delta = _summary(idata, "delta")
        bf10 = _bf10(idata, "delta")
        path = _path_summary(idata, "kappa_t")
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}Delta}}{{{_fmt(delta)}}}")
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}DeltaBF}}{{{_fmt_num(bf10)}}}")
        if path is not None:
            lines.append(rf"\providecommand{{\{command_prefix}{prefix}KappaStart}}{{{path['start']:+.3f}}}")
            lines.append(rf"\providecommand{{\{command_prefix}{prefix}KappaEnd}}{{{path['end']:+.3f}}}")
    output_prefixes = {
        ("Headline CPI", "HP output gap"): "HeadlineHP",
        ("Core CPI", "HP output gap"): "CoreHP",
        ("PPI", "HP output gap"): "PPIHP",
        ("Headline CPI", "BN output gap"): "HeadlineBN",
        ("Core CPI", "BN output gap"): "CoreBN",
        ("PPI", "BN output gap"): "PPIBN",
    }
    for (inflation, activity), prefix in output_prefixes.items():
        data_spec = INFLATION_SPECS[inflation][activity]
        item = runs.get(("hsa_steady", data_spec, "baseline"))
        if item is None:
            continue
        _, idata = item
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}Delta}}{{{_fmt(_summary(idata, 'delta'))}}}")
        lines.append(rf"\providecommand{{\{command_prefix}{prefix}DeltaBF}}{{{_fmt_num(_bf10(idata, 'delta'))}}}")
    lines.extend(_convergence_macros(runs, command_prefix))
    lines.extend(_state_space_macros(runs, command_prefix))
    lines.extend(_variant_macros(runs, command_prefix))
    lines.extend(_sddr_reliability_macros(runs, command_prefix))
    target = OUT_TABLES / filename
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Prose macros.
#
# Every number the report states in running text comes from one of the builders
# below. Nothing quantitative is typed into the .tex by hand: a re-estimation
# that moves a number moves the prose with it, which is what stopped the body
# text from silently disagreeing with the tables after the UNRATE switch.
# Specification constants (prior scales, the Rhat/ESS thresholds, the KAPPA_SCALE
# rescale) are deliberately NOT macros -- they are inputs, not results, and
# writing them literally is what makes the .tex readable.
# ---------------------------------------------------------------------------


def _sample_metadata(runs) -> dict[str, object]:
    """Sample labels and observation counts from the main run's saved metadata."""
    item = runs.get(("hsa_steady", PRIMARY_SPECS["Core CPI"], "baseline"))
    if item is None:
        return {}
    posterior_path, idata = item
    attrs = getattr(idata, "attrs", {})
    metadata_path = posterior_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    start = pd.Timestamp(str(attrs.get("sample_start", metadata.get("sample_start", "")))).to_period("Q")
    end = pd.Timestamp(str(attrs.get("sample_end", metadata.get("sample_end", "")))).to_period("Q")
    n_obs = int(attrs.get("n_obs", metadata.get("n_obs", 0)) or 0)
    competition = dict(metadata.get("competition_measurement", {}) or {})
    return {
        "start": str(start),
        "end": str(end),
        "start_year": int(start.year),
        "end_year": int(end.year),
        "n_obs": n_obs,
        "firm_obs": int(competition.get("finite_N_obs_count", n_obs) or 0),
        "firm_missing": int(competition.get("missing_N_obs_count", 0) or 0),
    }


def _sample_metadata_macros(runs, prefix: str) -> list[str]:
    sample = _sample_metadata(runs)
    if not sample:
        return []
    return [
        rf"\providecommand{{\{prefix}ReportSampleStart}}{{{sample['start']}}}",
        rf"\providecommand{{\{prefix}ReportSampleEnd}}{{{sample['end']}}}",
        rf"\providecommand{{\{prefix}ReportSampleStartYear}}{{{sample['start_year']}}}",
        rf"\providecommand{{\{prefix}ReportSampleEndYear}}{{{sample['end_year']}}}",
        rf"\providecommand{{\{prefix}ReportObservationCount}}{{{sample['n_obs']}}}",
        rf"\providecommand{{\{prefix}ReportFirmObservationCount}}{{{sample['firm_obs']}}}",
        rf"\providecommand{{\{prefix}ReportFirmMissingCount}}{{{sample['firm_missing']}}}",
    ]


def _param_fails(idata, name: str) -> bool | None:
    """Whether one scalar misses the acceptance rule on its own diagnostics."""
    if name not in idata.posterior:
        return None
    values = idata.posterior[name]
    rhat = float(np.nanmax(np.asarray(az.rhat(values), dtype=float)))
    ess = float(np.nanmin(np.asarray(az.ess(values, method="bulk"), dtype=float)))
    return (rhat > RHAT_LIMIT) or (ess < ESS_LIMIT)


def _steady_baseline_cells(runs) -> list:
    """The nine HSA-steady baseline cells: three price indices x three activity gaps."""
    out = []
    for inflation, activity_specs in INFLATION_SPECS.items():
        for activity, data_spec in activity_specs.items():
            item = runs.get(("hsa_steady", data_spec, "baseline"))
            if item is not None:
                out.append(item[1])
    return out


def _convergence_macros(runs, prefix: str) -> list[str]:
    hsa = [(key, idata) for key, (_, idata) in runs.items() if key[0] != "ces"]
    failing = [
        (key, _diagnostics(idata)["scalar"])
        for key, idata in hsa
        if not _diagnostics(idata)["scalar"]["converged"]
    ]
    ar2 = sum(1 for _, d in failing if d["worst_ess"] in {"rho_1", "rho_2"})
    drift = sum(1 for _, d in failing if d["worst_ess"] == "n")
    delta_fail = sum(1 for _, i in hsa if _param_fails(i, "delta") is True)
    kappa0_fail = sum(1 for _, i in hsa if _param_fails(i, "kappa_0") is True)

    steady = _steady_baseline_cells(runs)
    ess = [
        float(np.nanmin(np.asarray(az.ess(i.posterior["delta"], method="bulk"), dtype=float)))
        for i in steady
        if "delta" in i.posterior
    ]
    passing = sum(1 for i in steady if _param_fails(i, "delta") is False)

    const_theta = [i for key, (_, i) in runs.items() if key[0] == "hsa_const_theta"]
    ct_pass = sum(1 for i in const_theta if _diagnostics(i)["converged"])

    out = [
        # How many scalars the whole-run rule is a max/min over. The report leans on
        # the contrast between that rule and the same thresholds applied to delta
        # alone, so the width of the group has to be stated, not implied.
        rf"\providecommand{{\ScalarParameterCount}}{{{len(SCALAR_PARAMETERS)}}}",
        rf"\providecommand{{\{prefix}HsaCellCount}}{{{len(hsa)}}}",
        rf"\providecommand{{\{prefix}HsaFailCount}}{{{len(failing)}}}",
        rf"\providecommand{{\{prefix}ArTwoWorstCount}}{{{ar2}}}",
        rf"\providecommand{{\{prefix}DriftWorstCount}}{{{drift}}}",
        rf"\providecommand{{\{prefix}DeltaFailCount}}{{{delta_fail}}}",
        rf"\providecommand{{\{prefix}KappaZeroFailCount}}{{{kappa0_fail}}}",
        rf"\providecommand{{\{prefix}SteadyDeltaCellCount}}{{{len(steady)}}}",
        rf"\providecommand{{\{prefix}SteadyDeltaPassCount}}{{{passing}}}",
        rf"\providecommand{{\{prefix}ConstThetaCellCount}}{{{len(const_theta)}}}",
        rf"\providecommand{{\{prefix}ConstThetaPassCount}}{{{ct_pass}}}",
    ]
    if ess:
        out.append(rf"\providecommand{{\{prefix}SteadyDeltaMedianEss}}{{{_fmt_thousands(np.median(ess))}}}")

    # Worst-scalar mixing of the main cell under each prior set. The report uses
    # these to say which corner of the sweep breaks the sampler; stating it from
    # the runs is how that claim got corrected from "tight" to "weak".
    for prior_name in PRIOR_ORDER:
        item = runs.get(("hsa_steady", PRIMARY_SPECS["Core CPI"], prior_name))
        if item is None:
            continue
        scalar = _diagnostics(item[1])["scalar"]
        label = prior_name.capitalize()
        out.append(rf"\providecommand{{\{prefix}WorstEss{label}}}{{{_fmt_thousands(scalar['min_ess'])}}}")
        out.append(rf"\providecommand{{\{prefix}WorstRhat{label}}}{{{scalar['max_rhat']:.3f}}}")
    return out


def _ar2_period(rho1: float, rho2: float) -> float | None:
    """Period in quarters of the AR(2) cycle, or None when the roots are real."""
    disc = rho1**2 + 4.0 * rho2
    if disc >= 0.0:
        return None
    modulus_angle = np.arctan2(np.sqrt(-disc) / 2.0, rho1 / 2.0)
    if modulus_angle <= 0.0:
        return None
    return float(2.0 * np.pi / modulus_angle)


def _state_space_macros(runs, prefix: str) -> list[str]:
    """Design-comparison quantities, all read off the main cell."""
    item = runs.get(("hsa_steady", PRIMARY_SPECS["Core CPI"], "baseline"))
    if item is None:
        return []
    idata = item[1]
    post = idata.posterior
    if not {"Nbar", "Nhat", "kappa_0", "rho_1", "rho_2", "sigma_N"}.issubset(set(post.data_vars)):
        return []
    mean = lambda name: float(np.asarray(post[name], dtype=float).mean())  # noqa: E731
    rho1, rho2 = mean("rho_1"), mean("rho_2")
    T = post["Nbar"].shape[-1]
    nbar = np.asarray(post["Nbar"], dtype=float).reshape(-1, T)
    nhat = np.asarray(post["Nhat"], dtype=float).reshape(-1, T)
    kappa0 = np.asarray(post["kappa_0"], dtype=float).reshape(-1)

    out = [
        rf"\providecommand{{\{prefix}MainSigmaN}}{{{mean('sigma_N'):.3f}}}",
        rf"\providecommand{{\{prefix}MainRhoOne}}{{{rho1:+.2f}}}",
        rf"\providecommand{{\{prefix}MainRhoTwo}}{{{rho2:+.2f}}}",
        rf"\providecommand{{\{prefix}MainRhoSum}}{{{rho1 + rho2:.3f}}}",
        rf"\providecommand{{\{prefix}MainMeanReversion}}{{{1.0 - rho1 - rho2:.3f}}}",
        rf"\providecommand{{\{prefix}MainStateCorr}}{{{np.corrcoef(nbar[:, 0], nhat[:, 0])[0, 1]:+.4f}}}",
        rf"\providecommand{{\{prefix}MainLevelKappaCorr}}{{{np.corrcoef(nbar.mean(axis=1), kappa0)[0, 1]:+.2f}}}",
    ]
    period = _ar2_period(rho1, rho2)
    if period is not None:
        out.append(rf"\providecommand{{\{prefix}MainArTwoPeriod}}{{{period:.1f}}}")
    path = _path_summary(idata, "kappa_t")
    if path is not None:
        out.append(rf"\providecommand{{\{prefix}CoreUnempKappaDrop}}{{{path['start'] - path['end']:.2f}}}")
        # Response of inflation to a four-point negative unemployment gap at the
        # start-of-sample slope, the illustration used in Section "Main results".
        out.append(rf"\providecommand{{\{prefix}CoreUnempFourPointResponse}}{{{-4.0 * path['start']:+.2f}}}")
    return out


def _variant_macros(runs, prefix: str) -> list[str]:
    """Prior-sweep and entry-channel numbers the prose quotes cell by cell."""
    out: list[str] = []
    core = PRIMARY_SPECS["Core CPI"]
    for prior in ("weak", "tight"):
        item = runs.get(("hsa_steady", core, prior))
        if item is not None:
            out.append(
                rf"\providecommand{{\{prefix}CorePrior{prior.capitalize()}Delta}}"
                rf"{{{_fmt(_summary(item[1], 'delta'))}}}"
            )
    for inflation, label in (("Headline CPI", "Headline"), ("Core CPI", "Core")):
        item = runs.get(("hsa_const_theta", PRIMARY_SPECS[inflation], "baseline"))
        if item is None:
            continue
        idata = item[1]
        out.append(
            rf"\providecommand{{\{prefix}ConstTheta{label}Delta}}{{{_fmt(_summary(idata, 'delta'))}}}"
        )
        theta = _summary(idata, "theta")
        if theta is not None:
            out.append(
                rf"\providecommand{{\{prefix}ConstTheta{label}Theta}}{{{_fmt(theta)}}}"
            )
            out.append(
                rf"\providecommand{{\{prefix}ConstTheta{label}ThetaProb}}{{{theta['p_gt_0']:.2f}}}"
            )
    # The BN output gap under core CPI is the cell whose entry-coefficient sign
    # flips against the unemployment-gap cell; the prose contrasts the two.
    item = runs.get(("hsa_const_theta", INFLATION_SPECS["Core CPI"]["BN output gap"], "baseline"))
    if item is not None:
        theta = _summary(item[1], "theta")
        if theta is not None:
            out.append(rf"\providecommand{{\{prefix}ConstThetaCoreBNTheta}}{{{_fmt(theta)}}}")
            out.append(
                rf"\providecommand{{\{prefix}ConstThetaCoreBNThetaProbNeg}}"
                rf"{{{100.0 * (1.0 - theta['p_gt_0']):.0f}}}"
            )
    return out


def _fmt_thousands(value: float) -> str:
    return f"{int(round(float(value))):,}".replace(",", "{,}")


def _sddr_reliability_macros(runs, prefix: str) -> list[str]:
    """How much of the Rao-Blackwellised ordinate rests on how few draws.

    The report argues that \\eqref{eq:sddr-rb} is trustworthy where a kernel
    density estimate is not, and quotes the effective number of contributing
    mixture components to back that up. Those figures are computed here so the
    argument cannot drift away from the estimator it describes.
    """
    item = runs.get(("hsa_steady", PRIMARY_SPECS["Core CPI"], "baseline"))
    if item is None:
        return []
    idata = item[1]
    terms = _rao_blackwell_terms(idata, "delta")
    if terms is None or terms.size == 0:
        return []
    total = float(terms.sum())
    if not np.isfinite(total) or total <= 0.0:
        return []
    weights = terms / total
    effective = 1.0 / float(np.sum(weights**2))
    top3 = float(np.sort(weights)[-3:].sum()) * 100.0
    # Monte Carlo standard error of the mean of the mixture components.
    mcse = float(np.std(terms, ddof=1) / np.sqrt(terms.size) / (total / terms.size)) * 100.0
    return [
        rf"\providecommand{{\{prefix}SddrEffectiveTerms}}{{{_fmt_thousands(effective)}}}",
        rf"\providecommand{{\{prefix}SddrTotalTerms}}{{{_fmt_thousands(terms.size)}}}",
        rf"\providecommand{{\{prefix}SddrTopThreeShare}}{{{top3:.1f}}}",
        rf"\providecommand{{\{prefix}SddrMcsePercent}}{{{mcse:.0f}}}",
    ]


def build_delta_grid_table(runs) -> pd.DataFrame:
    """delta by activity measure x price index for the main model, with BF10.

    Section "Does the time-varying slope improve fit" reads this as a single
    grid. It used to be a hand-written tabular in the .tex filled with macros,
    which is how its caption came to say "quarterly" while its cells were the
    mixed-frequency numbers.
    """
    rows = []
    for activity in ["Unemployment gap", "HP output gap", "BN output gap"]:
        row = {"activity": "Negative unemployment gap" if activity == "Unemployment gap" else activity}
        for inflation in INFLATION_SPECS:
            item = runs.get(("hsa_steady", INFLATION_SPECS[inflation][activity], "baseline"))
            if item is None:
                row[inflation] = "--"
                continue
            _, idata = item
            row[inflation] = f"{_fmt(_summary(idata, 'delta'))} ({_fmt_num(_bf10(idata, 'delta'))})"
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_latex(table, "delta_grid", ["activity"] + list(INFLATION_SPECS))
    return table


def write_design_comparison_table(quarterly, annual, filename: str = "pchip_vs_mixed.tex") -> Path:
    """The side-by-side of what the interpolation changes.

    Spans both designs, so like the cross-design macros it is written once from
    main() rather than inside a per-design build.
    """
    def cell(runs, fn):
        item = runs.get(("hsa_steady", PRIMARY_SPECS["Core CPI"], "baseline"))
        return "--" if item is None else fn(item[1])

    def scalars(idata):
        post = idata.posterior
        mean = lambda name: float(np.asarray(post[name], dtype=float).mean())  # noqa: E731
        T = post["Nbar"].shape[-1]
        nbar = np.asarray(post["Nbar"], dtype=float).reshape(-1, T)
        nhat = np.asarray(post["Nhat"], dtype=float).reshape(-1, T)
        kappa0 = np.asarray(post["kappa_0"], dtype=float).reshape(-1)
        rho1, rho2 = mean("rho_1"), mean("rho_2")
        period = _ar2_period(rho1, rho2)
        return {
            "sigma_N": f"${mean('sigma_N'):.3f}$",
            "rho": f"${rho1:+.2f},\\,{rho2:+.2f}$",
            "period": "--" if period is None else f"${period:.1f}$ quarters",
            "reversion": f"${1.0 - rho1 - rho2:.3f}$",
            "state_corr": f"${np.corrcoef(nbar[:, 0], nhat[:, 0])[0, 1]:+.4f}$",
            "level_corr": f"${np.corrcoef(nbar.mean(axis=1), kappa0)[0, 1]:+.2f}$",
            "delta": _fmt(_summary(idata, "delta")),
            "kappa_path": (lambda p: "--" if p is None else f"${p['start']:+.3f} \\to {p['end']:+.3f}$")(
                _path_summary(idata, "kappa_t")
            ),
        }

    a, q = cell(annual, scalars), cell(quarterly, scalars)
    annual_sample = _sample_metadata(annual)
    quarterly_sample = _sample_metadata(quarterly)
    labels = [
        (
            "Firm-count observations used",
            f"{annual_sample['firm_obs']} of {annual_sample['n_obs']}",
            f"{quarterly_sample['firm_obs']} of {quarterly_sample['n_obs']}",
        ),
        (r"Estimated measurement error $\sigma_N$", a["sigma_N"], q["sigma_N"]),
        (r"$\rho_1,\rho_2$", a["rho"], q["rho"]),
        ("Implied AR(2) period", a["period"], q["period"]),
        (r"Mean reversion $1-\rho_1-\rho_2$", a["reversion"], q["reversion"]),
        (r"Posterior corr$(\bar N_0,\hat N_0)$", a["state_corr"], q["state_corr"]),
        (r"Posterior corr$(\bar N$ level$,\kappa_0)$", a["level_corr"], q["level_corr"]),
        (None, None, None),
        (r"$\delta$", a["delta"], q["delta"]),
        (r"$\kappa_t$ start $\to$ end", a["kappa_path"], q["kappa_path"]),
    ]
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"& Mixed-frequency (main) & PCHIP-interpolated \\", r"\midrule"]
    for name, left, right in labels:
        lines.append(r"\midrule" if name is None else f"{name} & {left} & {right} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    target = RESULTS_DIR / "tables" / "shared" / filename
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_cross_design_macros(*run_sets, filename: str = "cross_design_macros.tex") -> Path:
    """Quantities defined across both observation designs at once.

    The ridge diagnostic is the whole point of the PCHIP section: the weaker the
    AR(2) mean reversion, the tighter the trend/cycle ridge. It is a correlation
    over every estimated HSA cell in both designs, so it cannot live in a
    per-design macro file.
    """
    points: list[tuple[float, float]] = []
    for runs in run_sets:
        for key, (_, idata) in runs.items():
            post = idata.posterior
            if key[0] == "ces" or not {"rho_1", "rho_2", "Nbar", "Nhat"}.issubset(set(post.data_vars)):
                continue
            rho1 = float(np.asarray(post["rho_1"], dtype=float).mean())
            rho2 = float(np.asarray(post["rho_2"], dtype=float).mean())
            T = post["Nbar"].shape[-1]
            nbar = np.asarray(post["Nbar"], dtype=float).reshape(-1, T)
            nhat = np.asarray(post["Nhat"], dtype=float).reshape(-1, T)
            points.append((1.0 - rho1 - rho2, float(np.corrcoef(nbar[:, 0], nhat[:, 0])[0, 1])))
    lines = ["% Generated by scripts/12_build_cpi_ppi_report.py; do not edit by hand."]
    lines.append(rf"\providecommand{{\RidgeCellCount}}{{{len(points)}}}")
    if len(points) > 2:
        arr = np.asarray(points)
        lines.append(rf"\providecommand{{\RidgeMechanismCorr}}{{{np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]:+.2f}}}")
    target = RESULTS_DIR / "tables" / "shared" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def build_diagnostics_table(runs) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        data_spec = PRIMARY_SPECS["PPI"]
        item = runs.get((model, data_spec, "baseline"))
        if item is None:
            continue
        _, idata = item
        diagnostics = _diagnostics(idata)
        rows.append(
            {
                "model": MODEL_LABELS[model],
                "max Rhat": diagnostics["max_rhat"],
                "min bulk ESS": diagnostics["min_ess"],
                "status": _conv_status(diagnostics),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["max Rhat"] = table["max Rhat"].map(lambda x: f"{x:.3f}")
        table["min bulk ESS"] = table["min bulk ESS"].map(lambda x: f"{x:.0f}")
    _write_latex(table, "ppi_diagnostics", ["model", "max Rhat", "min bulk ESS", "status"])
    return table


def _array_diagnostics(idata, parameter: str) -> tuple[float, float]:
    values = idata.posterior[parameter]
    rhat = np.asarray(az.rhat(values), dtype=float)
    ess = np.asarray(az.ess(values, method="bulk"), dtype=float)
    return float(np.nanmax(rhat)), float(np.nanmin(ess))


def build_group_convergence_diagnostics(runs) -> pd.DataFrame:
    """Scalar / state-path / derived-path diagnostics for every model.

    This is the table that makes the acceptance rule auditable: it shows which
    group each cell passes, and names the single worst-mixing quantity in the
    scalar group so a reader can see whether a dagger is driven by a
    coefficient of interest or by a nuisance term such as the trend drift.
    """
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for inflation, data_spec in PRIMARY_SPECS.items():
            item = runs.get((model, data_spec, "baseline"))
            if item is None:
                continue
            idata = item[1]
            diagnostics = _diagnostics(idata)
            scalar, state, derived = diagnostics["scalar"], diagnostics["state"], diagnostics["derived"]
            n_diag = _group_diagnostics(idata, ["n"])
            rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "inflation": inflation,
                    "sampler": _sampler_label(idata),
                    "scalar Rhat": f"{scalar['max_rhat']:.3f}",
                    "scalar ESS": f"{scalar['min_ess']:.0f}",
                    "worst scalar": (scalar["worst_ess"] or "--").replace("_", r"\_"),
                    "n Rhat": "--" if not np.isfinite(n_diag["max_rhat"]) else f"{n_diag['max_rhat']:.3f}",
                    "n ESS": "--" if not np.isfinite(n_diag["min_ess"]) else f"{n_diag['min_ess']:.0f}",
                    "state Rhat": "--" if not np.isfinite(state["max_rhat"]) else f"{state['max_rhat']:.3f}",
                    "state ESS": "--" if not np.isfinite(state["min_ess"]) else f"{state['min_ess']:.0f}",
                    "derived Rhat": "--" if not np.isfinite(derived["max_rhat"]) else f"{derived['max_rhat']:.3f}",
                    "derived ESS": "--" if not np.isfinite(derived["min_ess"]) else f"{derived['min_ess']:.0f}",
                    "joint": "OK" if diagnostics["joint_converged"] else "watch",
                }
            )
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "group_convergence_diagnostics",
        [
            "model", "inflation", "sampler",
            "scalar Rhat", "scalar ESS", "worst scalar",
            "n Rhat", "n ESS",
            "state Rhat", "state ESS",
            "derived Rhat", "derived ESS",
            "joint",
        ],
    )
    return table


def build_primary_parameter_state_diagnostics(runs) -> pd.DataFrame:
    """Separate mixing of reported coefficients from the latent state paths."""
    rows: list[dict[str, object]] = []
    for inflation, data_spec in PRIMARY_SPECS.items():
        item = runs.get(("hsa_steady", data_spec, "baseline"))
        if item is None:
            continue
        idata = item[1]
        row: dict[str, object] = {"inflation": inflation}
        for parameter, label in [
            ("delta", r"$\delta$"),
            ("kappa_0", r"$\kappa_0$"),
            ("kappa_t", r"$\kappa_t$ path"),
            ("Nbar", r"$\bar N$ path"),
            ("Nhat", r"$\hat N$ path"),
        ]:
            max_rhat, min_ess = _array_diagnostics(idata, parameter)
            row[f"{label} $\\hat R$"] = f"{max_rhat:.3f}"
            row[f"{label} ESS"] = f"{min_ess:.0f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "primary_parameter_state_diagnostics",
        [
            "inflation",
            r"$\delta$ $\hat R$", r"$\delta$ ESS",
            r"$\kappa_0$ $\hat R$", r"$\kappa_0$ ESS",
            r"$\kappa_t$ path $\hat R$", r"$\kappa_t$ path ESS",
            r"$\bar N$ path $\hat R$", r"$\bar N$ path ESS",
            r"$\hat N$ path $\hat R$", r"$\hat N$ path ESS",
        ],
    )
    return table


def save_delta_forest(table: pd.DataFrame) -> None:
    if table.empty:
        return
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot = table.dropna(subset=["delta_mean", "delta_lo", "delta_hi"]).reset_index(drop=True)
    y = np.arange(len(plot))[::-1]
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for i, row in plot.iterrows():
        ax.errorbar(
            row["delta_mean"], y[i],
            xerr=[[row["delta_mean"] - row["delta_lo"]], [row["delta_hi"] - row["delta_mean"]]],
            fmt="o", color=colors[row["inflation"]], capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.inflation} / {r.activity}" for r in plot.itertuples()])
    ax.set_xlabel(r"Competition dependence $\delta$ (posterior mean and 95% interval)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "delta_by_inflation_activity.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_output_gap_delta_forest(table: pd.DataFrame) -> None:
    if table.empty:
        return
    plot = table.loc[table["activity"].isin(["HP output gap", "BN output gap"])].copy()
    plot = plot.dropna(subset=["delta_mean", "delta_lo", "delta_hi"]).reset_index(drop=True)
    y = np.arange(len(plot))[::-1]
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for i, row in plot.iterrows():
        ax.errorbar(
            row["delta_mean"], y[i],
            xerr=[[row["delta_mean"] - row["delta_lo"]], [row["delta_hi"] - row["delta_mean"]]],
            fmt="o", color=colors[row["inflation"]], capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.inflation} / {r.activity}" for r in plot.itertuples()])
    ax.set_xlabel(r"Competition dependence $\delta$ (posterior mean and 95% interval)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "delta_output_gaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_forest(
    table: pd.DataFrame,
    *,
    filename: str,
    xlabel: str,
    figsize: tuple[float, float],
    show_status_legend: bool = True,
) -> None:
    if table.empty:
        return
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plot = table.dropna(subset=["mean", "lo", "hi"]).reset_index(drop=True)
    y = np.arange(len(plot))[::-1]
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=figsize)
    for i, row in plot.iterrows():
        converged = bool(row["converged"])
        color = colors[row["inflation"]] if converged else "#888888"
        ax.errorbar(
            row["mean"],
            y[i],
            xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
            fmt="o" if converged else "x",
            color=color,
            ecolor=color,
            alpha=1.0 if converged else 0.65,
            capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    if show_status_legend and (~plot["converged"]).any():
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="black", lw=0, label="Convergence criteria met"),
                Line2D([0], [0], marker="x", color="#888888", lw=0, label="Diagnostic warning"),
            ],
            loc="best",
            frameon=False,
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parameter_row(idata, *, parameter: str, label: str, inflation: str) -> dict[str, object] | None:
    summary = _summary(idata, parameter)
    if summary is None:
        return None
    diagnostics = _diagnostics(idata)
    return {
        "label": label,
        "inflation": inflation,
        "parameter": parameter,
        "mean": summary["mean"],
        "lo": summary["lo"],
        "hi": summary["hi"],
        "converged": bool(diagnostics["converged"]),
        "max_rhat": diagnostics["max_rhat"],
        "min_bulk_ess": diagnostics["min_ess"],
    }


def save_additional_parameter_forests(runs) -> None:
    steady_rows: list[dict[str, object]] = []
    for inflation, activity_specs in INFLATION_SPECS.items():
        for activity, data_spec in activity_specs.items():
            item = runs.get(("hsa_steady", data_spec, "baseline"))
            if item is None:
                continue
            row = _parameter_row(
                item[1], parameter="kappa_0", label=f"{inflation} / {activity}", inflation=inflation
            )
            if row is not None:
                steady_rows.append(row)
    steady_table = pd.DataFrame(steady_rows)
    steady_table.to_csv(OUT_TABLES / "figure_kappa0_by_inflation_activity.csv", index=False)
    _save_forest(
        steady_table,
        filename="kappa0_by_inflation_activity.png",
        xlabel=r"Baseline slope at mean competition $\kappa_0$",
        figsize=(7.4, 5.4),
        show_status_legend=False,
    )

    slope_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for inflation, data_spec in PRIMARY_SPECS.items():
            item = runs.get((model, data_spec, "baseline"))
            if item is None:
                continue
            idata = item[1]
            label = f"{MODEL_LABELS[model]} / {inflation}"
            slope_parameter = "kappa" if model in {"ces", "hsa_dynamic"} else "kappa_0"
            slope_row = _parameter_row(
                idata, parameter=slope_parameter, label=label, inflation=inflation
            )
            alpha_row = _parameter_row(idata, parameter="alpha", label=label, inflation=inflation)
            if slope_row is not None:
                slope_rows.append(slope_row)
            if alpha_row is not None:
                alpha_rows.append(alpha_row)
            if model in {"hsa_dynamic", "hsa_const_theta", "hsa_full"}:
                entry_parameter = "theta" if model in {"hsa_dynamic", "hsa_const_theta"} else "theta_0"
                entry_row = _parameter_row(
                    idata, parameter=entry_parameter, label=label, inflation=inflation
                )
                if entry_row is not None:
                    entry_rows.append(entry_row)

    slope_table = pd.DataFrame(slope_rows)
    alpha_table = pd.DataFrame(alpha_rows)
    entry_table = pd.DataFrame(entry_rows)
    slope_table.to_csv(OUT_TABLES / "figure_slope_by_model_inflation.csv", index=False)
    alpha_table.to_csv(OUT_TABLES / "figure_alpha_by_model_inflation.csv", index=False)
    entry_table.to_csv(OUT_TABLES / "figure_entry_by_model_inflation.csv", index=False)
    _save_forest(
        slope_table,
        filename="slope_by_model_inflation.png",
        xlabel=r"Slope at mean competition $\kappa$ or $\kappa_0$",
        figsize=(7.8, 7.0),
    )

    if not alpha_table.empty and not entry_table.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.0))
        panels = [
            (axes[0], alpha_table, r"Inflation persistence $\alpha$"),
            (axes[1], entry_table, r"Entry effect $\theta$ or $\theta_0$"),
        ]
        colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
        for ax, table, xlabel in panels:
            plot = table.dropna(subset=["mean", "lo", "hi"]).reset_index(drop=True)
            y = np.arange(len(plot))[::-1]
            for i, row in plot.iterrows():
                converged = bool(row["converged"])
                color = colors[row["inflation"]] if converged else "#888888"
                ax.errorbar(
                    row["mean"], y[i],
                    xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                    fmt="o" if converged else "x", color=color, ecolor=color,
                    alpha=1.0 if converged else 0.65, capsize=3,
                )
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(plot["label"], fontsize=8)
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.25)
        axes[1].legend(
            handles=[
                Line2D([0], [0], marker="o", color="black", lw=0, label="Convergence criteria met"),
                Line2D([0], [0], marker="x", color="#888888", lw=0, label="Diagnostic warning"),
            ],
            loc="best", frameon=False, fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(OUT_FIGURES / "alpha_entry_by_model_inflation.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


# Prior-vs-posterior grid. One panel column per economic quantity, not per
# variable name: the slope level is ``kappa`` in the constant-slope models and
# ``kappa_0`` in the time-varying ones, and the entry coefficient is ``theta`` or
# ``theta_0``. Same object, different name, so one column each.
_PP_PANELS: list[tuple[tuple[str, ...], str]] = [
    (("kappa", "kappa_0"), "$\\kappa,\\kappa_0$\nslope level"),
    (("delta",), "$\\delta$\nslope vs $\\bar N$"),
    (("theta", "theta_0"), "$\\theta,\\theta_0$\nentry"),
    (("gamma",), "$\\gamma$\nentry vs $\\bar N$"),
]
_PP_PRICE_COLORS = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}


def _prior_moments(idata, name: str) -> tuple[float, float] | None:
    """The prior this run was actually estimated under, in physical units.

    Read from the run's own saved priors rather than from ``configs/``: the
    posterior was produced under the former, so overlaying the latter would draw
    a prior the draws never saw. Same source the Savage-Dickey ordinate uses.
    """
    spec = (getattr(idata, "attrs", {}).get("run_priors", {}) or {}).get(name)
    if spec is None:
        return None
    return (
        (float(spec["mean"]), float(spec["sd"])) if isinstance(spec, dict)
        else (float(spec[0]), float(spec[1]))
    )


def save_prior_vs_posterior_grid(runs, *, design_label: str) -> None:
    """One figure per activity gap: models down the rows, coefficients across.

    Reading a column shows what the model hierarchy does to a coefficient --
    ``delta`` is nearly identical in HSA steady, const-theta and full, and
    ``gamma`` sits on top of its own prior in every price index, which is the
    visual form of the identification claim in the scope section.
    """
    for activity in ("Unemployment gap", "HP output gap", "BN output gap"):
        nrow, ncol = len(MODEL_ORDER), len(_PP_PANELS)
        fig, axes = plt.subplots(nrow, ncol, figsize=(13.0, 11.0))
        drawn_any = False
        for row, model in enumerate(MODEL_ORDER):
            for col, (names, title) in enumerate(_PP_PANELS):
                ax = axes[row, col]
                prior, drawn = None, 0
                for price, activity_specs in INFLATION_SPECS.items():
                    item = runs.get((model, activity_specs[activity], "baseline"))
                    if item is None:
                        continue
                    idata = item[1]
                    name = next((n for n in names if n in idata.posterior), None)
                    if name is None:
                        continue
                    values = _draws(idata, name)
                    if values is None or values.size < 20 or float(np.std(values)) <= 0.0:
                        continue
                    if prior is None:
                        prior = _prior_moments(idata, name)
                    grid = np.linspace(values.min(), values.max(), 300)
                    density = gaussian_kde(values)(grid)
                    ax.plot(grid, density, color=_PP_PRICE_COLORS[price], lw=1.6)
                    ax.fill_between(grid, density, color=_PP_PRICE_COLORS[price], alpha=0.13)
                    drawn += 1
                if not drawn:
                    # The model restricts this coefficient away. The panel stays so
                    # the columns line up across models.
                    ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center",
                            va="center", fontsize=13, color="#CCCCCC")
                    ax.set_xticks([])
                else:
                    drawn_any = True
                    if prior is not None:
                        lo, hi = ax.get_xlim()
                        pad = 0.25 * (hi - lo)
                        grid = np.linspace(lo - pad, hi + pad, 400)
                        ax.plot(grid, norm.pdf(grid, *prior), color="#333333", lw=1.1, ls="--")
                        ax.set_xlim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
                    ax.axvline(0.0, color="#999999", lw=0.8, ls=":")
                    ax.tick_params(labelsize=7.5)
                ax.set_yticks([])
                ax.spines[["top", "right", "left"]].set_visible(False)
                if row == 0:
                    ax.set_title(title, fontsize=10, pad=10)
                if col == 0:
                    ax.set_ylabel(MODEL_LABELS[model], fontsize=10, labelpad=10)
        if not drawn_any:
            plt.close(fig)
            continue
        handles = [Line2D([], [], color=color, lw=2, label=price)
                   for price, color in _PP_PRICE_COLORS.items()]
        handles.append(Line2D([], [], color="#333333", lw=1.2, ls="--", label="prior"))
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                   fontsize=9.5, bbox_to_anchor=(0.5, -0.012))
        fig.suptitle(
            f"Prior vs posterior — {activity}, {design_label}, baseline priors",
            fontsize=12.5, y=0.995,
        )
        fig.tight_layout(rect=(0, 0.022, 1, 0.985))
        slug = INFLATION_SPECS["Core CPI"][activity].replace("_core", "")
        OUT_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_FIGURES / f"prior_vs_posterior_{slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_slope_prior_vs_posterior(runs) -> None:
    """Prior against posterior for the slope level, CES kappa vs HSA steady kappa_0.

    The figure the report calls ``fig:kappa_pp``. It previously existed only as a
    PNG in the build directory with no code behind it anywhere in the history --
    the one report input that could not be regenerated. Written here so a clean
    rebuild produces the whole document.
    """
    core = PRIMARY_SPECS["Core CPI"]
    series = [
        ("ces", "kappa", "CES: constant $\\kappa$", "#4477AA"),
        ("hsa_steady", "kappa_0", "HSA steady: $\\kappa_0$ at average competition", "#228833"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    # Collect the priors per model rather than taking the first one found. The two
    # models share a prior today, but calling it "common" without checking is how a
    # figure comes to assert something the config no longer says.
    priors: dict[tuple[float, float], list[str]] = {}
    span: list[float] = []
    plotted = False
    for model, parameter, label, color in series:
        item = runs.get((model, core, "baseline"))
        if item is None:
            continue
        draws = _draws(item[1], parameter)
        if draws is None or draws.size < 20 or float(np.std(draws)) <= 0.0:
            continue
        grid = np.linspace(draws.min(), draws.max(), 400)
        density = gaussian_kde(draws)(grid)
        ax.plot(grid, density, color=color, lw=2.0, label=label)
        ax.fill_between(grid, density, color=color, alpha=0.16)
        span += [draws.min(), draws.max()]
        moments = _prior_moments(item[1], parameter)
        if moments is not None:
            priors.setdefault(moments, []).append(MODEL_LABELS[model])
        plotted = True
    if not plotted:
        plt.close(fig)
        return

    # Draw the prior over a range wide enough to show its shape, not clipped to
    # where the posterior happens to sit.
    if priors and span:
        lo, hi = min(span), max(span)
        widest = max(sd for _, sd in priors)
        centre = np.mean([mean for mean, _ in priors])
        grid = np.linspace(min(lo, centre - 2.5 * widest), max(hi, centre + 2.5 * widest), 600)
        shared = len(priors) == 1
        for (mean, sd), models in priors.items():
            who = "common" if shared else "/".join(models)
            ax.plot(grid, norm.pdf(grid, loc=mean, scale=sd), color="#333333", lw=1.6,
                    ls="--" if shared else ":",
                    label=rf"{who} prior $\mathcal{{N}}({mean:g},{sd:g}^2)$")
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    ax.axvline(0.0, color="#888888", lw=1.0, ls=":")
    ax.set_xlabel("slope (inflation points per point of the negative unemployment gap)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURES / "pp_kappa_ces_vs_steady.png", dpi=200)
    plt.close(fig)


def save_kappa_path_comparison(runs) -> None:
    """Implied slope path by price index, one figure per activity measure.

    The unemployment-gap panel is the one the main text shows; the two output-gap
    panels are what the appendix compares it against, and they are built here so
    the flattening claim can be checked against the activity measures that do not
    support it rather than only the one that does.
    """
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    for activity, filename in (
        ("Unemployment gap", "kappa_t_unemployment_by_inflation.png"),
        ("HP output gap", "kappa_t_output_gap_hp_by_inflation.png"),
        ("BN output gap", "kappa_t_output_gap_bn_by_inflation.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        plotted = False
        for inflation, activity_specs in INFLATION_SPECS.items():
            item = runs.get(("hsa_steady", activity_specs[activity], "baseline"))
            if item is None or "kappa_t" not in item[1].posterior:
                continue
            idata = item[1]
            values = np.asarray(idata.posterior["kappa_t"], dtype=float)
            paths = values.reshape(-1, values.shape[-1])
            mean = np.nanmean(paths, axis=0)
            lo = np.nanquantile(paths, 0.025, axis=0)
            hi = np.nanquantile(paths, 0.975, axis=0)
            attrs = getattr(idata, "attrs", {})
            start = pd.Timestamp(str(attrs.get("sample_start", "1982-03-31"))).to_period("Q")
            dates = pd.period_range(start, periods=len(mean), freq="Q").to_timestamp(how="end")
            color = colors[inflation]
            ax.plot(dates, mean, color=color, lw=2.0, label=inflation)
            ax.fill_between(dates, lo, hi, color=color, alpha=0.13)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel(r"Implied NKPC slope $\kappa_t$")
        ax.set_xlabel("Quarter")
        ax.set_title(activity, fontsize=11)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        OUT_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_FIGURES / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_main_competition_decomposition(runs) -> None:
    """Save a stable main-model state decomposition instead of a shared overwritten figure."""
    item = runs.get(("hsa_steady", PRIMARY_SPECS["Core CPI"], "baseline"))
    if item is None:
        return
    idata = item[1]
    if "Nbar" not in idata.posterior or "Nhat" not in idata.posterior:
        return
    nbar_values = np.asarray(idata.posterior["Nbar"], dtype=float)
    nhat_values = np.asarray(idata.posterior["Nhat"], dtype=float)
    nbar = nbar_values.reshape(-1, nbar_values.shape[-1])
    nhat = nhat_values.reshape(-1, nhat_values.shape[-1])
    total = nbar + nhat
    attrs = getattr(idata, "attrs", {})
    start = pd.Timestamp(str(attrs.get("sample_start", "1982-03-31"))).to_period("Q")
    dates = pd.period_range(start, periods=nbar.shape[-1], freq="Q").to_timestamp(how="end")

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    series = [
        (total, "Latent total $N_t$", "#4477AA"),
        (nbar, r"Trend $\bar N_t$", "#228833"),
        (nhat, r"Cycle $\hat N_t$", "#EE7733"),
    ]
    for paths, label, color in series:
        mean = np.nanmean(paths, axis=0)
        lo = np.nanquantile(paths, 0.025, axis=0)
        hi = np.nanquantile(paths, 0.975, axis=0)
        ax.plot(dates, mean, color=color, lw=1.8, label=label)
        ax.fill_between(dates, lo, hi, color=color, alpha=0.12)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_title("Competition-state decomposition: HSA steady / Core CPI / unemployment gap")
    ax.set_ylabel("Ten-log-point units")
    ax.set_xlabel("Quarter")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(
        OUT_FIGURES / "competition_decomposition_hsa_steady_core_unemployment.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RESULTS_DIR / "runs")
    parser.add_argument("--min-iter", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true", help="Generate partial tables instead of failing on missing report cells.")
    args = parser.parse_args()

    quarterly_runs = _load_runs(
        args.runs_dir,
        min_iter=args.min_iter,
        competition_frequency="quarterly_interpolated",
    )
    annual_hsa_runs = _load_runs(
        args.runs_dir,
        min_iter=args.min_iter,
        competition_frequency="annual_q4",
    )

    missing_quarterly = sorted(set(report_run_keys()) - set(quarterly_runs))
    missing_annual = sorted(set(annual_q4_run_keys()) - set(annual_hsa_runs))
    missing = [("PCHIP", *key) for key in missing_quarterly] + [("annual-Q4", *key) for key in missing_annual]
    if missing and not args.allow_incomplete:
        preview = ", ".join("/".join(key) for key in missing[:8])
        raise SystemExit(f"Missing {len(missing)} required current-revision runs: {preview}")

    quarterly_required = set(report_run_keys())
    quarterly_display = {key: value for key, value in quarterly_runs.items() if key in quarterly_required}
    annual_required = set(annual_q4_run_keys())
    annual_display = {
        key: value
        for key, value in quarterly_display.items()
        if key[0] == "ces"
    }
    annual_display.update({key: value for key, value in annual_hsa_runs.items() if key in annual_required})

    # Within one observation design a cell must be reported under exactly one
    # sampler. (PCHIP vs annual-Q4 are different designs, so they are checked
    # separately rather than against each other.)
    assert_single_sampler_per_cell(quarterly_display)
    assert_single_sampler_per_cell(annual_display)
    for label, run_set in (("PCHIP", quarterly_display), ("annual-Q4", annual_display)):
        assert_expected_sampler(run_set, model="hsa_full", expected="particle_gibbs", label=label)
        assert_expected_sampler(run_set, model="hsa_const_theta", expected="joint_ffbs", label=label)
        samplers = sorted({_sampler_label(idata) for (m, _, _), (_, idata) in run_set.items() if m == "hsa_full"})
        print(f"  {label} hsa_full state sampler: {' / '.join(samplers) or 'n/a'}")

    base_tables = RESULTS_DIR / "tables"
    base_figures = RESULTS_DIR / "figures"
    # One directory per observation design, symmetrically. Neither design lives at
    # the top level: "the files without a subdirectory" is not a readable way to
    # say "the interpolated comparison".
    build_frequency_outputs(
        quarterly_display,
        tables_dir=base_tables / "quarterly_interpolated",
        figures_dir=base_figures / "quarterly_interpolated",
        command_prefix="",
    )
    build_frequency_outputs(
        annual_display,
        tables_dir=base_tables / "annual_q4",
        figures_dir=base_figures / "annual_q4",
        command_prefix="Annual",
    )
    write_cross_design_macros(quarterly_display, annual_display)
    write_design_comparison_table(quarterly_display, annual_display)
    subprocess.run(["python", str(ROOT / "scripts" / "make_spec_tables.py")], cwd=ROOT, check=True)
    subprocess.run(["python", str(ROOT / "scripts" / "11_additional_report_evidence.py")], cwd=ROOT, check=True)
    print(f"Saved PCHIP and annual-Q4 report inputs under {base_tables} and {base_figures}")
if __name__ == "__main__":
    main()
