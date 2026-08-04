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

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION
from nkpc_hsa.report.cpi_ppi_spec import (
    annual_q4_run_keys,
    INFLATION_SPECS,
    MODEL_LABELS,
    MODEL_ORDER,
    PRIMARY_SPECS,
    PRIOR_ORDER,
    report_run_keys,
)


OUT_TABLES = ROOT / "results" / "tables" / "cpi_ppi_report"
OUT_FIGURES = ROOT / "results" / "figures" / "cpi_ppi_report"
PG_RUNS_DIR = ROOT / "results" / "appendix_particle_gibbs" / "runs"
RHAT_LIMIT = 1.01
ESS_LIMIT = 400.0

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
        key = (model, data_spec, prior)
        current = selected.get(key)
        if current is None or _run_key(posterior, idata) >= _run_key(current[0], current[1]):
            selected[key] = (posterior, idata)
    return selected


def _sampler_label(idata) -> str:
    sampler = str(getattr(idata, "attrs", {}).get("state_sampler", "unknown"))
    return SAMPLER_LABELS.get(sampler, sampler)


def merge_preferred_runs(
    base_runs: dict[tuple[str, str, str], tuple[Path, object]],
    override_runs: dict[tuple[str, str, str], tuple[Path, object]],
    *,
    models: set[str],
) -> tuple[dict[tuple[str, str, str], tuple[Path, object]], list[tuple[str, str, str]]]:
    """Replace ``models``' cells in ``base_runs`` with ``override_runs`` where available.

    Used to route the PCHIP ``hsa_full`` cells through the Particle-Gibbs runs.
    Cells with no override run are left untouched, so a partially-populated
    override directory can never produce a half-substituted table: the merge is
    reported and validated by the caller.
    """
    merged = dict(base_runs)
    replaced: list[tuple[str, str, str]] = []
    for key, value in override_runs.items():
        if key[0] not in models:
            continue
        if key not in merged:
            continue
        merged[key] = value
        replaced.append(key)
    return merged, sorted(replaced)


def load_report_runs(
    *,
    runs_dir: Path | None = None,
    pg_runs_dir: Path | None = None,
    min_iter: int = 1,
    competition_frequency: str = "quarterly_interpolated",
    use_pg: bool = True,
    verbose: bool = False,
) -> dict[tuple[str, str, str], tuple[Path, object]]:
    """The single entry point every report artifact must use to obtain its runs.

    Assembles the authoritative run-set for one observation design: production
    runs for every model, with the ``hsa_full`` cells routed through the
    Particle-Gibbs directory when runs for that design exist there. Any script
    that builds a report table calls this rather than ``_load_runs`` directly,
    so a cell cannot be reported under one sampler in one table and a different
    sampler in another.
    """
    runs_dir = ROOT / "results" / "runs" if runs_dir is None else runs_dir
    pg_runs_dir = PG_RUNS_DIR if pg_runs_dir is None else pg_runs_dir
    runs = _load_runs(runs_dir, min_iter=min_iter, competition_frequency=competition_frequency)
    if use_pg and pg_runs_dir.exists():
        override = _load_runs(
            pg_runs_dir, min_iter=min_iter, competition_frequency=competition_frequency
        )
        runs, replaced = merge_preferred_runs(runs, override, models={"hsa_full"})
        if verbose:
            print(f"  {competition_frequency}: merged {len(replaced)} Particle-Gibbs hsa_full cells")
    return runs


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


def _bf10(idata, parameter: str) -> float | None:
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


def _conv_status(diagnostics: dict[str, object], *, japanese: bool = True) -> str:
    """Short status string distinguishing coefficient from joint convergence."""
    watch = "要注意" if japanese else "watch"
    if not bool(diagnostics["converged"]):
        return watch
    if diagnostics["has_states"] and not bool(diagnostics["joint_converged"]):
        return "OK (coef)"
    return "OK"


def _write_latex(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    display = df.reindex(columns=columns).copy()
    display.to_latex(
        OUT_TABLES / f"{name}.tex",
        index=False,
        escape=False,
        na_rep="--",
        column_format="l" * len(columns),
    )
    df.to_csv(OUT_TABLES / f"{name}.csv", index=False)


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
        ["model", "inflation", "slope", "delta", "BF10(delta)", "entry", "gamma", "kappa path", "diagnostics"],
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
    columns = ["model", "inflation", "slope", "delta", "BF10(delta)", "entry", "gamma", "kappa path", "diagnostics"]
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
    sample = data[list(cols.values())].dropna()
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
    data = pd.read_csv(ROOT / "data" / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    comparisons = [
        ("逆マークアップ（理論対応）", "inv_markup"),
        ("Headline CPI / 失業ギャップ", PRIMARY_SPECS["Headline CPI"]),
        ("Core CPI / 失業ギャップ", PRIMARY_SPECS["Core CPI"]),
        ("PPI / 失業ギャップ", PRIMARY_SPECS["PPI"]),
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
                "diagnostics": "OK" if converged else r"要注意$^{\dagger}$",
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
        ["specification", "CES kappa", "HSA kappa", "CES-HSA", "Pr(CES-HSA<0)", "HSA-implied OVB", "Pr(OVB<0)", "diagnostics"],
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
                        "parameter": f"${parameter}$",
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
    write_result_macros(runs, command_prefix=command_prefix)
    save_delta_forest(hsa_table)
    save_output_gap_delta_forest(hsa_table)
    save_additional_parameter_forests(runs)
    save_kappa_path_comparison(runs)
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
    target = OUT_TABLES / filename
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
                    "joint": "OK" if diagnostics["joint_converged"] else "要注意",
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
            ("delta", "delta"),
            ("kappa_0", "kappa0"),
            ("kappa_t", "kappa path"),
            ("Nbar", "Nbar path"),
            ("Nhat", "Nhat path"),
        ]:
            max_rhat, min_ess = _array_diagnostics(idata, parameter)
            row[f"{label} Rhat"] = f"{max_rhat:.3f}"
            row[f"{label} ESS"] = f"{min_ess:.0f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    _write_latex(
        table,
        "primary_parameter_state_diagnostics",
        [
            "inflation",
            "delta Rhat", "delta ESS",
            "kappa0 Rhat", "kappa0 ESS",
            "kappa path Rhat", "kappa path ESS",
            "Nbar path Rhat", "Nbar path ESS",
            "Nhat path Rhat", "Nhat path ESS",
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


def save_kappa_path_comparison(runs) -> None:
    colors = {"Headline CPI": "#4477AA", "Core CPI": "#228833", "PPI": "#CC6677"}
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    plotted = False
    for inflation, data_spec in PRIMARY_SPECS.items():
        item = runs.get(("hsa_steady", data_spec, "baseline"))
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
        return
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel(r"Implied NKPC slope $\kappa_t$")
    ax.set_xlabel("Quarter")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "kappa_t_unemployment_by_inflation.png", dpi=220, bbox_inches="tight")
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
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "results" / "runs")
    parser.add_argument("--min-iter", type=int, default=1000)
    parser.add_argument("--allow-incomplete", action="store_true", help="Generate partial tables instead of failing on missing report cells.")
    parser.add_argument("--compile", action="store_true", help="Compile the Japanese CPI/PPI report with XeLaTeX.")
    parser.add_argument(
        "--pg-runs-dir",
        type=Path,
        default=PG_RUNS_DIR,
        help="Particle-Gibbs run directory used for the PCHIP hsa_full cells.",
    )
    parser.add_argument(
        "--no-pg",
        action="store_true",
        help="Ignore the Particle-Gibbs runs and report alternating-FFBS hsa_full everywhere.",
    )
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

    # ------------------------------------------------------------------
    # Route the PCHIP hsa_full cells through the Particle-Gibbs runs, in the
    # single place where the report's run-set is assembled. Every table, macro
    # and figure downstream is then generated from one merged run-set, so a
    # cell can never appear under one sampler in one table and another sampler
    # in the next. Annual-Q4 is deliberately NOT substituted: no annual-Q4
    # Particle-Gibbs runs exist, and silently reusing PCHIP ones would compare
    # different observation schemes.
    # ------------------------------------------------------------------
    pg_replaced: list[tuple[str, str, str]] = []
    if not args.no_pg and args.pg_runs_dir.exists():
        pg_quarterly = _load_runs(
            args.pg_runs_dir,
            min_iter=args.min_iter,
            competition_frequency="quarterly_interpolated",
        )
        quarterly_runs, pg_replaced = merge_preferred_runs(
            quarterly_runs, pg_quarterly, models={"hsa_full"}
        )
        pg_annual = _load_runs(
            args.pg_runs_dir,
            min_iter=args.min_iter,
            competition_frequency="annual_q4",
        )
        if pg_annual:
            annual_hsa_runs, annual_replaced = merge_preferred_runs(
                annual_hsa_runs, pg_annual, models={"hsa_full"}
            )
            print(f"Merged {len(annual_replaced)} annual-Q4 Particle-Gibbs hsa_full cells.")
        else:
            print("No annual-Q4 Particle-Gibbs runs found: annual-Q4 hsa_full stays alternating FFBS.")
        print(f"Merged {len(pg_replaced)} PCHIP Particle-Gibbs hsa_full cells into the report run-set.")
        expected_full = {key for key in report_run_keys() if key[0] == "hsa_full"}
        unresolved = sorted(expected_full - set(pg_replaced))
        if unresolved and not args.allow_incomplete:
            preview = ", ".join("/".join(key) for key in unresolved[:8])
            raise SystemExit(
                f"{len(unresolved)} PCHIP hsa_full cells have no Particle-Gibbs run "
                f"({preview}). Re-run scripts/appendix_pg_full_runs.py, or pass --no-pg "
                "to report alternating FFBS for every hsa_full cell."
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
        samplers = sorted({_sampler_label(idata) for (m, _, _), (_, idata) in run_set.items() if m == "hsa_full"})
        print(f"  {label} hsa_full state sampler: {' / '.join(samplers) or 'n/a'}")

    base_tables = ROOT / "results" / "tables" / "cpi_ppi_report"
    base_figures = ROOT / "results" / "figures" / "cpi_ppi_report"
    build_frequency_outputs(
        quarterly_display,
        tables_dir=base_tables,
        figures_dir=base_figures,
        command_prefix="",
    )
    build_frequency_outputs(
        annual_display,
        tables_dir=base_tables / "annual_q4",
        figures_dir=base_figures / "annual_q4",
        command_prefix="Annual",
    )
    subprocess.run(["python", str(ROOT / "scripts" / "11_additional_report_evidence.py")], cwd=ROOT, check=True)
    print(f"Saved PCHIP and annual-Q4 report inputs under {base_tables} and {base_figures}")
    if args.compile:
        tex = ROOT / "paper" / "nkpc_hsa_report_ja.tex"
        build_dir = ROOT / "results" / "report" / "build" / "nkpc_hsa_report_ja"
        build_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "xelatex", "-interaction=nonstopmode", "-halt-on-error",
            f"-output-directory={build_dir}", str(tex),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(command, cwd=ROOT, check=True)
        built_pdf = build_dir / f"{tex.stem}.pdf"
        final_pdf = ROOT / "results" / "report" / f"{tex.stem}.pdf"
        shutil.copy2(built_pdf, final_pdf)
        print(f"Saved {final_pdf}")


if __name__ == "__main__":
    main()
