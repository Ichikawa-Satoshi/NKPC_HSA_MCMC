"""Current-theory report inputs with strict provenance validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import arviz as az
import numpy as np
import pandas as pd

from nkpc_hsa.provenance import StaleArtifactError, validate_artifact_metadata
from nkpc_hsa.reporting.tables import write_latex_fragment
from nkpc_hsa.theory_models import (
    RESTRICTION_TAXONOMY,
    STRUCTURAL_FREQUENCY,
    THEORY_ESTIMATION_REVISION,
    THEORY_MODELS,
    theory_model_definition,
)


COMPARABILITY_FIELDS = (
    "code_revision",
    "estimation_revision",
    "data_spec",
    "sample_start",
    "sample_end",
    "n_obs",
    "structural_frequency",
    "inflation_observation",
    "inflation_transformation",
    "expectation_series",
    "expectation_horizon",
    "expectation_information_date",
    "activity_proxy",
    "activity_mapping",
    "marginal_cost_loading",
    "competition_proxy",
    "n_transform",
    "competition_measurement_frequency",
    "prior_spec",
)


def _read_run(run_dir: Path) -> tuple[dict[str, Any], Any]:
    meta_path, posterior_path = run_dir / "metadata.json", run_dir / "posterior.nc"
    if not meta_path.exists() or not posterior_path.exists():
        raise StaleArtifactError(f"Incomplete theory run: {run_dir}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_artifact_metadata(metadata, metadata, artifact=run_dir)
    model = str(metadata.get("model", ""))
    if model not in THEORY_MODELS:
        raise StaleArtifactError(f"STALE / HISTORICAL non-theory model in theory namespace: {run_dir}")
    definition = theory_model_definition(model)
    required = {
        "estimation_revision": THEORY_ESTIMATION_REVISION,
        "model_hierarchy": definition.hierarchy,
        "model_definition": definition.metadata(),
        "exact_restrictions": list(definition.exact_restrictions),
        "structural_frequency": STRUCTURAL_FREQUENCY,
    }
    mismatch = {key: (required[key], metadata.get(key)) for key in required if metadata.get(key) != required[key]}
    if mismatch:
        raise StaleArtifactError(f"STALE / HISTORICAL model definition in {run_dir}: {mismatch}")
    return metadata, az.from_netcdf(posterior_path)


def load_current_theory_runs(
    runs_dir: str | Path,
    *,
    min_iter: int = 12000,
    require_converged: bool = True,
    inflation_observation: str = "qoq",
) -> dict[str, tuple[Path, dict[str, Any], Any]]:
    """Load at most one current run per slug; no historical directory is scanned."""
    selected: dict[str, tuple[Path, dict[str, Any], Any]] = {}
    for run_dir in sorted(Path(runs_dir).glob("*")):
        if not run_dir.is_dir() or not (run_dir / "metadata.json").exists():
            continue
        metadata, idata = _read_run(run_dir)
        if metadata.get("inflation_observation") != inflation_observation:
            continue
        if int(metadata.get("n_iter", 0) or 0) < min_iter:
            raise StaleArtifactError(
                f"STALE / HISTORICAL quick/incomplete run {run_dir}: "
                f"n_iter={metadata.get('n_iter')} < {min_iter}"
            )
        if bool(metadata.get("code_dirty", True)):
            raise StaleArtifactError(f"STALE / HISTORICAL dirty-code run cannot enter current report: {run_dir}")
        if require_converged and metadata.get("convergence_status") != "converged":
            raise StaleArtifactError(
                f"Run has not passed current convergence diagnostics: {run_dir} "
                f"({metadata.get('convergence_status')!r})"
            )
        model = str(metadata["model"])
        current = selected.get(model)
        if current is None or str(metadata.get("run_id", "")) >= str(current[1].get("run_id", "")):
            selected[model] = (run_dir, metadata, idata)
    return selected


def assert_comparable(runs: Mapping[str, tuple[Path, Mapping[str, Any], Any]]) -> None:
    if not runs:
        return
    ordered = [runs[key][1] for key in sorted(runs)]
    baseline = ordered[0]
    mismatches: list[str] = []
    for metadata in ordered[1:]:
        for field in COMPARABILITY_FIELDS:
            if metadata.get(field) != baseline.get(field):
                mismatches.append(
                    f"{metadata.get('model')}.{field}: {metadata.get(field)!r} != {baseline.get(field)!r}"
                )
    if mismatches:
        raise StaleArtifactError(
            "Current theory table would mix incomparable results: " + "; ".join(mismatches)
        )


def _restriction_table() -> pd.DataFrame:
    rows = []
    labels = {
        "theory_identities": "Theory identity / cross-equation relation",
        "maintained_hsa_laws": "Maintained HSA law",
        "domain_admissibility": "Domain / local admissibility",
        "empirical_approximations": "Moving-reference empirical approximation",
        "deliberately_not_imposed": "Deliberately not imposed (future R4)",
    }
    for category, restrictions in RESTRICTION_TAXONOMY.items():
        rows.append({"Category": labels[category], "Contents": "; ".join(restrictions)})
    return pd.DataFrame(rows)


def _model_table() -> pd.DataFrame:
    rows = []
    for definition in THEORY_MODELS.values():
        rows.append(
            {
                "Model": definition.hierarchy,
                "Slug": definition.slug,
                "Reference": definition.reference_design.replace("_", " "),
                "Cross restriction": "yes" if definition.cross_equation_restriction else "no",
                "Second law": "maintained" if definition.second_law else "not maintained",
                "Third law": definition.third_law,
                "Gamma": definition.gamma_restriction,
                "State sampler": definition.state_sampler.replace("_", " "),
            }
        )
    return pd.DataFrame(rows)


def _posterior_table(runs: Mapping[str, tuple[Path, Mapping[str, Any], Any]]) -> pd.DataFrame:
    rows = []
    for slug in THEORY_MODELS:
        if slug not in runs:
            continue
        _, metadata, idata = runs[slug]
        posterior = idata.posterior
        for parameter in ("kappa_0", "kappa_N_empirical", "d_kappa_d_logN", "theta_0", "gamma"):
            if parameter not in posterior:
                continue
            values = np.asarray(posterior[parameter], dtype=float).reshape(-1)
            rows.append(
                {
                    "Model": metadata["model_hierarchy"],
                    "Parameter": parameter,
                    "Mean": float(np.mean(values)),
                    "SD": float(np.std(values, ddof=1)),
                    "2.5%": float(np.quantile(values, 0.025)),
                    "97.5%": float(np.quantile(values, 0.975)),
                    "Convergence": metadata.get("convergence_status", "pending_diagnostics"),
                }
            )
    return pd.DataFrame(rows)


def _observation_comparison_table(
    qoq_runs: Mapping[str, tuple[Path, Mapping[str, Any], Any]],
    yoy_runs: Mapping[str, tuple[Path, Mapping[str, Any], Any]],
) -> pd.DataFrame:
    rows = []
    for design, runs in (("Q/Q", qoq_runs), ("4Q YoY", yoy_runs)):
        for slug in ("hsa_f0", "hsa_u"):
            if slug not in runs:
                continue
            _, metadata, idata = runs[slug]
            row: dict[str, Any] = {
                "Model": metadata["model_hierarchy"],
                "Inflation observation": design,
                "T": metadata["n_obs"],
                "Convergence": metadata.get("convergence_status", "pending_diagnostics"),
            }
            for parameter in ("kappa_0", "theta_0", "kappa_N_empirical"):
                row[parameter] = (
                    float(np.mean(np.asarray(idata.posterior[parameter], dtype=float)))
                    if parameter in idata.posterior else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_path_comparison(
    runs: Mapping[str, tuple[Path, Mapping[str, Any], Any]],
    parameter: str,
    output_path: Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    available = [slug for slug in THEORY_MODELS if slug in runs and parameter in runs[slug][2].posterior]
    if not available:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for slug in available:
        _, metadata, idata = runs[slug]
        values = np.asarray(idata.posterior[parameter], dtype=float)
        paths = values.reshape(-1, values.shape[-1])
        center = np.nanmedian(paths, axis=0)
        lo, hi = np.nanquantile(paths, [0.05, 0.95], axis=0)
        start = str(metadata.get("sample_start", ""))
        try:
            quarters = pd.period_range(pd.Period(start, freq="Q"), periods=paths.shape[1], freq="Q")
            x = quarters.to_timestamp(how="end")
        except Exception:
            x = np.arange(paths.shape[1])
        label = str(metadata.get("model_hierarchy", slug))
        line = ax.plot(x, center, lw=1.7, label=label)[0]
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.10)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.45)
    ax.set_title(f"Posterior {parameter} paths: median and 90% interval")
    ax.set_ylabel(parameter)
    ax.set_xlabel("quarter")
    ax.grid(alpha=0.2)
    ax.legend(ncol=min(5, len(available)), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_restriction_coefficients(
    runs: Mapping[str, tuple[Path, Mapping[str, Any], Any]],
    output_path: Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    parameters = ("kappa_N_empirical", "theta_0", "gamma")
    records = []
    for slug in THEORY_MODELS:
        if slug not in runs:
            continue
        _, metadata, idata = runs[slug]
        for parameter in parameters:
            if parameter not in idata.posterior:
                continue
            values = np.asarray(idata.posterior[parameter], dtype=float).reshape(-1)
            records.append(
                (
                    str(metadata.get("model_hierarchy", slug)),
                    parameter,
                    float(np.mean(values)),
                    float(np.quantile(values, 0.05)),
                    float(np.quantile(values, 0.95)),
                )
            )
    if not records:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax, parameter in zip(axes, parameters):
        subset = [record for record in records if record[1] == parameter]
        if not subset:
            ax.axis("off")
            continue
        y = np.arange(len(subset))
        means = np.array([record[2] for record in subset])
        lo = np.array([record[3] for record in subset])
        hi = np.array([record[4] for record in subset])
        ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="o", capsize=3)
        ax.set_yticks(y, [record[0] for record in subset])
        ax.axvline(0.0, color="black", lw=0.8, alpha=0.45)
        ax.set_title(parameter)
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Restriction-relevant posterior coefficients (90% intervals)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_theory_report_inputs(
    runs_dir: str | Path,
    output_dir: str | Path,
    *,
    allow_missing: bool = False,
) -> dict[str, tuple[Path, dict[str, Any], Any]]:
    runs = load_current_theory_runs(runs_dir, inflation_observation="qoq")
    yoy_runs = load_current_theory_runs(runs_dir, inflation_observation="yoy_4q")
    missing = sorted(set(THEORY_MODELS) - set(runs))
    missing_yoy = sorted({"hsa_f0", "hsa_u"} - set(yoy_runs))
    if (missing or missing_yoy) and not allow_missing:
        raise RuntimeError(
            "Missing current theory runs " + ", ".join(missing)
            + "; missing Q/Q-vs-YoY comparison runs " + ", ".join(missing_yoy)
            + ". Historical results are not a substitute; estimate with scripts/10_estimate_theory_models.py."
        )
    assert_comparable(runs)
    assert_comparable(yoy_runs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures_dir = out.parent.parent / "figures" / "theory"
    write_latex_fragment(_restriction_table(), out / "restriction_taxonomy.tex", escape=True, column_format="p{0.27\\linewidth}p{0.68\\linewidth}")
    write_latex_fragment(_model_table(), out / "model_hierarchy.tex", escape=True)
    posterior = _posterior_table(runs)
    write_latex_fragment(
        posterior if not posterior.empty else pd.DataFrame({"Status": ["CURRENT THEORY ESTIMATES PENDING"]}),
        out / "posterior_results.tex",
        escape=True,
    )
    observation_comparison = _observation_comparison_table(runs, yoy_runs)
    write_latex_fragment(
        observation_comparison if not observation_comparison.empty else pd.DataFrame(
            {"Status": ["Q/Q VS 4Q-YOY CURRENT COMPARISON PENDING"]}
        ),
        out / "inflation_observation_comparison.tex",
        escape=True,
    )
    manifest = pd.DataFrame(
        [
            {
                "Model": meta["model_hierarchy"],
                "Run": path.name,
                "Revision": meta["estimation_revision"],
                "Code": str(meta["code_revision"])[:12],
                "Inflation": meta["inflation_observation"],
                "Sample": f"{meta['sample_start']}--{meta['sample_end']}",
                "Status": meta.get("convergence_status", "pending_diagnostics"),
            }
            for path, meta, _ in [*runs.values(), *yoy_runs.values()]
        ]
    )
    write_latex_fragment(
        manifest if not manifest.empty else pd.DataFrame({"Status": ["NO CURRENT RUNS; HISTORICAL ARTIFACTS EXCLUDED"]}),
        out / "run_manifest.tex",
        escape=True,
    )
    design = out / "current_design.tex"
    pending = bool(missing or missing_yoy)
    generated_figures = [] if pending else [
        _plot_restriction_coefficients(runs, figures_dir / "restriction_coefficients.png"),
        _plot_path_comparison(runs, "kappa_t", figures_dir / "kappa_t_paths.png"),
        _plot_path_comparison(runs, "theta_t", figures_dir / "theta_t_paths.png"),
    ]
    generated_figures = [path for path in generated_figures if path is not None]
    design.write_text(
        "\\section{Current research design: fixed and moving local references}\n"
        "The current specification starts from the quarterly, fixed-reference local Rotemberg--HSA equation. "
        "F0 is its fixed-$N_0$ empirical benchmark. U, R1, R2, and R3 add a slow-moving local reference; "
        "this moving-reference step is a semi-structural approximation and is not Eq. 21 itself.\n\n"
        "\\subsection{Restriction taxonomy}\n"
        "\\input{../results/tables/theory/restriction_taxonomy.tex}\n"
        "\\subsection{Model hierarchy and non-nesting}\n"
        "Within the moving-reference family, $U \\supset R1 \\supset R2$ and R3 is $R1+\\{\\gamma=0\\}$. "
        "F0 is not nested in that family because its reference is fixed.\n"
        "\\input{../results/tables/theory/model_hierarchy.tex}\n"
        "\\subsection{Quarterly structural scaling}\n"
        "With $\\bar N=10(\\log N-\\log N_0)$ and inflation in percentage points, "
        "$d\\kappa/d\\log N=10\\kappa_N$ and the restricted marginal-cost specification has "
        "$100\\kappa_N=b_x\\zeta_0\\theta_0$. The equality is never imposed directly on a gap proxy "
        "without an explicit $b_x$, or on the legacy direct 4Q-YoY regression.\n"
        "\\subsection{Current posterior results}\n"
        + (
            "\\textbf{CURRENT THEORY ESTIMATES PENDING. Historical tables below are not used as evidence for R1--R3.}\n"
            if pending
            else "\\input{../results/tables/theory/posterior_results.tex}\n"
        )
        +
        "\\subsection{Q/Q versus four-quarter YoY observation design}\n"
        "The two transformations of a common price index are estimated in separate likelihoods and are never stacked as if independent. "
        "F0 and U provide the direct observation-design comparison. R1--R3 remain Q/Q-only in this revision because a direct YoY regression "
        "does not preserve the meaning of the quarterly structural cross-equation coefficient.\n"
        + (
            "\\textbf{CURRENT OBSERVATION-DESIGN COMPARISON PENDING.}\n"
            if pending
            else "\\input{../results/tables/theory/inflation_observation_comparison.tex}\n"
        )
        +
        "\\subsection{Current-run provenance}\n"
        "\\input{../results/tables/theory/run_manifest.tex}\n"
        + (
            "\\subsection{Restriction-relevant posterior figures}\n"
            "\\begin{figure}[H]\\centering\n"
            "\\includegraphics[width=0.92\\linewidth]{restriction_coefficients.png}\n"
            "\\caption{Restriction-relevant posterior coefficients.}\\end{figure}\n"
            "\\begin{figure}[H]\\centering\n"
            "\\includegraphics[width=0.88\\linewidth]{kappa_t_paths.png}\n"
            "\\caption{Posterior slope paths by model.}\\end{figure}\n"
            "\\begin{figure}[H]\\centering\n"
            "\\includegraphics[width=0.88\\linewidth]{theta_t_paths.png}\n"
            "\\caption{Posterior entry-coefficient paths by model.}\\end{figure}\n"
            if len(generated_figures) == 3 else ""
        ),
        encoding="utf-8",
    )
    return runs
