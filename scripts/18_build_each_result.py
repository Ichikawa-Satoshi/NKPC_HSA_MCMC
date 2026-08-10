"""Build one self-contained LaTeX document per data combination.

Every (activity x price index) x competition cell gets its own directory under
``results/each_result/``, holding the tables and figures for that cell and a
``.tex`` that ``\\input``s them. The report grid collapses across data
specifications; this is the complementary per-cell view.

    PYTHONPATH=src python scripts/18_build_each_result.py \\
        --runs-root results/runs --runs-root results/extensions/sec_inverse_hhi/runs \\
        --data data/processed/model_ready.csv \\
        --data data/processed/model_ready_sec_inverse_hhi.csv

``--data`` may be repeated; each run is matched to the first file that contains
every column its data spec needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.inference.pointwise_loglik import (
    UnsupportedModel,
    pointwise_log_likelihood,
    waic_from_pointwise,
)
from nkpc_hsa.inference.wrappers import _coerce_model_data, _prepare_competition_measurement, model_sample_index
from nkpc_hsa.reporting.each_result import build_cells, discover_runs
from nkpc_hsa.reporting.each_result_artifacts import (
    LoadedRun,
    coefficient_table,
    convergence_table,
    data_description_table,
    load_cell,
    parameter_units_table,
    select_for_paths,
    design_label,
    model_label,
    plot_competition_decomposition,
    plot_path_across_models,
    plot_prior_posterior,
    prior_comparison_table,
    tex_escape,
)
from nkpc_hsa.reporting.tables import write_latex_fragment

HEADLINE_PARAMETERS = ("alpha", "kappa_0", "kappa", "delta", "theta", "theta_0", "gamma")


def _required_columns(data_spec: dict) -> list[str]:
    keys = ("pi_col", "pi_prev_col", "pi_expect_col", "x_col", "x_prev_col", "n_col", "e_col")
    return [str(data_spec[key]) for key in keys if data_spec.get(key)]


def _match_data(frames: list[tuple[Path, pd.DataFrame]], data_spec: dict) -> tuple[Path, pd.DataFrame] | None:
    needed = _required_columns(data_spec)
    for path, frame in frames:
        if all(column in frame.columns for column in needed):
            return path, frame
    return None


def _spec_with_run_window(run: LoadedRun) -> dict:
    """The run's data spec, with the estimation window it actually used.

    Runs written before the window was added to ``data_spec.json`` carry no
    ``sample_start``/``sample_end`` there, and rebuilding without one silently
    picks up whatever the current inputs allow -- the T = 124 to 128 drift that
    ``configs/models.yaml`` warns about. ``metadata.json`` always recorded the
    window the sampler used, so take it from there.
    """
    spec = dict(run.ref.data_spec)
    for key in ("sample_start", "sample_end"):
        if not spec.get(key) and run.ref.metadata.get(key):
            spec[key] = run.ref.metadata[key]
    return spec


def _model_data_for(run: LoadedRun, frames: list[tuple[Path, pd.DataFrame]], *, allow_drift: bool = False) -> dict:
    """Rebuild exactly the arrays the sampler consumed, including the N observation design.

    Raises so the caller can report *why* a cell has no comparison rather than
    silently printing "not matched" for every kind of failure.
    """
    matched = _match_data(frames, run.ref.data_spec)
    if matched is None:
        raise LookupError(f"no --data file has all of {_required_columns(run.ref.data_spec)}")
    _, frame = matched
    spec = _spec_with_run_window(run)
    model_data = _coerce_model_data(frame, data_spec=spec)
    sample_index = model_sample_index(frame, spec)
    design = spec.get("competition_measurement") or {
        "frequency": run.ref.frequency,
        "annual_timing": "q4",
    }
    context = _prepare_competition_measurement(
        model=run.model,
        data=frame,
        data_spec=spec,
        model_data=model_data,
        sample_index=sample_index,
        n_transform=str(run.ref.metadata.get("n_transform", "log100_centered10")),
        competition_measurement=design,
    )
    out = {key: np.asarray(model_data[key], dtype=float) for key in ("pi", "pi_prev", "pi_expect", "x", "x_prev")}
    observation = context.get("N_obs_used")
    out["N_obs"] = np.full(out["pi"].size, np.nan) if observation is None else np.asarray(observation, dtype=float)
    if not allow_drift:
        _check_data_matches_run(run, out)
    return out


def _data_drift(run: LoadedRun, data: dict) -> str | None:
    """Describe how the rebuilt competition observation differs from the saved one."""
    recorded = run.ref.metadata.get("competition_measurement") or {}
    finite = data["N_obs"][np.isfinite(data["N_obs"])]
    if not recorded or finite.size == 0:
        return None
    checks = {
        "count": ("finite_N_obs_count", float(finite.size)),
        "mean": ("finite_N_obs_mean", float(np.mean(finite))),
        "min": ("finite_N_obs_min", float(np.min(finite))),
        "max": ("finite_N_obs_max", float(np.max(finite))),
    }
    differences = []
    for label, (key, rebuilt) in checks.items():
        saved = recorded.get(key)
        if saved is None:
            continue
        if not np.isclose(float(saved), rebuilt, rtol=1e-6, atol=1e-8):
            differences.append(f"{label} {float(saved):.4g}->{rebuilt:.4g}")
    return ", ".join(differences) if differences else None


def _check_data_matches_run(run: LoadedRun, data: dict) -> None:
    """Refuse to score a run against a model-ready file that has since changed.

    Each run recorded the moments of the competition observation it actually
    used. A rebuilt input series (a new SEC vintage, a re-centred transform)
    silently changes those, and the resulting LOO would describe neither the
    saved posterior nor the current data.
    """
    recorded = run.ref.metadata.get("competition_measurement") or {}
    finite = data["N_obs"][np.isfinite(data["N_obs"])]
    if not recorded or finite.size == 0:
        return
    checks = {
        "finite_N_obs_count": float(finite.size),
        "finite_N_obs_mean": float(np.mean(finite)),
        "finite_N_obs_min": float(np.min(finite)),
        "finite_N_obs_max": float(np.max(finite)),
    }
    for key, rebuilt in checks.items():
        saved = recorded.get(key)
        if saved is None:
            continue
        if not np.isclose(float(saved), rebuilt, rtol=1e-6, atol=1e-8):
            raise ValueError(
                f"the model-ready data no longer match this run ({key}: saved {float(saved):.6g}, "
                f"rebuilt {rebuilt:.6g}); re-estimate the cell or point --data at the vintage it used"
            )


def model_comparison_table(
    runs: list[LoadedRun],
    frames: list[tuple[Path, pd.DataFrame]],
    *,
    allow_drift: bool = False,
) -> pd.DataFrame:
    """In-sample fit, LOO and WAIC from recomputed per-period predictive densities."""
    rows = []
    for run in runs:
        row = {"Model": model_label(run.model), "Design": design_label(run.ref.frequency), "Prior": run.prior}
        try:
            if run.ref.data_spec.get("e_col"):
                # A joint N/E run was estimated with a six-state filter and an extra
                # establishment observation row. Running the three-state filter here
                # would silently score a different model, so refuse instead.
                raise UnsupportedModel(
                    "joint N/E runs need the six-state filter; the three-state one would "
                    "ignore the establishment observation rows"
                )
            data = _model_data_for(run, frames, allow_drift=allow_drift)
            drift = _data_drift(run, data) if allow_drift else None
            log_lik = pointwise_log_likelihood(run.model, run.posterior.posterior, data, run.priors)
        except UnsupportedModel as error:
            row["Note"] = tex_escape(str(error).split(":", 1)[-1].strip())
            rows.append(row)
            continue
        except Exception as error:  # a numerically failed filter must not kill the cell
            row["Note"] = tex_escape(f"{type(error).__name__}: {error}")
            rows.append(row)
            continue
        posterior = run.posterior.posterior
        idata = az.from_dict(
            {
                "posterior": {
                    name: np.asarray(posterior[name])
                    for name in posterior.data_vars
                    if posterior[name].ndim == 2
                },
                "log_likelihood": {"obs": log_lik},
            }
        )
        if drift:
            row["Note"] = tex_escape(f"scored on current data; saved run used: {drift}")
        row["$\\sum_t \\overline{\\log p_t}$"] = f"{float(np.sum(np.mean(log_lik, axis=(0, 1)))):.2f}"
        try:
            loo = az.loo(idata, pointwise=False)
            elpd = float(getattr(loo, "elpd", getattr(loo, "elpd_loo", np.nan)))
            row["elpd LOO (se)"] = f"{elpd:.2f} ({float(getattr(loo, 'se', np.nan)):.2f})"
            row["$p_{\\mathrm{LOO}}$"] = f"{float(getattr(loo, 'p', getattr(loo, 'p_loo', np.nan))):.2f}"
        except Exception as error:
            row["elpd LOO (se)"] = f"failed: {type(error).__name__}"
        # ArviZ 1.0 removed az.waic, so WAIC-2 is computed from the same array.
        waic = waic_from_pointwise(log_lik)
        row["elpd WAIC (se)"] = f"{waic['elpd_waic']:.2f} ({waic['se']:.2f})"
        row["$p_{\\mathrm{WAIC}}$"] = f"{waic['p_waic']:.2f}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.fillna("")


CELL_TEMPLATE = r"""\documentclass[11pt,landscape]{article}
\usepackage[landscape,margin=1.8cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{tabularx}
\usepackage{array}
\usepackage{makecell}
\renewcommand{\arraystretch}{1.25}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.6em}
%% Every table and figure gets a page of its own, so nothing is squeezed and
%% nothing has to be read next to something else.
\newcommand{\onepagetable}[2]{%%
  \subsection*{#1}%%
  {\centering\resizebox{\linewidth}{!}{\input{#2}}\par}%%
  \clearpage}
\newcommand{\onepagefigure}[2]{%%
  \subsection*{#1}%%
  {\centering\includegraphics[width=\linewidth,height=0.80\textheight,keepaspectratio]{#2}\par}%%
  \clearpage}

\title{%(title)s}
\date{}

\begin{document}
\maketitle
\tableofcontents
\clearpage

\section{Data}
%(data_note)s
\onepagetable{Series, sample and estimation settings}{tables/data_description.tex}
\onepagetable{Parameter units}{tables/parameter_units.tex}

\section{Coefficients (baseline prior)}
%(coefficient_blocks)s

\section{Prior versus posterior (baseline prior)}
%(prior_posterior_figures)s

\section{Slope path $\kappa_t$}
%(kappa_figure)s

\section{Competition loading path $\theta_t$}
%(theta_figure)s

\section{Competition state decomposition}
%(decomposition_figure)s

\section{Model comparison}
LOO and WAIC come from one-step-ahead Kalman predictive densities recomputed
from the stored draws. \textbf{Compare only within one observation design}:
the Q4 and PCHIP designs score different observation vectors -- 31 firm-count
observations against 124 -- so the PCHIP elpd is mechanically higher and a
cross-design difference measures the design, not the model. A model with no row
has no exact predictive density implemented.

%(comparison_tables)s

\section{Convergence}
\onepagetable{R-hat and bulk ESS}{tables/convergence.tex}

\appendix
\section{Prior robustness}
The weak and tight priors are robustness checks on the baseline above.

%(appendix_blocks)s

\end{document}
"""

MAIN_PRIOR = "baseline"


def _page_table(title: str, relative: str) -> str:
    return f"\\onepagetable{{{title}}}{{{relative}}}\n"


def _page_figure(title: str, relative: str) -> str:
    return f"\\onepagefigure{{{title}}}{{{relative}}}\n"


def build_cell(
    cell,
    runs: list[LoadedRun],
    out_dir: Path,
    frames: list[tuple[Path, pd.DataFrame]],
    *,
    allow_drift: bool = False,
) -> Path:
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    # Rebuild from clean: a table or figure renamed between versions would
    # otherwise stay behind and look current.
    for directory in (tables, figures):
        if directory.exists():
            for stale in directory.iterdir():
                if stale.is_file():
                    stale.unlink()
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    designs = list(dict.fromkeys(run.ref.frequency for run in runs))

    write_latex_fragment(
        data_description_table(cell, runs),
        tables / "data_description.tex",
        escape=False,
        column_format="@{}l >{\\raggedright\\arraybackslash}X@{}",
        tabularx=True,
    )
    write_latex_fragment(
        parameter_units_table(runs),
        tables / "parameter_units.tex",
        escape=False,
        column_format="@{}l >{\\raggedright\\arraybackslash}X@{}",
        tabularx=True,
    )
    write_latex_fragment(convergence_table(runs), tables / "convergence.tex", escape=False)

    # Model comparison, one table per observation design: the elpd levels are
    # not comparable across designs, so putting them in one table invites the
    # comparison the caption forbids.
    comparison_tables = []
    for frequency in designs:
        subset = [run for run in runs if run.ref.frequency == frequency]
        table = model_comparison_table(subset, frames, allow_drift=allow_drift)
        table = table.drop(columns=[c for c in ("Design", "Note") if c in table.columns])
        table = table.loc[:, [c for c in table.columns if table[c].astype(str).str.strip().any()]]
        metrics = [c for c in table.columns if c not in ("Model", "Prior")]
        if metrics:
            table = table[table[metrics].astype(str).apply(lambda row: row.str.strip().any(), axis=1)]
        if table.empty or table.shape[1] <= 2:
            continue
        name = f"model_comparison_{frequency}.tex"
        write_latex_fragment(
            table,
            tables / name,
            escape=False,
            column_format="@{}ll" + "r" * (table.shape[1] - 2) + "@{}",
        )
        comparison_tables.append(_page_table(f"{design_label(frequency)} design", f"tables/{name}"))

    def coefficient_block(prior: str) -> list[str]:
        blocks = []
        for frequency in designs:
            table = coefficient_table(runs, prior, frequency)
            if table.empty:
                continue
            name = f"coefficients_{prior}_{frequency}.tex"
            write_latex_fragment(
                table,
                tables / name,
                escape=False,
                column_format="@{}l" + "r" * (table.shape[1] - 1) + "@{}",
            )
            blocks.append(_page_table(f"{prior.capitalize()} prior, {design_label(frequency)} design", f"tables/{name}"))
        return blocks

    def prior_posterior_blocks(prior: str) -> list[str]:
        blocks = []
        for run in [r for r in runs if r.prior == prior]:
            name = f"prior_posterior_{run.model}_{run.ref.frequency}_{run.prior}.pdf"
            if plot_prior_posterior(run, figures / name) is None:
                continue
            blocks.append(
                _page_figure(
                    f"{model_label(run.model)}, {design_label(run.ref.frequency)} design, {prior} prior",
                    f"figures/{name}",
                )
            )
        return blocks

    coefficient_blocks = coefficient_block(MAIN_PRIOR) or coefficient_block(cell.priors()[0])

    # Time-series figures: baseline prior only, one figure per observation design.
    path_blocks: dict[str, list[str]] = {"kappa": [], "theta": [], "decomposition": []}
    for frequency in designs:
        chosen = select_for_paths(runs, frequency=frequency, prior=MAIN_PRIOR)
        if not chosen:
            continue
        label = design_label(frequency)
        suffix = "" if chosen[0].prior == MAIN_PRIOR else f", {chosen[0].prior} prior"
        for key, variable, symbol in (("kappa", "kappa_t", "\\kappa_t"), ("theta", "theta_t", "\\theta_t")):
            name = f"{variable}_{frequency}.pdf"
            if plot_path_across_models(chosen, variable, figures / name, ylabel=f"${symbol}$") is not None:
                path_blocks[key].append(
                    _page_figure(f"${symbol}$, {label} design{suffix}", f"figures/{name}")
                )
        name = f"competition_decomposition_{frequency}.pdf"
        if plot_competition_decomposition(chosen, figures / name) is not None:
            path_blocks["decomposition"].append(
                _page_figure(f"Competition states, {label} design{suffix}", f"figures/{name}")
            )

    appendix_blocks = []
    for prior in [p for p in cell.priors() if p != MAIN_PRIOR]:
        appendix_blocks.append(f"\\subsection{{{prior.capitalize()} prior}}\n")
        appendix_blocks.extend(coefficient_block(prior))
        appendix_blocks.extend(prior_posterior_blocks(prior))
    prior_table = prior_comparison_table(runs, HEADLINE_PARAMETERS)
    if not prior_table.empty:
        write_latex_fragment(prior_table, tables / "prior_comparison.tex", escape=False)
        appendix_blocks.append(
            "\\subsection{Headline parameters across priors}\n"
            + _page_table("Baseline, weak and tight", "tables/prior_comparison.tex")
        )

    document = CELL_TEMPLATE % {
        "title": cell.title,
        "data_note": "Every run in this directory shares the series below and differs only by model, observation design and prior.",
        "coefficient_blocks": "".join(coefficient_blocks) or "No coefficient draws were found.\n",
        "prior_posterior_figures": "".join(prior_posterior_blocks(MAIN_PRIOR)) or "No scalar parameters were found.\n",
        "kappa_figure": "".join(path_blocks["kappa"]) or "No model in this cell has a time-varying $\\kappa_t$.\n",
        "theta_figure": "".join(path_blocks["theta"]) or "No model in this cell has a time-varying $\\theta_t$.\n",
        "decomposition_figure": "".join(path_blocks["decomposition"]) or "No model in this cell carries competition states.\n",
        "comparison_tables": "".join(comparison_tables) or "No model in this cell has an exact predictive density.\n",
        "appendix_blocks": "".join(appendix_blocks) or "Only the baseline prior was estimated for this cell.\n",
    }
    target = out_dir / f"{cell.name}.tex"
    target.write_text(document, encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one LaTeX document per data combination.")
    parser.add_argument("--runs-root", type=Path, action="append", default=None, help="Repeatable.")
    parser.add_argument("--data", type=Path, action="append", default=None, help="Repeatable model-ready CSV.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "each_result")
    parser.add_argument(
        "--only-cell",
        action="append",
        help="Rebuild only cells whose directory matches, e.g. unempgap_core/N_gustavo. Repeatable.",
    )
    parser.add_argument(
        "--allow-data-drift",
        action="store_true",
        help="Score runs whose inputs have since been rebuilt, annotating the difference. "
             "Within a cell every model still sees identical data, so the ranking holds; "
             "the level is not the vintage the posterior was drawn under.",
    )
    args = parser.parse_args()

    roots = args.runs_root or [RESULTS_DIR / "runs"]
    data_paths = args.data or [DATA_DIR / "processed" / "model_ready.csv"]
    frames = [
        (path, pd.read_csv(path, parse_dates=["DATE"]).set_index("DATE"))
        for path in data_paths
        if path.exists()
    ]

    runs = discover_runs([Path(root) for root in roots])
    cells = build_cells(runs)
    print(f"runs={len(runs)} cells={len(cells)} out={args.out_dir}")

    written = []
    wanted = set(args.only_cell or [])
    for cell in cells:
        if wanted and str(cell.relative_dir) not in wanted:
            continue
        loaded = load_cell(cell)
        if not loaded:
            continue
        out_dir = args.out_dir / cell.relative_dir
        target = build_cell(cell, loaded, out_dir, frames, allow_drift=args.allow_data_drift)
        written.append(target)
        print(f"  {cell.relative_dir}  models={len(loaded)}  -> {target.name}")

    index = args.out_dir / "index.md"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        "# each_result cells\n\n"
        + "\n".join(f"- `{path.relative_to(args.out_dir)}`" for path in written)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(written)} cell documents and {index}")


if __name__ == "__main__":
    main()
