from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, data_spec_label, load_model_config
from nkpc_hsa.report.figures import save_placeholder_figure, save_posterior_predictive_placeholder
from nkpc_hsa.report.latex import compile_report, write_default_report
from nkpc_hsa.report.tables import write_latex_fragment


def _ensure_fragment(path: Path, text: str) -> None:
    if not path.exists() or "Run scripts/" in path.read_text(encoding="utf-8"):
        write_latex_fragment(pd.DataFrame({"note": [text]}), path)


def _ensure_inputs(tables_dir: Path, figures_dir: Path, model_comparison_dir: Path) -> None:
    _ensure_fragment(tables_dir / "coefficient_means.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "result_blocks.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "kappa_model_comparison.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "time_varying_coefficients.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "sddr.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "prior_set_coefficients.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "posterior_summary.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "competition_decomposition.tex", "Run scripts/07_make_tables_figures.py first.")
    _ensure_fragment(tables_dir / "mcmc_diagnostics.tex", "Run scripts/03_run_diagnostics.py first.")
    _ensure_fragment(tables_dir / "prior_robustness.tex", "Run scripts/04_prior_robustness.py first.")
    _ensure_fragment(tables_dir / "period_robustness.tex", "Run scripts/05_period_robustness.py first.")
    _ensure_fragment(model_comparison_dir / "model_comparison.tex", "Run scripts/06_model_comparison.py first.")
    for path, text in [
        (figures_dir / "kappa_model_comparison.png", "No kappa comparison figure available yet."),
        (figures_dir / "kappa_t_path.png", "No kappa_t path available yet."),
        (figures_dir / "theta_t_path.png", "No theta_t path available yet."),
        (figures_dir / "prior_posterior_kappa.png", "No prior/posterior kappa figure available yet."),
        (figures_dir / "prior_posterior_kappa_0.png", "No prior/posterior kappa_0 figure available yet."),
        (figures_dir / "prior_posterior_delta.png", "No prior/posterior delta figure available yet."),
        (figures_dir / "prior_posterior_theta.png", "No prior/posterior theta figure available yet."),
        (figures_dir / "prior_posterior_theta_0.png", "No prior/posterior theta_0 figure available yet."),
        (figures_dir / "prior_posterior_gamma.png", "No prior/posterior gamma figure available yet."),
        (figures_dir / "prior_set_key_coefficients.png", "No prior-set coefficient figure available yet."),
        (figures_dir / "period_set_key_coefficients.png", "No period coefficient figure available yet."),
        (figures_dir / "competition_decomposition_path.png", "No competition decomposition figure available yet."),
    ]:
        if not path.exists():
            save_placeholder_figure(path, text)
    save_posterior_predictive_placeholder(figures_dir / "posterior_predictive_placeholder.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", default=str(ROOT / "reports" / "main.tex"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec report to compile. Repeat for multiple. Defaults to config run_data_specs.",
    )
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()

    config = load_model_config(args.config)
    data_specs = configured_data_specs(config, args.data_specs)
    written = []
    if not args.skip_combined:
        write_default_report(args.tex)
        _ensure_inputs(ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "results" / "model_comparison")
        written.append(Path(args.tex))
    if not args.combined_only:
        for data_spec_name, data_spec in data_specs.items():
            tex = ROOT / "reports" / f"{data_spec_name}.tex"
            write_default_report(
                tex,
                title_suffix=f": {data_spec_label(data_spec)}",
                table_dir=f"../tables/{data_spec_name}",
                figure_dir=f"../figures/{data_spec_name}",
                model_comparison_dir=f"../model_comparison/{data_spec_name}",
            )
            _ensure_inputs(
                ROOT / "results" / "tables" / data_spec_name,
                ROOT / "results" / "figures" / data_spec_name,
                ROOT / "results" / "model_comparison" / data_spec_name,
            )
            written.append(tex)
    if args.skip_compile:
        print("Wrote " + ", ".join(str(path) for path in written) + "; skipped PDF compilation.")
        return
    for tex in written:
        pdf = compile_report(tex, ROOT / "results" / "report")
        print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
