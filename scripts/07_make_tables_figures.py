from __future__ import annotations

from pathlib import Path

import argparse
import arviz as az
import pandas as pd
import yaml

from _bootstrap import ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.report.figures import (
    save_kappa_model_comparison,
    save_placeholder_figure,
    save_posterior_density,
    save_posterior_predictive_placeholder,
    save_prior_posterior_overlay,
    save_time_varying_path,
)
from nkpc_hsa.report.tables import (
    coefficient_means_table,
    kappa_comparison_table,
    posterior_summary_table,
    sddr_summary_table,
    time_varying_coefficients_table,
    write_latex_fragment,
)


def _load_prior(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_idata_by_run(runs_dir: str | Path) -> dict[str, object]:
    idata_by_run = {}
    for posterior in sorted(Path(runs_dir).glob("*/posterior.nc")):
        idata_by_run[posterior.parent.name] = az.from_netcdf(posterior)
    return idata_by_run


def _filter_by_data_spec(idata_by_run: dict[str, object], data_spec: str | None) -> dict[str, object]:
    if data_spec is None:
        return idata_by_run
    return {
        run: idata
        for run, idata in idata_by_run.items()
        if str(getattr(idata, "attrs", {}).get("data_spec", "")) == data_spec
    }


def _write_existing_table(
    csv_path: Path,
    tex_path: Path,
    *,
    data_spec: str | None,
    note: str,
) -> None:
    if not csv_path.exists():
        write_latex_fragment(pd.DataFrame({"note": [note]}), tex_path)
        return
    table = pd.read_csv(csv_path)
    if data_spec is not None and "data_spec" in table.columns:
        table = table[table["data_spec"].astype(str) == data_spec]
    if table.empty:
        table = pd.DataFrame({"note": [note]})
    write_latex_fragment(table, tex_path)


def _make_outputs(
    idata_by_run: dict[str, object],
    *,
    priors: dict,
    tables_dir: Path,
    figures_dir: Path,
    data_spec: str | None = None,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    first_idata = None
    idata_by_run = _filter_by_data_spec(idata_by_run, data_spec)
    for run_name, idata in sorted(idata_by_run.items()):
        if first_idata is None:
            first_idata = idata
        table = posterior_summary_table(idata)
        if not table.empty:
            table.insert(0, "run", run_name)
            table.insert(1, "model", getattr(idata, "attrs", {}).get("model", ""))
            table.insert(2, "data_spec", getattr(idata, "attrs", {}).get("data_spec", ""))
            table.insert(3, "prior_spec", getattr(idata, "attrs", {}).get("prior_spec", ""))
            table.insert(4, "constraint_spec", getattr(idata, "attrs", {}).get("constraint_spec", "unrestricted"))
            rows.append(table)
    summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"note": ["No posterior summaries available."]})
    summary.to_csv(tables_dir / "posterior_summary.csv", index=False)
    write_latex_fragment(summary, tables_dir / "posterior_summary.tex")

    coeff = coefficient_means_table(idata_by_run)
    coeff_out = coeff if not coeff.empty else pd.DataFrame({"note": ["No coefficient posterior means available."]})
    coeff_out.to_csv(tables_dir / "coefficient_means.csv", index=False)
    write_latex_fragment(coeff_out, tables_dir / "coefficient_means.tex")

    kappa = kappa_comparison_table(idata_by_run)
    kappa_out = kappa if not kappa.empty else pd.DataFrame({"note": ["No kappa draws available for model comparison."]})
    kappa_out.to_csv(tables_dir / "kappa_model_comparison.csv", index=False)
    write_latex_fragment(kappa_out, tables_dir / "kappa_model_comparison.tex")
    if not save_kappa_model_comparison(kappa, figures_dir / "kappa_model_comparison.png"):
        save_placeholder_figure(figures_dir / "kappa_model_comparison.png", "No kappa draws available yet.")

    tv = time_varying_coefficients_table(idata_by_run)
    tv_out = tv if not tv.empty else pd.DataFrame({"note": ["No time-varying kappa_t/theta_t draws available."]})
    tv_out.to_csv(tables_dir / "time_varying_coefficients.csv", index=False)
    write_latex_fragment(tv_out, tables_dir / "time_varying_coefficients.tex")
    if not save_time_varying_path(idata_by_run, "kappa_t", figures_dir / "kappa_t_path.png"):
        save_placeholder_figure(figures_dir / "kappa_t_path.png", "No kappa_t path available. Run HSA steady or HSA full.")
    if not save_time_varying_path(idata_by_run, "theta_t", figures_dir / "theta_t_path.png"):
        save_placeholder_figure(figures_dir / "theta_t_path.png", "No theta_t path available. Run HSA full.")

    sddr = sddr_summary_table(idata_by_run, priors)
    sddr_out = sddr if not sddr.empty else pd.DataFrame({"note": ["No SDDR restrictions available for current runs."]})
    sddr_out.to_csv(tables_dir / "sddr.csv", index=False)
    write_latex_fragment(sddr_out, tables_dir / "sddr.tex")

    if first_idata is not None:
        for var in ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]:
            if var in first_idata.posterior:
                save_posterior_density(first_idata, var, figures_dir / f"posterior_density_{var}.png")
        for var in ["kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]:
            if var in priors:
                prior = priors[var]
                if not isinstance(prior, dict):
                    prior_tuple = (float(prior[0]), float(prior[1]))
                else:
                    prior_tuple = (float(prior["mean"]), float(prior["sd"]))
                for idata in idata_by_run.values():
                    if var in getattr(idata, "posterior", {}):
                        save_prior_posterior_overlay(
                            idata,
                            var,
                            prior_tuple,
                            figures_dir / f"prior_posterior_{var}.png",
                        )
                        break
            if not (figures_dir / f"prior_posterior_{var}.png").exists():
                save_placeholder_figure(
                    figures_dir / f"prior_posterior_{var}.png",
                    f"No posterior draws for {var} available yet.",
                )
    for var in ["kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]:
        path = figures_dir / f"prior_posterior_{var}.png"
        if not path.exists():
            save_placeholder_figure(path, f"No posterior draws for {var} available yet.")
    save_posterior_predictive_placeholder(figures_dir / "posterior_predictive_placeholder.png")

    _write_existing_table(
        ROOT / "results" / "tables" / "mcmc_diagnostics.csv",
        tables_dir / "mcmc_diagnostics.tex",
        data_spec=data_spec,
        note="Run scripts/03_run_diagnostics.py first.",
    )
    _write_existing_table(
        ROOT / "results" / "tables" / "prior_robustness.csv",
        tables_dir / "prior_robustness.tex",
        data_spec=data_spec,
        note="Run scripts/04_prior_robustness.py first.",
    )
    _write_existing_table(
        ROOT / "results" / "tables" / "period_robustness.csv",
        tables_dir / "period_robustness.tex",
        data_spec=data_spec,
        note="Run scripts/05_period_robustness.py first.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--priors", default=str(ROOT / "configs" / "priors_baseline.yaml"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument(
        "--data-spec",
        action="append",
        dest="data_specs",
        help="Data spec name to render. Repeat for multiple. Defaults to config run_data_specs.",
    )
    parser.add_argument("--combined-only", action="store_true")
    args = parser.parse_args()

    config = load_model_config(args.config)
    data_specs = configured_data_specs(config, args.data_specs)
    idata_by_run = _load_idata_by_run(args.runs_dir)
    priors = _load_prior(args.priors)

    _make_outputs(
        idata_by_run,
        priors=priors,
        tables_dir=ROOT / "results" / "tables",
        figures_dir=ROOT / "results" / "figures",
        data_spec=None,
    )
    if not args.combined_only:
        for data_spec_name in data_specs:
            _make_outputs(
                idata_by_run,
                priors=priors,
                tables_dir=ROOT / "results" / "tables" / data_spec_name,
                figures_dir=ROOT / "results" / "figures" / data_spec_name,
                data_spec=data_spec_name,
            )


if __name__ == "__main__":
    main()
