# NKPC HSA MCMC

Bayesian state-space MCMC workflows for CES and Matsuyama-Fujiwara /
Fujiwara-Matsuyama HSA New Keynesian Phillips Curve specifications.

The canonical workflow is script based. Notebooks are kept only for exploration.

Install the local package once before running ad hoc Python commands:

```bash
python -m pip install -e .
```

The pipeline scripts also add `src/` to `sys.path`, so they can be run directly
from the project root during development.

## Reproducible Pipeline

```bash
python scripts/01_build_data.py
python scripts/02_estimate_models.py --config configs/models.yaml
python scripts/03_run_diagnostics.py
python scripts/04_prior_robustness.py
python scripts/05_period_robustness.py
python scripts/06_model_comparison.py
python scripts/07_make_tables_figures.py
python scripts/08_compile_report.py
```

For a smoke test, use `--quick` with scripts `02`, `04`, and `05`.

## Structure

- `src/nkpc_hsa/`: reusable package code.
- `scripts/`: canonical pipeline entry points.
- `configs/`: model, path, and prior YAML files.
- `data/raw/`: raw input files, grouped by source/topic.
- `data/processed/`: generated model-ready datasets.
- `results/runs/`: one folder per estimation run with posterior draws, configs, and metadata.
- `results/tables/`, `results/figures/`, `results/diagnostics/`, `results/prior_robustness/`, `results/period_robustness/`, `results/model_comparison/`, `results/report/`: generated outputs.
- `reports/`: LaTeX report sources.
- `notebooks/`: exploration only.
- `analysis/gibbs/func_gibbs/`: legacy Gibbs sampler backend used by the current adapters.
- `archive/`: old notebooks, old scripts, and old generated outputs retained for reference.

Raw data should never be overwritten. Processed data and estimation outputs are
regenerated under `data/processed/` and `results/`. Legacy outputs are retained
under `archive/legacy_results/` and are not used by the pipeline.

## Unit Conventions

Kappa-related priors in YAML files are specified in economic/physical units.
Some Gibbs samplers use internal parameters multiplied by `KAPPA_SCALE = 100`
because the regression column is divided by 100. Posterior draws saved to
`InferenceData`, exported tables, figures, SDDR outputs, and Chib marginal
likelihood inputs are in physical units.

The default HSA competition aggregator transform is:

```text
N_model = 100 * log(N_level)
```

If a supplied `N` series is already transformed, call the wrapper with
`n_transform="identity"` and verify that the run metadata records this.

The default HSA dynamic covariance convention is `e_zeta_only`: the sampler
allows correlation between the NKPC shock `e_t` and output-gap shock `zeta_t`
only. Set `covariance_structure: diagonal` or `covariance_structure: full` in
`configs/models.yaml` only when that alternative is intentional and documented.

## Coefficient Sign Constraints

By default, regression coefficients are sampled from their unconstrained
Gaussian conditional posterior, except for the AR(2) stationarity restriction
controlled by `enforce_stationary` inside the Gibbs samplers. To impose hard
coefficient restrictions, set `defaults.coefficient_constraints` in
`configs/models.yaml` or pass `--positive` to the estimation scripts.

Example smoke run with nonnegative kappa and theta:

```bash
python scripts/02_estimate_models.py --quick --positive kappa,theta
```

Example YAML:

```yaml
defaults:
  coefficient_constraints:
    enabled: true
    max_tries: 1000
    positive: [kappa, theta, kappa_0, delta]
    bounds:
      alpha: [0.0, 1.0]
```

Bounds for `kappa`, `kappa_0`, and `delta` are specified in physical units and
converted internally by the wrapper. Bounds for `theta`, `theta_0`, `gamma`,
`alpha`, `rho_1`, and `rho_2` are used as written. These constraints define a
different posterior with hard prior support; they should be reported as a
restricted robustness specification rather than silently mixed with the
unrestricted baseline.

## Robustness And Comparison

Prior robustness is controlled by `configs/priors_baseline.yaml`,
`configs/priors_weak.yaml`, and `configs/priors_tight.yaml`. Data-period
robustness is controlled by `configs/periods.yaml`; add a new named period
there and rerun `scripts/05_period_robustness.py`.

Model comparison reports SDDR for nested restrictions when posterior draws are
available, posterior predictive scores when data can be matched, and Chib
marginal likelihood only when the legacy ordinate calculation is compatible
with the model family and supplied data. WAIC/LOO are not used as primary
criteria for these latent-state models.

## Adding A Model Variant

1. Add the model implementation or adapter under `src/nkpc_hsa/models/`.
2. Expose it in `src/nkpc_hsa/inference/wrappers.py`.
3. Specify the model name in `configs/models.yaml`.
4. Add focused tests for any new unit conversions or transforms.

## Adding A Prior Specification

Create a new YAML file under `configs/`. Keep kappa, kappa_0, and delta in
physical units; the wrapper converts them to sampler-internal units.

## Report

`scripts/08_compile_report.py` writes `reports/main.tex` and compiles
`results/report/main.pdf`. Table fragments are read from `results/tables/` and
figures from `results/figures/`.
