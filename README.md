# NKPC HSA MCMC

Bayesian state-space MCMC workflows for CES and Matsuyama-Fujiwara /
Fujiwara-Matsuyama HSA New Keynesian Phillips Curve specifications.

The canonical workflow is script-based. Notebooks are kept only for exploration.

---

## Quick Start

```bash
cd /path/to/NKPC_HSA_MCMC   # move to the project root (all commands run from here)
python -m pip install -e .   # first time only: install the package locally
```

Smoke test (completes in seconds):

```bash
python scripts/01_build_data.py
python scripts/02_estimate_models.py --quick
```

---

## Full Pipeline (estimation to reports)

Run the steps below in order from the project root.
5 models (CES / hsa_steady / hsa_dynamic / hsa_full / hsa_const_theta)
x 8 data specifications = 40 Gibbs sampling runs for the baseline pass
(`n_iter=12000`, burn-in `4000`, `2` chains).

```bash
cd /path/to/NKPC_HSA_MCMC   # always move here first

# 1. Build data (writes data/processed/model_ready.csv)
python scripts/01_build_data.py

# 2. Full estimation (the slowest step: 40 Gibbs sampling runs)
python scripts/02_estimate_models.py \
    --config configs/models.yaml \
    --priors configs/priors_baseline.yaml

# 2b. Constrained estimation (kappa >= 0, compared against the
#     unrestricted runs in the report)
python scripts/02_estimate_models.py \
    --config configs/models.yaml \
    --priors configs/priors_baseline.yaml \
    --positive kappa

# Same, imposing kappa_t >= 0 at every period in time-varying-kappa models
python scripts/02_estimate_models.py \
    --config configs/models.yaml \
    --priors configs/priors_baseline.yaml \
    --positive kappa_t

# 3. MCMC convergence diagnostics (R-hat, ESS -> results/diagnostics/)
python scripts/03_run_diagnostics.py

# 4. Prior robustness (re-estimates under baseline / weak / tight priors)
python scripts/04_prior_robustness.py

# 5. Sub-sample robustness (re-estimates each period in configs/periods.yaml)
python scripts/05_period_robustness.py

# 6. Model comparison (SDDR and predictive scores -> results/model_comparison/)
python scripts/06_model_comparison.py

# 7. Tables and figures (-> results/tables/, results/figures/)
python scripts/07_make_tables_figures.py

# 8. Compile the PDF reports (-> results/report/*.pdf)
python scripts/08_compile_report.py

# 9. (Optional) identification diagnostics (-> results/diagnostics/)
python scripts/09_identification_diagnostics.py

# 10. Browsable HTML report (-> results/report.html)
python scripts/10_build_html_report.py
```

Estimation output is saved under
`results/runs/<model>_<data_spec>_<prior>_<timestamp>/posterior.nc`.

To estimate selected data specifications only:

```bash
python scripts/02_estimate_models.py --data-spec inv_markup            # inverse markup gap only
python scripts/02_estimate_models.py --data-spec output_gap_bn         # BN output gap only
python scripts/02_estimate_models.py --data-spec output_gap_bn_core    # BN output gap (core CPI)
python scripts/02_estimate_models.py --data-spec output_gap_hp         # HP output gap only
python scripts/02_estimate_models.py --data-spec output_gap_hp_core    # HP output gap (core CPI)
python scripts/02_estimate_models.py --data-spec labor_share_gap_hp    # HP labor-share gap only
python scripts/02_estimate_models.py --data-spec unemployment_gap      # unemployment gap only
python scripts/02_estimate_models.py --data-spec unemployment_gap_core # unemployment gap (core CPI)
```

---

## Reproducible Pipeline

```bash
python scripts/01_build_data.py
python scripts/02_estimate_models.py --config configs/models.yaml
python scripts/02_estimate_models.py --config configs/models.yaml --positive kappa
python scripts/03_run_diagnostics.py
python scripts/04_prior_robustness.py
python scripts/05_period_robustness.py
python scripts/06_model_comparison.py
python scripts/07_make_tables_figures.py
python scripts/08_compile_report.py
```

Smoke-test version:

```bash
python scripts/01_build_data.py
python scripts/02_estimate_models.py --quick
python scripts/03_run_diagnostics.py
python scripts/04_prior_robustness.py --quick
python scripts/05_period_robustness.py --quick
python scripts/06_model_comparison.py
python scripts/07_make_tables_figures.py
python scripts/08_compile_report.py
```

## Structure

- `src/nkpc_hsa/`: reusable package code.
- `src/nkpc_hsa/gibbs/`: Gibbs sampler backend used by the current adapters
  (moved from `analysis/gibbs/func_gibbs/`).
- `scripts/`: canonical pipeline entry points.
- `configs/`: model, path, and prior YAML files.
- `data/raw/`: raw input files, grouped by source/topic.
- `data/processed/`: generated model-ready datasets.
- `results/runs/`: one folder per estimation run with posterior draws, configs,
  metadata, and run-level support files.
- `results/tables/`, `results/figures/`, `results/model_comparison/`: report
  inputs generated from saved runs.
- `results/diagnostics/`, `results/prior_robustness/`,
  `results/period_robustness/`: robustness and diagnostic outputs.
- `results/report/`: primary PDF reports. LaTeX build byproducts are moved to
  `results/report/build/`.
- `paper/`: LaTeX report sources.
- `references/`: literature PDFs and research notes (including
  `references/notes/annual_q4_state_space.md`, the mixed-frequency methodology
  note).
- `archive/`: old notebooks, scripts, and generated outputs retained for
  reference (git-ignored).

Raw data should never be overwritten. Processed data and estimation outputs are
regenerated under `data/processed/` and `results/`. Legacy outputs are retained
under `archive/legacy_results/` and are not used by the pipeline.

## Models

All variants share the NKPC observation equation

```text
pi_t = alpha*pi_{t-1} + (1-alpha)*E_t pi_{t+1} + kappa_t*x_t - theta_t*Nhat_t + e_t
```

and differ only in the slope restrictions:

| Model | kappa_t | theta_t |
|---|---|---|
| `ces` | constant | 0 |
| `hsa_steady` | `kappa_0 + delta*Nbar_t` | 0 |
| `hsa_dynamic` | constant | constant |
| `hsa_const_theta` | `kappa_0 + delta*Nbar_t` | constant |
| `hsa_full` | `kappa_0 + delta*Nbar_t` | `theta_0 + gamma*Nbar_t` |

HSA models decompose competition as `N_obs_t = Nbar_t + Nhat_t + nu_t`, with an
AR(2) gap `Nhat_t` (truncated to the stationary region) and a random-walk
trend `Nbar_t` with drift. `hsa_const_theta` restricts `hsa_full` by fixing
`gamma = 0`; it exists because the `gamma` regressor (`Nhat*Nbar`) is nearly
collinear with the `theta_0` regressor (`Nhat`) and `gamma` is not identified.

## Unit Conventions

Kappa-related priors in YAML files are specified in economic/physical units.
Some Gibbs samplers use internal parameters multiplied by `KAPPA_SCALE = 100`
because the regression column is divided by 100. Posterior draws saved to
`InferenceData`, exported tables, figures, SDDR outputs, and Chib marginal
likelihood inputs are in physical units.

The default HSA competition aggregator transform is:

```text
N_model = (100 * log(N_level) - sample_mean(100 * log(N_level))) / 10
```

If a supplied `N` series is already transformed, call the wrapper with
`n_transform="identity"` and verify that the run metadata records this.
Under the default transform, one unit of `N_model`, `Nhat`, or `Nbar` is a
ten-log-point movement around the sample mean.

Competition measurement frequency is configured under
`defaults.competition_measurement` in `configs/models.yaml`:

```yaml
competition_measurement:
  frequency: quarterly_interpolated
  annual_timing: q4
```

`quarterly_interpolated` is the default and preserves the existing behavior:
annual competition data are transformed and PCHIP-interpolated to quarterly
frequency, and the resulting quarterly `N_obs_t` enters the N measurement
equation every quarter. `annual_q4` is a mixed-frequency robustness
specification: annual competition data are transformed but not interpolated,
the annual observation is loaded only in Q4, and Q1-Q3 are treated as missing
in the N measurement equation. In `annual_q4` runs, quarterly latent
competition states are inferred by the state-space model; the PCHIP series
shown in comparison plots is not used in estimation. See
`references/notes/annual_q4_state_space.md` for the full state-space
formulation.

Use the CLI override when needed:

```bash
python scripts/02_estimate_models.py --competition-frequency annual_q4
```

Each saved run also writes support artifacts under
`results/runs/<run_name>/report/`, `results/runs/<run_name>/tables/`, and
`results/runs/<run_name>/figures/`. The primary manuscript-style reports remain
the PDFs under `results/report/`. To build report inputs from only
mixed-frequency annual-Q4 runs:

```bash
python scripts/07_make_tables_figures.py --competition-frequency annual_q4
python scripts/08_compile_report.py
```

To include both the default quarterly-interpolated case and the annual-Q4
mixed-frequency case in the same PDF report, estimate both cases and then build
tables and figures without a competition-frequency filter:

```bash
python scripts/02_estimate_models.py --competition-frequency quarterly_interpolated
python scripts/02_estimate_models.py --competition-frequency annual_q4
python scripts/07_make_tables_figures.py
python scripts/08_compile_report.py
```

For annual-Q4 HSA runs, the PDF report includes a posterior decomposition of
competition, `N_t = Nbar_t + Nhat_t`. The full quarterly decomposition is
written to `results/tables/competition_decomposition.csv` and to the
data-spec-specific table folders.

The default HSA dynamic covariance convention is `e_zeta_only`: the sampler
allows correlation between the NKPC shock `e_t` and output-gap shock `zeta_t`
only. Set `covariance_structure: diagonal` or `covariance_structure: full` in
`configs/models.yaml` only when that alternative is intentional and documented.

Reported `delta`, `theta`, `theta_0`, and `gamma` are already in the same
ten-log-point units used during estimation. Do not multiply them by 10 again.

Because the transformed `N` series is measured in ten-log-point units, the
inverse-gamma hyperparameters for the N-state shock variances (`a_u`/`b_u`,
`a_eps`/`b_eps`) and the N measurement-error variance (`a_N`/`b_N`) must be
scaled to that unit: the observed quarterly variance of the transformed series
is roughly 0.01, so the prior scales sit in that decade (see comments in
`configs/priors_baseline.yaml`). The same applies to the `u` and `eps` entries
of `S_Sigma` for the `hsa_dynamic` inverse-Wishart prior. Earlier IG(2, 2)
settings forced these variances two orders of magnitude too large, which made
the `Nbar`/`Nhat` decomposition spuriously volatile and attenuated `delta`.

`hsa_full` includes an explicit N measurement error
(`N_obs_t = Nhat_t + Nbar_t + nu_t`, `nu_t ~ N(0, sigma_N^2)`) like the other
HSA models. Conditional on this measurement equation, its two-block state
sampler (`Nhat | Nbar`, then `Nbar | Nhat`) is an exact Gibbs step; the old
`target_scale`/`rw_scale` pseudo measurement variances have been removed.

Real-activity specifications are configured in `configs/models.yaml`.
`output_gap_BN` is the Baxter-King/BN-filtered cycle supplied in the raw output
file. `output_gap_HP` is generated by `scripts/01_build_data.py`: the HP filter
is applied to `100 * output`, where `output` is the log real-output level from
the legacy raw-data builder. The resulting HP cycle is therefore in the same
100-log-point unit as `output_gap_BN`.
`labor_share_gap_HP` is generated from `data/raw/laborshare/PRS85006173.csv` by
applying the HP filter to `100 * log(labor_share_index)`. It is treated as an
additional real-activity proxy in the same script pipeline.
The `*_core` data specifications re-estimate a given activity measure with core
CPI inflation (`pi_cpi_core`) on the left-hand side instead of headline CPI;
they exist because oil-driven headline swings in 2008Q3-2009Q4 contaminate the
GDP-based specifications.
`kappa_0` and `theta_0` are intercepts at average competition because the
default `N` transform is centered.

## Coefficient Sign Constraints

By default, regression coefficients are sampled from their unconstrained
Gaussian conditional posterior, except for the AR(2) stationarity restriction
controlled by `enforce_stationary` inside the Gibbs samplers. To impose hard
coefficient restrictions, set `defaults.coefficient_constraints` in
`configs/models.yaml` or pass `--positive` to the estimation scripts.

Example smoke run with nonnegative kappa and theta:

```bash
python scripts/02_estimate_models.py --quick --positive kappa,theta
python scripts/02_estimate_models.py --quick --positive kappa_t
```

Example YAML:

```yaml
defaults:
  coefficient_constraints:
    enabled: true
    max_tries: 1000
    positive: [kappa, theta, kappa_t]
    bounds:
      alpha: [0.0, 1.0]
```

Bounds for `kappa`, `kappa_0`, `kappa_t`, and `delta` are specified in physical
units and converted internally by the wrapper. Bounds for `theta`, `theta_0`,
`gamma`, `alpha`, `rho_1`, and `rho_2` are used as written. In HSA steady/full
models, `kappa_t` is a path restriction: each candidate draw must satisfy the
bound for every sampled period of `kappa_t = kappa_0 + delta*Nbar_t`. A
generic `--positive kappa` also applies to the time-varying kappa path in
models that do not have a scalar `kappa`. These constraints define a different
posterior with hard prior support; they should be reported as a restricted
robustness specification rather than silently mixed with the unrestricted
baseline.

## Robustness And Comparison

Prior robustness is controlled by `configs/priors_baseline.yaml`,
`configs/priors_weak.yaml`, and `configs/priors_tight.yaml`. Data-period
robustness is controlled by `configs/periods.yaml`; add a new named period
there and rerun `scripts/05_period_robustness.py`.

Model comparison reports SDDR for nested restrictions when posterior draws are
available, posterior predictive scores when data can be matched, and Chib
marginal likelihood when the legacy ordinate calculation is compatible with the
model family and supplied data. For `hsa_full`, the Chib output is a
conditional Chib calculation for the inflation equation, conditioning on
posterior mean `Nhat` and `Nbar` paths, because the full observation equation
contains the nonlinear term `gamma * Nbar_t * Nhat_t`. Chib log-ML is not
computed for `hsa_const_theta`, and log-ML values for `annual_q4` blocks are
currently invalid (the Chib routine does not handle missing N observations);
use SDDR and predictive scores there. WAIC/LOO are not used as primary
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

`scripts/08_compile_report.py` uses `paper/main.tex` and compiles the main
PDF report to `results/report/main.pdf`. It also writes one PDF per configured
data specification, such as `results/report/inv_markup.pdf`,
`results/report/output_gap_bn.pdf`, `results/report/output_gap_hp.pdf`,
`results/report/output_gap_hp_core.pdf`,
`results/report/labor_share_gap_hp.pdf`, and
`results/report/unemployment_gap.pdf`. These PDFs are the primary report
format.

Table fragments are read from `results/tables/` and figures from
`results/figures/`. Baseline model results are grouped by data specification,
competition measurement frequency, prior, period, and coefficient-constraint
condition, so the quarterly-interpolated and annual-Q4 cases can appear in the
same PDF. Prior-set robustness, sample-period robustness, and annual-Q4 HSA
competition decomposition are displayed in separate tables and figures. LaTeX
auxiliary files are moved to `results/report/build/` after compilation so the
report folder primarily contains PDFs.

`scripts/10_build_html_report.py` builds a single browsable HTML version of
the same content (`results/report.html`): a sidebar per activity measure, one
section per result block with coefficient/SDDR/model-comparison tables, and a
figure selector with click-to-enlarge.
