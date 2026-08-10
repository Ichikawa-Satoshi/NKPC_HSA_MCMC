# Repository and production call chain before the theory-model migration

Snapshot revision: `c9c7ddff026690188235ab3d5c3454364fa2b84b` (`main`)

This map records the production system as it existed before the fixed-reference /
moving-reference model hierarchy was added. It was reconstructed by following the
executable entry points and imports; it is not inferred from model names.

## End-to-end call chain

| Stage | Production entry point | Called implementation | Output / consumer |
|---|---|---|---|
| Raw data | `data/raw/*` under the configured Dropbox root | Source-specific readers in `src/nkpc_hsa/dataprep/func_data_build.py` | Quarterly frames |
| Processed data | `scripts/01_build_data.py` | `dataprep.build.build_processed_dataset` -> `func_data_build.build_dataset`, HP and labor-share extensions | `data/processed/model_ready.csv` |
| Data-cell selection | `scripts/02_estimate_models.py` | `config.load_model_config` and `configured_data_specs`; `_coerce_model_data` applies the sample window and joint complete-case selection | `{pi, pi_prev, pi_expect, x, x_prev, N}` |
| Inflation construction | `func_data_build.load_monthly_inflation_series` | Headline CPI/PPI: `100*(P/P[-4]-1)`; core CPI: `100*(log(P)-log(P[-4]))` | Four-quarter YoY percentage-point series |
| Expectations | `func_data_build.load_cleveland_fed_expectations` | Raw `Epi` is multiplied by 100 and quarterly-averaged | `Epi`; configured for every production data cell |
| Competition transform | `inference.wrappers._prepare_competition_measurement` | `transform_competition_series(..., "log100_centered10")`: `(100 log N - sample mean)/10`; annual-Q4 or quarterly-interpolated observation construction | `N_obs` supplied to HSA samplers |
| Estimation wrapper | `scripts/02_estimate_models.py` | `inference.wrappers.run_model` -> `_run_sampler` | One or more chains |
| Registry / dispatch | Hard-coded dictionary in `inference.wrappers._run_sampler` | See model table below | Sampler result mappings |
| Prior / scaling boundary | `models.common.prior_specs_to_internal` | Kappa-like priors multiplied by `KAPPA_SCALE=100`; sampler regressors use `x/100`; stored draws divided by 100 | Physical-unit posterior coefficients |
| State dispatch | Per-model sampler | CES: none; steady/dynamic/const-theta: joint FFBS; full: Particle Gibbs through `models/hsa_full.py` -> `gibbs/hsa_full_pg` | State paths and Particle-Gibbs diagnostics |
| Posterior storage | `inference.wrappers._extract_draws_from_result`, `_stack_chains`, `_to_idata`, `_save_run` | Normalizes names and variances, creates ArviZ `InferenceData` | `results/runs/<cell>/posterior.nc`, `metadata.json`, `priors.json`, `data_spec.json` |
| Diagnostics | `scripts/03_run_diagnostics.py` | `inference.diagnostics` plus ArviZ R-hat / ESS | `results/diagnostics/*` |
| Report tables / figures | `scripts/12_build_cpi_ppi_report.py` and `make_*` scripts | `_load_runs` selects only `ESTIMATION_REVISION`, observation frequency, sample/model cell, and unrestricted legacy constraints | `results/tables/*`, `results/figures/*` |
| Report builder | `scripts/build_report.py` | Ordered `STEPS`, then LaTeX compile of `report/nkpc_hsa_report.tex` | `report/nkpc_hsa_report.pdf` |

## Pre-migration model slugs and executable definitions

| Slug | Public facade | Dispatched sampler | Pre-migration equation role |
|---|---|---|---|
| `ces` | `src/nkpc_hsa/models/ces.py` | `gibbs/ces/model.py::func_nkpc_ces` | Constant-slope, no competition state |
| `hsa_steady` | `models/hsa_steady.py` | `gibbs/hsa_steady/model.py::func_nkpc_hsa_decomp_tv_kappa_noerror` | `kappa_t=kappa_0+delta*Nbar_t`, no entry term |
| `hsa_dynamic` | `models/hsa_dynamic.py` | `gibbs/hsa_dynamic/model.py::func_nkpc_hsa_decomp` | Constant `kappa`, constant `theta`, correlated-shock state model |
| `hsa_const_theta` | `models/hsa_const_theta.py` | `gibbs/hsa_const_theta/model.py::func_nkpc_hsa_const_theta` | Moving `kappa_t`, constant `theta`, `gamma=0` |
| `hsa_full` | `models/hsa_full.py` | `gibbs/hsa_full_pg/model.py::func_nkpc_hsa_full_pg` | Moving `kappa_t` and `theta_t`, Particle Gibbs |

The slugs were listed in `configs/models.yaml::models` and duplicated in the
hard-coded dispatch dictionary. There was no declarative model registry.

## Pre-migration posterior schema

Scalar names were normalized at the wrapper boundary. Common fields included
`alpha`, `kappa` or `kappa_0`, legacy `delta`, `theta` or `theta_0`, `gamma`,
`phi_1`, `lambda_ez`, `rho_1`, `rho_2`, `n`, and standard deviations. State and
derived paths were `Nbar`, `Nhat`, `kappa_t`, and `theta_t`; Particle Gibbs also
stored `pg_ess_mean`, `pg_ess_min`, and `pg_moved_frac`. The new theoretical
names `kappa_N_empirical`, `d_kappa_d_logN`, and `zeta0` did not exist.

## Pre-migration convergence and artifact provenance

The report used `max R-hat <= 1.01` and `min bulk ESS >= 400`, reported for
scalar, state, and derived-path groups. Run selection checked the legacy
`ESTIMATION_REVISION`, iteration count, period, legacy constraint label,
competition measurement frequency, and `n_transform`. It did not validate an
exact mathematical-restriction manifest, the structural inflation frequency,
the inflation observation design, the expectation information date/horizon,
the activity mapping, or an artifact content signature. Therefore revision
filtering protected known sampler/data changes, but was not sufficient for the
new restriction hierarchy.

## Historical result locations

The configured results root is external to the checkout:
`~/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/results`. At this snapshot it
contained 138 run metadata files under `results/runs`, plus `tables`, `figures`,
`diagnostics`, `evidence`, `period_robustness`, `prior_decomposition`,
`error_robustness`, `extensions`, `audit`, and `each_result`. These artifacts
belong to the legacy revision namespace and are retained in place.

## Known pre-migration semantic boundary

All configured inflation columns were four-quarter YoY transformations, while
the sampler interpreted their regression coefficients directly. Consequently,
the old estimates are reduced-form / ablation evidence. They cannot be renamed,
rescaled, or promoted into evidence for a quarterly structural
Rotemberg-HSA cross-equation restriction.
