# NKPC HSA MCMC

Bayesian state-space MCMC for New Keynesian Phillips Curves whose slope depends
on **competition**. The competition channel follows the HSA (homothetic
single-aggregator) demand system of Fujiwara–Matsuyama: when the effective
number of firms falls, markups rise, pass-through falls, and the Phillips curve
flattens.

The empirical question is whether that flattening is visible in US data — that
is, whether **δ > 0** in `κ_t = κ₀ + δ·N̄_t`, where `N̄_t` is a latent
low-frequency firm-count trend estimated jointly with the Phillips curve.

The workflow is script-based and reproducible from the project root.

---

## Quick start

```bash
python -m pip install -e .
```

The package lives under `src/`, so the editable install is what makes
`import nkpc_hsa` work — `pytest` and every script rely on it.

```bash
python main_scripts/01_build_data.py
python -m pytest -q
```

---

## Repository structure

The repository has three layers: a **shared library** (`src/nkpc_hsa/`), the
**production** pipeline (`main_scripts/`), and the **tests** — one self-contained
experiment bundle per subdirectory of `tests/`.

```
NKPC_HSA_MCMC/
├── src/nkpc_hsa/     SHARED library — imported by production and every test, never copied
│     config.py  paths.py  progress.py  provenance.py  theory_models.py
│     dataprep/   raw series -> data/processed/model_ready.csv
│     gibbs/      the Gibbs/FFBS sampler engine (one subpackage per model)
│     models/     thin public facade over gibbs/
│     inference/  run_model, diagnostics, robustness, identification
│     reporting/  tables, figures and LaTeX fragments
│     error_robustness/   MA(3)-disturbance variants of every sampler + runner
│     phillips/   shared Phillips-curve toolkit (was nine_cell/): data, state,
│                 inflation, estimation, joint, temporal, markup_measurement
│
├── main_scripts/    本推定 — the canonical estimation pipeline, run from the repo root
│     01–19_*.py      build data → estimate → diagnose → robustness → comparison → report
│     build_report.py, build_restriction_report.py    the two report builders
│     run_restriction_production.py    drives the theory F0/U/R1/R2/R3 sequence
│     make_*.py        table/figure generators;   _bootstrap.py   shared path setup
│
├── tests/           テスト — one self-contained experiment bundle per subdirectory
│     __init__.py  _bootstrap.py       (bundles import as `tests.<name>`)
│     <name>/         observed_hhi · markup_measurement · markup_full_joint ·
│       functions.py    markup_feedback · markup_interpolation · nolag_price_gap ·
│       run.py           n_gustavo_state_space · design
│       config.yaml    (design/ has none — it uses configs/nine_cell_design.yaml)
│       results/       report, tables/, figures/, draws/ — INSIDE the bundle, git-ignored
│       README.md      what the experiment asks and how to run it
│
├── configs/         SHARED settings: models.yaml, priors_*.yaml, periods.yaml,
│                    nine_cell_design.yaml (nkpc_hsa.phillips.load_design_data reads it),
│                    error_robustness.yaml
├── data/            raw/ (never written by code) · processed/ (built by main_scripts/01)
├── results/         SHARED production outputs (symlink to Dropbox); git-ignored.
│                    Experiment outputs do NOT go here — they live in each bundle's results/.
├── report/          two deliverables: nkpc_hsa_report + the restriction report (source + PDF)
├── docs/            spec, code/equation crosswalk, estimation flow, data dictionary, audits
└── references/      literature PDFs and research notes
```

Everything a report PDF needs is generated under `results/`, so `report/` holds
only the source and the output.

---

## Operating method (運用方法)

The repository separates three concerns; keep new work in the right layer.

**Shared library — `src/nkpc_hsa/`.** All reusable code: the sampler engine
(`gibbs/`), data building (`dataprep/`), and the Phillips-curve toolkit
(`phillips/`). Both production and every test import from here — never copy code
into a bundle. `pip install -e .` makes `import nkpc_hsa` work everywhere.

**Production — `main_scripts/`.** The canonical, reproducible pipeline: the
numbered scripts `01–19`, the two report builders, and
`run_restriction_production.py`. Run from the repo root, e.g.:

```bash
python main_scripts/01_build_data.py
python main_scripts/run_restriction_production.py     # theory F0/U/R1/R2/R3
python main_scripts/build_report.py --compile         # historical report
```

Production writes to the shared `results/` (a Dropbox symlink) and compiles the
reports in `report/`. `main_scripts/_bootstrap.py` finds the repo root (walking up
to `pyproject.toml`) and puts `src/` on the path, so every script sees the same
`ROOT` / `DATA_DIR` / `RESULTS_DIR`.

**Per-run result PDF.** For a standardized one-document-per-cell view of the saved
posteriors in `results/runs/`, run:

```bash
python main_scripts/18_build_each_result.py --compile
```

Each cell PDF (under `results/each_result/<cell>/`) collects: the posterior
coefficient table, prior-vs-posterior panels for every parameter, the time-varying
`κ_t` / `θ_t` paths, the competition-state decomposition, the LOO/WAIC precision
comparison, and the R-hat / bulk-ESS convergence table. Omit `--compile` to write
only the `.tex` + tables + figures.

**Tests — `tests/<name>/`.** Each subdirectory is a self-contained experiment
("テスト"): `functions.py` (its own estimation functions — omitted when it reuses
only `nkpc_hsa.phillips`), `run.py` (the runner), `config.yaml` (its settings),
`results/` (its outputs), and `README.md` (its description). Run one with:

```bash
python tests/observed_hhi/run.py --quick     # smoke
python tests/observed_hhi/run.py             # full
```

Every `run.py` self-bootstraps (it finds the repo root via `pyproject.toml`), so it
runs from any directory. Bundles import as `tests.<name>` and pull the heavy
engine from `nkpc_hsa`; they never write outside their own `results/`, which is
git-ignored and reproducible — including the raw MCMC `draws/`.
[`tests/README.md`](tests/README.md) indexes every experiment (what each one asks,
its inputs, and how they depend on each other).

**Adding an experiment.** Copy an existing bundle directory under `tests/`, keep
the four files (`functions.py` / `run.py` / `config.yaml` / `README.md`), import
shared pieces from `nkpc_hsa`, and write outputs to the bundle's own `results/`.
Put genuinely shared logic in `nkpc_hsa/phillips/`, not in the bundle.

**One data set, many indicators.** `data/processed/model_ready.csv` carries every
series (CPI / PPI / core, markup, output- and unemployment-gaps, HHI, …).
Production and each experiment select the columns they need through their config,
so the data is never forked per experiment.

---

## Running the estimation

### Nine-cell design in `report/design.tex`

The revised identification-first design is the `tests/design/` bundle. It
uses the existing Dropbox data root and the repository's path/config helpers, and
keeps its output inside the bundle at `tests/design/results/`:

```bash
python main_scripts/15_build_extension_data.py
python tests/design/run.py --test-run --compile
```

The test run completes the full data/state/cell/report call chain with deliberately
short chains and stamps every output **NOT FOR INFERENCE**. It runs all nine cells,
E0--E2, quarterly annualized inflation, exact `A4`-aggregated YoY, endpoint-matched
QoQ, the measurement-only cut, a secondary QoQ full joint, and the fast-by-fast and
CS1/CS2 smoke sensitivities. The polished validation PDF is
`report/nine_cell_design_report.pdf`.

A long-chain core run is selected by omitting `--test-run`; it is not labeled a
complete production design while mandatory robustness/Chib modules remain outstanding.
The report and `tables/design_compliance.csv` do not silently mark unexecuted gates as
passed: absent real-time vintages, PPI-specific one-quarter expectations, an externally
identified inverse-markup scale, and omitted modules remain explicit limitations.

### Reproducing the report

The headline result is **HSA steady, core CPI, negative unemployment gap,
mixed-frequency**. End to end:

```bash
python main_scripts/01_build_data.py
python main_scripts/13_estimate_cpi_ppi_report.py --jobs 6 --competition-frequency quarterly_interpolated --no-build
python main_scripts/13_estimate_cpi_ppi_report.py --jobs 6 --competition-frequency annual_q4 --compile
```

Step 2 estimates the 77 interpolated cells, step 3 the 61 mixed-frequency ones,
and step 3 then builds every report artifact and compiles the PDF. Both
estimation steps are expensive — 12000 iterations × 2 chains per cell, about
90 minutes each at `--jobs 6`. Use `--quick` for a smoke test. Existing runs of
the current revision are reused; `--force` re-estimates them.

`--no-build` on the first step is what keeps it from building a report while
half the run set is still missing.

### Progress display

The estimation drivers show live progress on a terminal and nothing extra when
their output is redirected, so a log file does not fill up with redraws:

| command | what the bar measures |
|---|---|
| `main_scripts/run_restriction_production.py` | one line per theory cell plus an overall bar, advancing **per draw** — each cell reports its own iteration count back to the driver |
| `main_scripts/10_estimate_theory_models.py` | a single per-draw bar spanning every chain of the cell |
| `main_scripts/13_estimate_cpi_ppi_report.py` | one bar over the cell set, advancing **per completed cell** — the cells run in separate worker processes with no channel back |

All three take `--progress {auto,bar,plain,stream,off}`, and `NKPC_HSA_PROGRESS`
sets the default. Use `plain` for a periodic summary line under `nohup` or in
CI; `stream` emits the machine-readable events the production driver
aggregates, and is not meant to be read by a person.

The bar is display only. It never touches the draws, the seeds or the saved
run, so a run estimated with it on and the same run with it off are identical.

### No-lag price/output-gap robustness grid

The N_Gustavo-only quarterly state can be held fixed while lagged inflation is
removed and persistent inflation is assigned to an AR(1) error. The robustness
driver crosses five price indices, four activity/slack measures, four empirical
competition-channel models, and six fast-state timings:

```bash
python tests/nolag_price_gap/run.py --quick
python tests/nolag_price_gap/run.py
python tests/nolag_price_gap/run.py --theory-faithful
```

The production command reuses the retained 20,000-iteration/5,000-warmup state
posterior written by `tests/n_gustavo_state_space/run.py`; it does not
smooth the competition state again using inflation. The stable PDF is written
to `output/pdf/nolag_price_gap_model_tests.pdf`. The `--theory-faithful`
variant fixes the standalone empirical slow-level nuisance `psi` exactly to
zero and writes `output/pdf/theory_faithful_nolag_model_tests.pdf`.

### Rebuilding the report without re-estimating

```bash
python main_scripts/build_report.py --compile
```

`build_report.py` is the **single entry point for every report artifact**, and
the only place that knows the order they must run in. It runs, in sequence:

| script | produces |
|---|---|
| `12_build_cpi_ppi_report.py` | most tables, all result macros, most figures; itself chains `make_spec_tables.py` and `11_additional_report_evidence.py` |
| `make_headline_results_table.py` | `headline_results.tex`, `model_comparison_unemp.tex`, `ppi_results.tex` |
| `predictive_comparison.py` | `results/evidence/tables/predictive_comparison.csv` |
| `make_fit_comparison_table.py` | `fit_comparison.tex` and its macros — **reads the CSV above, so order matters** |
| `make_data_series_figure.py` | `data_series.png` (from `data/processed/`, not from the runs) |

Add an artifact to `STEPS` in that script, not to a private habit of running
things by hand. Before this existed, only the first of the five was wired into
the estimation pipeline, so a re-estimation refreshed most tables and silently
left the headline table, the fit comparison and the data figure at their
previous vintage — no error, and a PDF that compiled cleanly while disagreeing
with its own run set.

`--compile` runs xelatex twice and then fails if the log contains any undefined
control sequence, which is how a missing generated macro is caught rather than
shipped. `--skip-predictive` reuses the existing scores instead of recomputing
them. `12_build_cpi_ppi_report.py` fails loudly if a report cell is missing —
pass `--allow-incomplete` to build partial tables anyway.

The 12 estimations behind `main_scripts/prior_decomposition_rho_delta.py` are deliberately
**not** part of the report build (~45 minutes). `build_report.py` runs that script with
`--macros-only`, which refreshes `prior_decomposition_macros.tex` from the existing
factorial CSVs without re-estimating them. Run the script directly when the sweep changes.

### The two observation designs

The firm count is **annual**; inflation is quarterly. Two ways to reconcile them:

| design | what it does | how to select |
|---|---|---|
| `annual_q4` | **main.** N is observed in Q4 only; in other quarters the Kalman filter drops that observation row | default, declared in `configs/models.yaml` |
| `quarterly_interpolated` | N is PCHIP-interpolated to quarterly and treated as observed every quarter | `--competition-frequency quarterly_interpolated` |

Interpolation manufactures information. It drives the AR(2) mean-reversion term
`1−ρ₁−ρ₂` toward zero and induces a near-perfect negative correlation between
the two firm-count states — `corr(N̄₀, N̂₀) = −0.9996`, against `+0.13` under
`annual_q4`. The report keeps both designs so the artifact stays visible.

### Experimental quarterly-establishment identification

The restored BDS/BED files support a separate HSA const-theta experiment aimed at
the weakly identified quarterly `Nhat` AR(2) and `theta` blocks. BED establishment births and deaths
are quarterly flows, so `main_scripts/01_build_data.py` reconstructs the end-of-quarter
stock from 1993Q2 onward. The experimental data spec uses exactly 1993Q2–2012Q4
(79 quarters), decomposes transformed `E` with an HP filter, and adds

```text
Ehat_obs_t = lambda_E * Nhat_t + omega_t
```

to the joint Kalman/FFBS update. `lambda_E`, `sigma_E`, and the inflation loading
`theta` are sampled rather than fixed. HSA const-theta is used because HSA steady
sets `theta=0` and would identify only a nuisance state. These runs are written
outside the production report run set.

The current pilot is initialization-sensitive because a second mode can set
`lambda_E` near zero and absorb the entire establishment cycle in `sigma_E`.
Treat it as a model-development diagnostic, not as identified evidence for
`theta`; the competing regions are documented in the experiment note.

```bash
python main_scripts/01_build_data.py
python main_scripts/14_estimate_establishment_augmented.py --quick --chains 1 --no-save
python main_scripts/14_estimate_establishment_augmented.py
```

The first command is required after changing the raw BED/BDS inputs. See
[`docs/establishment_identification.md`](docs/establishment_identification.md)
for the stock convention, equations, and current limitations.

### Supporting analyses (not part of the report build)

```bash
python main_scripts/03_run_diagnostics.py             # trace / autocorr plots, R-hat, ESS per run
python main_scripts/09_identification_diagnostics.py
python main_scripts/chib_marginal_likelihood.py       # conditional marginal likelihood
python main_scripts/appendix_particle_gibbs_hsa_full.py validate|pilot|produce
python main_scripts/er_01_diagnose.py                  # overlapping-inflation error diagnostics
python main_scripts/er_02_estimate.py --quick          # MA(3) error-structure smoke test
```

### Observed inverse-HHI model tests

The observed-HHI experiment bypasses the annual-firm/QCEW common factor and
uses the quarterly SEC inverse HHI directly. It compares fast-component
definitions, lags, HHI aggregators, inflation-error models, constant/varying
direct loadings, the calibrated HSA cross-equation restriction, and a
deterministic predicted-level/innovation decomposition of the same observed
HHI series. The latter is not a latent measurement model.

```bash
python tests/observed_hhi/run.py --jobs 4
python tests/observed_hhi/run.py --reuse-existing
```

The formal PDF is written to `output/pdf/observed_hhi_model_tests.pdf`; full
tables and figures are written under `results/observed_hhi_model_tests/`.

### Interim primary: N_Gustavo-only quarterly state space

Until a validated Capital IQ or LSEG quarterly HHI is added, the mandatory
interim primary specification uses annual `N_Gustavo` alone. Annual values are
observed at Q4; Q1--Q3 are missing observations in a mixed-frequency
state-space model. QCEW, SEC HHI, deterministic quarterly interpolation, and
inflation feedback into the state smoother are excluded.

```bash
python tests/n_gustavo_state_space/run.py
python tests/n_gustavo_state_space/run.py --reuse-existing
```

The formal PDF is written to
`output/pdf/n_gustavo_state_space_tests.pdf`; machine-readable results are
stored under `results/n_gustavo_state_space_tests/`.

`03_run_diagnostics.py` writes one directory per run under `results/diagnostics/`,
matching the run directory names one-to-one. Re-run it after re-estimating, or
its plots describe a previous vintage.

Two older scripts are **superseded and should not be run casually**:

- `04_prior_robustness.py` re-estimates the prior sweep on its own. The sweep the
  report uses is part of the main run set instead (weak and tight cells are 24 of
  the 61 mixed-frequency cells), built into `prior_sensitivity_*.tex` by
  `12_build_cpi_ppi_report.py`. Running script 04 produces a second, parallel set
  of estimates that nothing reads.
- `06_model_comparison.py` computes Chib marginal likelihoods. Section 11 of the
  report states these are computed but not promoted into any table.

`rerun_*.py` re-estimate specific model families into their own run directories.

### What is in `results/`

| directory | written by | notes |
|---|---|---|
| `runs/` | `13_estimate_cpi_ppi_report.py` | one directory per (model, data spec, prior, observation design). **No timestamp** — a cell is re-estimated in place, and the estimation time lives in `metadata.json` as `run_id`. The observation design is part of the name because without it the two designs of the same cell collide. |
| `diagnostics/` | `03_run_diagnostics.py` | one directory per run |
| `tables/`, `figures/` | `12_build_cpi_ppi_report.py`, the `make_*` scripts | what the report `\input`s and `\includegraphics`es; `annual_q4/` is the mixed-frequency design |
| `evidence/tables/` | `predictive_comparison.py` | build intermediate: the scores `make_fit_comparison_table.py` reads. Recreated by every `build_report.py` run |
| `prior_decomposition/` | `prior_decomposition_rho_delta.py` | 12 diagnostic cells, deliberately outside `runs/` so they cannot enter the report run-set |

The whole tree is git-ignored and reproducible from the scripts. Runs from a
superseded `ESTIMATION_REVISION` may remain on disk for comparison, but stay out of the
current report: the revision string in each run's `metadata.json` records which vintage
of the inputs produced it, and both the estimator and the report builder select on it.
The estimator also verifies `n_obs`, `sample_start`, and `sample_end` before reusing a
cell.

---

## Data

Every user must explicitly configure the GitHub checkout and Dropbox storage
roots before running any script:

```bash
export NKPC_HSA_PROJECT_DIR="/Users/satoshi/GitHub/NKPC_HSA_MCMC"
export NKPC_HSA_DROPBOX_DIR="/Users/satoshi/Dropbox/NKPC_HSA_MCMC"
```

There is no path auto-detection or fallback. Code and configuration paths are
resolved relative to `NKPC_HSA_PROJECT_DIR`; raw and processed data use
`<NKPC_HSA_DROPBOX_DIR>/data`; sampling and report outputs use
`<NKPC_HSA_DROPBOX_DIR>/results`.

`main_scripts/build_report.py` creates or verifies the repository-local `results`
symlink needed by the LaTeX source's relative paths.

`main_scripts/01_build_data.py` turns `data/raw/` into
`data/processed/model_ready.csv`. Each production estimation cell selects six
columns and drops missing values **jointly**, so the report specifications have
**T = 124** quarters, 1982Q1–2012Q4. The establishment-augmented experimental
specification is deliberately shorter: 79 quarters, 1993Q2–2012Q4.

The committed model-ready file has 454 rows; the explicitly configured estimation
window and complete-case selection reduce every production specification to those 124
quarters.

| role | series | source |
|---|---|---|
| `π_t` | headline CPI (`pi_cpi`), core CPI (`pi_cpi_core`), PPI (`pi_ppi`) | `CPIAUCSL`, `CPILFESL`, `PPIACO` |
| `E_tπ_{t+1}` | `Epi` | Cleveland Fed inflation expectations |
| `x_t` | negative unemployment gap (`unemp_gap`), BN output gap, HP output gap, HP labor-share gap, inverse markup | CBO `NROU` − `UNRATE` (SA); BN filter output; HP filter (λ=1600) computed in `dataprep/build.py` |
| `N_t` | `N_Gustavo` (inverse HHI of US listed firms), `N_TNIC` | `BN_N_Gustavo_26.csv`, `BN_N_TNIC_26.csv` |
| `E_t` (experiment) | reconstructed quarterly establishment stock | annual BDS `ESTAB` anchor plus quarterly BED births minus deaths |

Two conventions that are easy to get wrong:

- **`x_t` is the *negative* unemployment gap**, `u* − u`. It is positive in
  booms, so a **positive κ is a conventionally-signed, downward-sloping**
  Phillips curve.
- **`N` is transformed to ten-log-point deviations**: `(100·log N − mean)/10`
  (`n_transform="log100_centered10"`). `δ`, `θ`, `θ₀` and `γ` are therefore
  *per ten log points*, and are already reported on that scale — do not rescale
  them again.

Every column — construction, units, provenance, and the series that are loaded
but unused — is documented in
[`docs/data_dictionary.md`](docs/data_dictionary.md).

**Known caveat.** Every inflation series is a four-quarter change sampled
quarterly, so `π_t` and `π_{t−1}` share three of four quarters. Even
white-noise quarterly inflation would give `corr(π_t, π_{t−1}) = 0.75`; in the
data it is 0.97. The lagged-inflation coefficient `α ≈ 0.79` therefore absorbs
both genuine inertia and this overlap, and the likelihood treats the residual as
i.i.d. when it cannot be (Ljung–Box(8) p = 0.00015).

---

## Models

All five share the inflation equation

```
π_t = α·π_{t−1} + (1−α)·E_tπ_{t+1} + κ_t·x_t − θ_t·N̂_t + λ_eζ·ζ_t + e_t
```

and differ in whether `κ_t` and `θ_t` move with competition. The HSA models
decompose the firm count as `N^obs_t = N̄_t + N̂_t + ν_t`, with `N̄_t` a
random-walk-with-drift trend and `N̂_t` an AR(2) cycle truncated to the
stationary region.

| model | slope `κ_t` | cycle loading `θ_t` | state sampler | code |
|---|---|---|---|---|
| `ces` | `κ` | — | no firm-count state | [`gibbs/ces/`](src/nkpc_hsa/gibbs/ces) |
| `hsa_steady` | `κ₀ + δ·N̄_t` | — | exact joint FFBS | [`gibbs/hsa_steady/`](src/nkpc_hsa/gibbs/hsa_steady) |
| `hsa_const_theta` | `κ₀ + δ·N̄_t` | `θ` (constant) | exact joint FFBS | [`gibbs/hsa_const_theta/`](src/nkpc_hsa/gibbs/hsa_const_theta) |
| `hsa_dynamic` | `κ` | `θ` (constant) | joint FFBS with shock covariance | [`gibbs/hsa_dynamic/`](src/nkpc_hsa/gibbs/hsa_dynamic) |
| `hsa_full` | `κ₀ + δ·N̄_t` | `θ₀ + γ·N̄_t` | **Particle Gibbs** | [`gibbs/hsa_full_pg/`](src/nkpc_hsa/gibbs/hsa_full_pg) |

`hsa_full` needs Particle Gibbs because the `γ·N̄_t·N̂_t` term makes the
observation equation bilinear in the two states, so no exact linear-Gaussian
FFBS exists. The particle count comes from `configs/models.yaml →
defaults.n_particles` (512) and is recorded in each run's metadata.
`hsa_const_theta` is `hsa_full` with `γ = 0`; it exists because the `γ`
regressor `N̂·N̄` is nearly collinear with the `θ₀` regressor `N̂`.

Shared machinery:

| what | where |
|---|---|
| exact 3-state Kalman/FFBS used by steady and const-θ | [`gibbs/common/joint_ffbs.py`](src/nkpc_hsa/gibbs/common/joint_ffbs.py) |
| conditional marginal likelihood (corrected Chib) | [`gibbs/conditional_ml.py`](src/nkpc_hsa/gibbs/conditional_ml.py) |
| dispatch, data prep, saving | [`inference/wrappers.py`](src/nkpc_hsa/inference/wrappers.py) — `run_model` |
| public facade | [`models/`](src/nkpc_hsa/models) |

`inference/wrappers.py` and `main_scripts/12_build_cpi_ppi_report.py` are the two
files to open first.

### Internal scaling

`κ`, `κ₀` and `δ` are held internally multiplied by `KAPPA_SCALE = 100`, because
the regression column is divided by 100. Priors in `configs/priors_*.yaml` are
in **physical units**; the wrappers convert, and posterior draws are divided
back before being saved. Everything you read — `InferenceData`, tables, figures,
SDDR and Chib inputs — is in physical units. Never apply the factor yourself.

### Equation-to-code map

[`docs/code_equation_crosswalk.md`](docs/code_equation_crosswalk.md) maps every
symbol to the file and line that computes it: regressor columns, state-space
matrices, Kalman and FFBS recursions, the Particle Gibbs sweep, and each
reported macro back to the posterior it came from.
[`docs/estimation_flow.md`](docs/estimation_flow.md) walks one Gibbs iteration
block by block.

---

## Conventions worth knowing before changing anything

- **`results/` is git-ignored** and must never be committed; the whole tree is
  reproducible from `main_scripts/`.
- **`data/raw/` is never written by scripts.**
- Run directories are named `model_spec_prior_frequency` with **no timestamp** —
  one directory per cell, re-estimated in place. `load_report_runs` selects on
  `metadata.json`, not on the name, but the frequency must stay in the name or the
  two observation designs of the same cell collide. The classification
  of every past run; `tests/test_observation_design_default.py` pins the rule.
- Coefficient hard constraints (`κ ≥ 0`, `κ_t ≥ 0`, …) come from
  `configs/models.yaml → defaults.coefficient_constraints` or the `--positive`
  flag, are specified in physical units, and are enforced by rejection sampling.
  Treat constrained runs as **restricted robustness specifications**, not the
  baseline.
- N-state variance priors are in **squared ten-log-point** units. Their scale
  must stay near the 0.01 decade implied by the transformed series; resetting
  them to O(1) silently changes the model.

`CLAUDE.md` holds the full list.

---

## Documentation

| file | what it answers |
|---|---|
| [`docs/estimation_specification.md`](docs/estimation_specification.md) | what is estimated, how, and what is known to be wrong or unverified |
| [`docs/code_equation_crosswalk.md`](docs/code_equation_crosswalk.md) | which line of code implements which symbol |
| [`docs/estimation_flow.md`](docs/estimation_flow.md) | one Gibbs iteration, block by block |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | every column: source, construction, units, provenance |
