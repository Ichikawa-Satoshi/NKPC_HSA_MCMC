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
python scripts/01_build_data.py
python -m pytest -q
```

---

## Repository structure

```
configs/          estimation settings — the knobs live here, not in code
  models.yaml       defaults (12000 iters, 2 chains, annual_q4, 512 particles),
                    14 data specs, 5 models
  priors_*.yaml     baseline / tight / weak prior sets
  periods.yaml      sub-sample windows for period robustness

data/
  raw/              source series; never written by scripts
  processed/        model_ready.csv — built by scripts/01

src/nkpc_hsa/     the package
  dataprep/         raw series -> model_ready.csv
  gibbs/            the sampler engine (one subpackage per model)
  models/           thin public facade over gibbs/
  inference/        run_model, diagnostics, robustness, identification
  reporting/        tables, figures and LaTeX fragments

scripts/          pipeline entry points, all runnable from the project root

results/          every generated output; git-ignored, reproducible
  runs/             posterior draws, one directory per estimation cell
  tables/           the LaTeX fragments the report \inputs (annual_q4/ = main design)
  figures/          the PNGs the report \includegraphics
  diagnostics/      per-run trace / autocorr plots, R-hat, ESS
  evidence/         build intermediates (predictive comparison scores)
  prior_decomposition/   the delta-vs-AR(2) prior factorial

report/           the deliverable only
  nkpc_hsa_report.tex   the only report source; reads ../results/{tables,figures}
  nkpc_hsa_report.pdf   compiled output

docs/             estimation specification, code/equation crosswalk,
                  estimation flow, data dictionary
references/       literature PDFs and research notes
```

Everything the PDF needs is generated under `results/`, so `report/` holds only
the source and the output. Compile from inside `report/` — the `.tex` refers to
`../results/{tables,figures}/` relatively.

---

## Running the estimation

### Reproducing the report

The headline result is **HSA steady, core CPI, negative unemployment gap,
mixed-frequency**. End to end:

```bash
python scripts/01_build_data.py
python scripts/13_estimate_cpi_ppi_report.py --jobs 6 --competition-frequency quarterly_interpolated --no-build
python scripts/13_estimate_cpi_ppi_report.py --jobs 6 --competition-frequency annual_q4 --compile
```

Step 2 estimates the 77 interpolated cells, step 3 the 61 mixed-frequency ones,
and step 3 then builds every report artifact and compiles the PDF. Both
estimation steps are expensive — 12000 iterations × 2 chains per cell, about
90 minutes each at `--jobs 6`. Use `--quick` for a smoke test. Existing runs of
the current revision are reused; `--force` re-estimates them.

`--no-build` on the first step is what keeps it from building a report while
half the run set is still missing.

### Rebuilding the report without re-estimating

```bash
python scripts/build_report.py --compile
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

`scripts/prior_decomposition_rho_delta.py` is deliberately **not** part of the
build: it estimates 12 diagnostic cells of its own (~45 minutes) and writes
`prior_decomposition_macros.tex`. Run it directly when the prior sweep changes.

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

### Supporting analyses (not part of the report build)

```bash
python scripts/03_run_diagnostics.py             # trace / autocorr plots, R-hat, ESS per run
python scripts/09_identification_diagnostics.py
python scripts/chib_marginal_likelihood.py       # conditional marginal likelihood
python scripts/appendix_particle_gibbs_hsa_full.py validate|pilot|produce
```

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
superseded `ESTIMATION_REVISION` are deleted rather than accumulated; the
revision string in each run's `metadata.json` records which vintage of the
inputs produced it, and both the estimator and the report builder select on it.

---

## Data

`scripts/01_build_data.py` turns `data/raw/` into
`data/processed/model_ready.csv`. Each estimation cell selects six columns and
drops missing values **jointly**, so every specification has **T = 124**
quarters, 1982Q1–2012Q4.

| role | series | source |
|---|---|---|
| `π_t` | headline CPI (`pi_cpi`), core CPI (`pi_cpi_core`), PPI (`pi_ppi`) | `CPIAUCSL`, `CPILFESL`, `PPIACO` |
| `E_tπ_{t+1}` | `Epi` | Cleveland Fed inflation expectations |
| `x_t` | negative unemployment gap (`unemp_gap`), BN output gap, HP output gap, HP labor-share gap, inverse markup | CBO `NROU` − `UNRATE` (SA); BN filter output; HP filter (λ=1600) computed in `dataprep/build.py` |
| `N_t` | `N_Gustavo` (inverse HHI of US listed firms), `N_TNIC` | `BN_N_Gustavo_26.csv`, `BN_N_TNIC_26.csv` |

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

`inference/wrappers.py` and `scripts/12_build_cpi_ppi_report.py` are the two
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
  reproducible from `scripts/`.
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
