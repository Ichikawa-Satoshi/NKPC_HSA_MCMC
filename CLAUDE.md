# Repository Operating Notes

## Active Layout

The repository separates **production** (the canonical theory-run estimation) from
**experiments** ("tests"). Shared code lives once in `src/nkpc_hsa/`; each experiment
is a self-contained bundle.

- `src/nkpc_hsa/` is the canonical **shared** package: data loading, wrappers, diagnostics, model comparison, reporting, and the Gibbs backend. Both production and every experiment import from here; nothing is copied.
- `src/nkpc_hsa/gibbs/` is the Gibbs/FFBS sampler engine (moved from `analysis/gibbs/func_gibbs/` in July 2026). `src/nkpc_hsa/models/` is the thin public facade over it; wrappers import through the facade.
- `src/nkpc_hsa/phillips/` is the shared Phillips-curve toolkit (renamed from `nine_cell/` in Aug 2026): `data`, `state`, `inflation`, `estimation`, `joint`, `temporal`, `markup_measurement`. Import path `nkpc_hsa.phillips`. It is shared infrastructure — experiment-specific code does NOT belong here.
- `tests/<name>/` is one **test bundle** per experiment: `functions.py` (its estimation functions, absent when it reuses only `nkpc_hsa.phillips`), `run.py` (`python tests/<name>/run.py`), `config.yaml` (its own settings), `results/` (report, `tables/`, `figures/`, `draws/` — written **inside the bundle**), and `README.md`. Bundles import the shared engine, never copy it. `tests/_bootstrap.py` is the shared path setup.
  - Experiment `results/` (including raw `draws/`) are git-ignored via `tests/.gitignore` (`*/results/`) — reproducible, so never committed, but they live next to the code.
  - Some experiments consume a stored posterior produced by another (e.g. `nolag_price_gap` ← `n_gustavo_state_space`; `markup_full_joint`/`markup_feedback` ← `markup_measurement`) via a `config.yaml` posterior path. To regenerate end-to-end, run the producer first and point the consumer path at the producer bundle's `results/`.
- `main_scripts/` contains the reproducible **production** pipeline entry points (`01–19`, the report builders, `make_*`), runnable from the project root. The former experiment runners `20–28` now live under `tests/<name>/run.py`.
- `configs/` stores **shared** path, model, and prior specifications. `nine_cell_design.yaml` is the shared design/measurement definition that `nkpc_hsa.phillips.load_design_data` reads by default; keep it here, not in a bundle. Per-experiment settings live in `tests/<name>/config.yaml`.
- `data/raw/` contains raw inputs and must not be overwritten by scripts.
- `data/processed/` contains generated model-ready data.
- `results/` holds **every generated output**, git-ignored and reproducible from scripts; it must never be committed. One directory per producer:
  - `results/runs/` — one directory per (model, data spec, prior, observation design). No timestamp in the name: a cell is re-estimated in place, and the estimation time lives in `metadata.json` as `run_id`. The observation design must be part of the name, or the two designs of the same cell collide.
  - `results/theory_runs/2026-08-moving-reference-hsa-v1/` — the separate F0/U/R1/R2/R3 namespace. Never copy a legacy `results/runs` posterior into it. Current-report inputs require signed definition/data provenance, 12,000 iterations, a clean code revision, and passed diagnostics.
  - `results/tables/` and `results/figures/` — the tables and figures the report `\input`s and `\includegraphics`es. Written by `main_scripts/12_build_cpi_ppi_report.py` and the `make_*` scripts; `annual_q4/` subdirectories hold the mixed-frequency design, the top level the interpolated one.
  - `results/diagnostics/`, `results/prior_robustness/`, `results/period_robustness/`, `results/prior_decomposition/` — each owned by the script of the same name. Do not write aggregates into `results/tables/`; that belongs to the report.
  - `results/error_robustness/` — the separate MA(3) disturbance diagnostics and runs driven by `main_scripts/er_01_diagnose.py` / `er_02_estimate.py`; these do not enter the headline report until a full production sweep is complete.
- `report/` has two deliberately separate deliverables. `nkpc_hsa_report.tex/.pdf` is the preserved historical reduced-form/ablation report and is built only by `main_scripts/build_report.py`. `nkpc_hsa_restriction_report.tex/.pdf` is the F0/U/R1/R2/R3 report and is built only by `main_scripts/build_restriction_report.py`. Neither builder may modify or import the other report's artifacts.
- The `report/*.tex` / `report/*.pdf` deliverables are **committed sources**, not build products: no code deletes or rewrites them (`build_report.py` only reads the `.tex` and writes generated `\input`s under the git-ignored `report/generated/`). If one goes missing, restore it with `git checkout -- report/<file>`. The experiment bundles under `tests/<name>/` write only to their own git-ignored `tests/<name>/results/`; the sole exception is `design`, which builds its *own* `nine_cell_design_report.tex` into `report/` and never touches the two report deliverables.
- `main_scripts/build_report.py` remains the single builder for the preserved historical report. Add historical artifacts to its `STEPS`; do not add theory-run artifacts there.
- `main_scripts/10_estimate_theory_models.py`, `11_run_theory_diagnostics.py`, and `19_build_theory_report.py` are the estimate/diagnose/artifact sequence for the new theory namespace. `main_scripts/run_restriction_production.py` runs that sequence and compiles the restriction PDF. `build_restriction_report.py --allow-missing-runs` is preview-only; production builds fail if any required run is missing or stale.
- `docs/` contains the human-readable estimation specification, the code/equation crosswalk, the estimation flow, and the data dictionary. It describes the current production code and must be updated whenever the code it cites moves.
- `references/` contains literature PDFs and research notes (this is what the pre-2026 `docs/` directory held; do not confuse it with the `docs/` above).

## Gibbs Backend

The production wrappers call the Gibbs backend at `src/nkpc_hsa/gibbs/`
(import path `nkpc_hsa.gibbs`). It is the migrated legacy engine and is active
code, not scrap. Pre-move history is preserved under the git tag
`pre-restructure` (old path `analysis/gibbs/func_gibbs/`).

## Conventions

- Kappa priors in config files are physical/economic units. Wrappers handle any internal `KAPPA_SCALE` conversion.
- New restricted-model draws use `kappa_N_empirical`, never legacy `delta`. With the production N transform, `d_kappa_d_logN=10*kappa_N_empirical` and `100*kappa_N_empirical=b_x*zeta0*theta0`; `KAPPA_SCALE` is already accounted for at the sampler/storage boundary.
- Output-gap data specs are configured in `configs/models.yaml`. `output_gap_BN`, `output_gap_HP`, and `labor_share_gap_HP` are all in 100-log-point units; the HP output version is generated from `100 * output`, and the labor-share version is generated from `100 * log(labor_share_index)` in `main_scripts/01_build_data.py` via `src/nkpc_hsa/dataprep/build.py`.
- HSA competition series use `n_transform="log100_centered10"` by default: `(100 * log(N) - sample mean) / 10`. Coefficients on `Nhat` and `Nbar` are therefore estimated per ten-log-point deviation from the sample mean.
- Reported `delta`, `theta`, `theta_0`, and `gamma` are already on the ten-log-point `N` scale. Do not multiply these by 10 again in tables or prior/posterior plots.
- Resolve configured data cells through `configured_data_specs`; it injects the study-wide `sample_start` / `sample_end`. Directly copying `config["data_specs"][name]` bypasses the window and changes the current processed data from T=124 to T=128.
- HSA dynamic shock covariance uses `covariance_structure="e_zeta_only"` by default; this allows only `e_t` and `zeta_t` correlation.
- N-state shock and measurement variance priors (`a_u`/`b_u`, `a_eps`/`b_eps`, `a_N`/`b_N`, and the `u`/`eps` entries of `S_Sigma`) are in squared ten-log-point units; their scales must stay near the 0.01 decade implied by the transformed `N` series. Do not reset them to O(1) values.
- `hsa_full` includes an explicit N measurement error with variance `sigma_N^2` and samples `Nhat | Nbar` and `Nbar | Nhat` as exact conditional FFBS blocks. The legacy `target_scale`/`rw_scale` pseudo variances are gone.
- Chib marginal-likelihood calculations in `src/nkpc_hsa/gibbs/gibbs_marginal_likelihood.py` take a `priors` argument (physical units, `priors_*.yaml` shape); `model_comparison.py` passes each run's saved priors so prior and ordinate terms match the sampling priors.
- Coefficient hard constraints are controlled by `defaults.coefficient_constraints` in `configs/models.yaml` or by the script `--positive` option. Bounds for `kappa`, `kappa_0`, and `delta` are specified in physical units and converted internally before rejection sampling in the coefficient block. Treat constrained runs as restricted robustness specifications.
- `kappa_t` is also a supported hard constraint for HSA steady/full models. It is checked as a whole path, so candidate draws must satisfy the bound for every period of `kappa_t = kappa_0 + delta * Nbar_t`. In time-varying kappa models, a generic positive `kappa` constraint is interpreted as a `kappa_t` path constraint.
- New outputs should go under `results/`, not `references/` or `report/`.
- Do not commit `.DS_Store`, `__pycache__/`, `.pyc`, or LaTeX auxiliary files.
