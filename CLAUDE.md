# Repository Operating Notes

## Active Layout

- `src/nkpc_hsa/` is the canonical package: data loading, wrappers, diagnostics, model comparison, reporting, and the Gibbs backend.
- `src/nkpc_hsa/gibbs/` is the Gibbs/FFBS sampler engine (moved from `analysis/gibbs/func_gibbs/` in July 2026). `src/nkpc_hsa/models/` is the thin public facade over it; wrappers import through the facade.
- `scripts/` contains the reproducible pipeline entry points and should be runnable from the project root.
- `configs/` stores path, model, and prior specifications.
- `data/raw/` contains raw inputs and must not be overwritten by scripts.
- `data/processed/` contains generated model-ready data.
- `results/` contains generated run outputs, diagnostics, figures, tables, and report PDFs. The whole tree is git-ignored: it is reproducible from scripts and must never be committed.
- `paper/` contains LaTeX source files only (formerly `reports/`).
- `references/` contains literature PDFs and research notes (formerly `docs/`).
- `archive/` is historical code and outputs, git-ignored. Active code must not import from it.

## Gibbs Backend

The production wrappers call the Gibbs backend at `src/nkpc_hsa/gibbs/`
(import path `nkpc_hsa.gibbs`). It is the migrated legacy engine and is active
code, not scrap. Pre-move history is preserved under the git tag
`pre-restructure` (old path `analysis/gibbs/func_gibbs/`).

## Conventions

- Kappa priors in config files are physical/economic units. Wrappers handle any internal `KAPPA_SCALE` conversion.
- Output-gap data specs are configured in `configs/models.yaml`. `output_gap_BN`, `output_gap_HP`, and `labor_share_gap_HP` are all in 100-log-point units; the HP output version is generated from `100 * output`, and the labor-share version is generated from `100 * log(labor_share_index)` in `scripts/01_build_data.py` via `src/nkpc_hsa/data/build.py`.
- HSA competition series use `n_transform="log100_centered10"` by default: `(100 * log(N) - sample mean) / 10`. Coefficients on `Nhat` and `Nbar` are therefore estimated per ten-log-point deviation from the sample mean.
- Reported `delta`, `theta`, `theta_0`, and `gamma` are already on the ten-log-point `N` scale. Do not multiply these by 10 again in tables or prior/posterior plots.
- HSA dynamic shock covariance uses `covariance_structure="e_zeta_only"` by default; this allows only `e_t` and `zeta_t` correlation.
- N-state shock and measurement variance priors (`a_u`/`b_u`, `a_eps`/`b_eps`, `a_N`/`b_N`, and the `u`/`eps` entries of `S_Sigma`) are in squared ten-log-point units; their scales must stay near the 0.01 decade implied by the transformed `N` series. Do not reset them to O(1) values.
- `hsa_full` includes an explicit N measurement error with variance `sigma_N^2` and samples `Nhat | Nbar` and `Nbar | Nhat` as exact conditional FFBS blocks. The legacy `target_scale`/`rw_scale` pseudo variances are gone.
- Chib marginal-likelihood calculations in `src/nkpc_hsa/gibbs/gibbs_marginal_likelihood.py` take a `priors` argument (physical units, `priors_*.yaml` shape); `model_comparison.py` passes each run's saved priors so prior and ordinate terms match the sampling priors.
- Coefficient hard constraints are controlled by `defaults.coefficient_constraints` in `configs/models.yaml` or by the script `--positive` option. Bounds for `kappa`, `kappa_0`, and `delta` are specified in physical units and converted internally before rejection sampling in the coefficient block. Treat constrained runs as restricted robustness specifications.
- `kappa_t` is also a supported hard constraint for HSA steady/full models. It is checked as a whole path, so candidate draws must satisfy the bound for every period of `kappa_t = kappa_0 + delta * Nbar_t`. In time-varying kappa models, a generic positive `kappa` constraint is interpreted as a `kappa_t` path constraint.
- New outputs should go under `results/`, not `references/`, `paper/`, or `archive/`.
- Do not commit `.DS_Store`, `__pycache__/`, `.pyc`, or LaTeX auxiliary files.
