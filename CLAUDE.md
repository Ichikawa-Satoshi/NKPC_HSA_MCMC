# Repository Operating Notes

## Active Layout

- `src/nkpc_hsa/` is the canonical package for data loading, wrappers, diagnostics, model comparison, and reporting.
- `scripts/` contains the reproducible pipeline entry points and should be runnable from the project root.
- `configs/` stores path, model, and prior specifications.
- `data/raw/` contains raw inputs and must not be overwritten by scripts.
- `data/processed/` contains generated model-ready data.
- `results/` contains generated run outputs, diagnostics, figures, tables, and report PDFs.
- `results/prior_robustness/` and `results/period_robustness/` contain robustness workflow outputs.
- `reports/` contains LaTeX source files only.
- `notebooks/legacy/` is for exploratory or historical notebooks only.
- `archive/` is for historical scripts and old generated outputs. Active code should not import from `archive/`.

## Compatibility Backend

The current production wrappers still call the migrated legacy Gibbs backend in
`analysis/gibbs/func_gibbs/`. This directory is active code, not scrap. Do not
move it into `archive/` or `_scrap` unless the imports are replaced by native
`src/nkpc_hsa/models/` implementations.

## Conventions

- Kappa priors in config files are physical/economic units. Wrappers handle any internal `KAPPA_SCALE` conversion.
- Output-gap data specs are configured in `configs/models.yaml`. `output_gap_BN`, `output_gap_HP`, and `labor_share_gap_HP` are all in 100-log-point units; the HP output version is generated from `100 * output`, and the labor-share version is generated from `100 * log(labor_share_index)` in `scripts/01_build_data.py` via `src/nkpc_hsa/data/build.py`.
- HSA competition series use `n_transform="log100_centered10"` by default: `(100 * log(N) - sample mean) / 10`. Coefficients on `Nhat` and `Nbar` are therefore estimated per ten-log-point deviation from the sample mean.
- Reported `delta`, `theta`, `theta_0`, and `gamma` are already on the ten-log-point `N` scale. Do not multiply these by 10 again in tables or prior/posterior plots.
- HSA dynamic shock covariance uses `covariance_structure="e_zeta_only"` by default; this allows only `e_t` and `zeta_t` correlation.
- Coefficient hard constraints are controlled by `defaults.coefficient_constraints` in `configs/models.yaml` or by the script `--positive` option. Bounds for `kappa`, `kappa_0`, and `delta` are specified in physical units and converted internally before rejection sampling in the coefficient block. Treat constrained runs as restricted robustness specifications.
- `kappa_t` is also a supported hard constraint for HSA steady/full models. It is checked as a whole path, so candidate draws must satisfy the bound for every period of `kappa_t = kappa_0 + delta * Nbar_t`. In time-varying kappa models, a generic positive `kappa` constraint is interpreted as a `kappa_t` path constraint.
- New outputs should go under `results/`, not `docs/`, `reports/`, or `archive/`.
- Do not commit `.DS_Store`, `__pycache__/`, `.pyc`, or LaTeX auxiliary files.
