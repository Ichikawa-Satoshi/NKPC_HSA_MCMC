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
- HSA competition series use `100 * log(N)` by default unless metadata explicitly records `n_transform="identity"`.
- HSA dynamic shock covariance uses `covariance_structure="e_zeta_only"` by default; this allows only `e_t` and `zeta_t` correlation.
- Coefficient hard constraints are controlled by `defaults.coefficient_constraints` in `configs/models.yaml` or by the script `--positive` option. Bounds for `kappa`, `kappa_0`, and `delta` are specified in physical units and converted internally before rejection sampling in the coefficient block. Treat constrained runs as restricted robustness specifications.
- New outputs should go under `results/`, not `docs/`, `reports/`, or `archive/`.
- Do not commit `.DS_Store`, `__pycache__/`, `.pyc`, or LaTeX auxiliary files.
