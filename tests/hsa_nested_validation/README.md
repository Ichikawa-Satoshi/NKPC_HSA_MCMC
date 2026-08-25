# HSA nested validation test

This folder implements the exact-N AR(2)-cycle nested-validation workflow in
`SPECIFICATION.md`. `mock` checks the data path, state identity, nesting algebra,
sampler plumbing, saved outputs, and report rendering. Its posterior numbers
are not empirical results.

The active ladder estimates PPI and Core CPI separately and uses only the joint
slow/cycle state split. Each price has four negative-unemployment-gap models and
eight inverse-markup models. The total is therefore
`(4 + 8) x 2 prices = 24 fits`. The former B4/B5 models and all state-cut fits
are excluded.

The earlier PPI-only 28-fit full run is preserved under
`results/full_v1_28_fit_legacy/`; it must not be interpreted as a result from
this revision. The previous exact-N AR(1) 24-fit run and report are preserved as
`results/full_v3_ar1_legacy/` and
`output/pdf/hsa_nested_validation_report_v3_ar1_legacy.pdf`. They are not AR(2)
results. The active AR(2) full run is stored under `results/full/`; all 24 fits
are present and its convergence/exact-identity gate passes (maximum R-hat
1.0081, minimum state bulk ESS 707, and identity error 2.22e-16). The active
report is `output/pdf/hsa_nested_validation_report.pdf`.

Run the mock workflow from the repository root:

```bash
PYTHONPATH=src:. python tests/hsa_nested_validation/run.py --mode mock --workers 4
PYTHONPATH=src:. python tests/hsa_nested_validation/validate.py --mode mock
PYTHONPATH=src:. python tests/hsa_nested_validation/build_report.py --mode mock
```

Run the longer pre-flight check when desired:

```bash
PYTHONPATH=src:. python tests/hsa_nested_validation/run.py --mode quick --workers 4
PYTHONPATH=src:. python tests/hsa_nested_validation/validate.py --mode quick
PYTHONPATH=src:. python tests/hsa_nested_validation/build_report.py --mode quick
```

Reproduce the confirmatory run with:

```bash
PYTHONPATH=src:. python tests/hsa_nested_validation/run.py --mode full --workers 4
PYTHONPATH=src:. python tests/hsa_nested_validation/validate.py --mode full
PYTHONPATH=src:. python tests/hsa_nested_validation/build_report.py --mode full
```

Do not interpret a full run unless the manifest passes the full convergence and
exact-identity gates. Formal marginal likelihood and annual-origin forecasting
remain downstream phases and are not silently approximated by this command.
