# Capital IQ observed quarterly N test

This bundle tests the two quarterly effective-firm series constructed from the
Capital IQ company-revenue panel:

- `N_capitaliq_firmw`: market HHIs aggregated with firm-count weights;
- `N_capitaliq_revw`: market HHIs aggregated with revenue weights.

The design follows `tests/observed_hhi`: the observed competition coordinate is
`q_t = 10 log(N_t)`, centered within each estimation sample, and its short-run
movement is the current one-sided EWMA innovation with an eight-quarter
half-life. No annual interpolation or QCEW measurement block is used.

Three representative inflation/activity cells are estimated under persistent
AR(1) and iid inflation errors. The persistent-AR(1) version is the primary
specification. All code, draws, tables, figures, manifests, and reports remain
inside this bundle.

## Run

From the repository root:

```bash
python tests/capital_iq_quarterly/run.py --quick   # smoke, results/smoke/
python tests/capital_iq_quarterly/run.py           # full, results/
```

Use `--jobs N` to change parallel workers. Both commands fail with a non-zero
exit status if their convergence/validity gates fail. The full report is written
to `tests/capital_iq_quarterly/results/capital_iq_quarterly_test_report.pdf`.

## Outputs

- `results/capital_iq_quarterly_test_report.pdf`: formal result document;
- `results/tables/`: coefficient, specification, and input-audit CSVs;
- `results/figures/`: competition paths and posterior/diagnostic plots;
- `results/draws/`: compressed per-specification posterior draws;
- `results/manifest.json`: configuration, gates, elapsed time, and pass/fail.
