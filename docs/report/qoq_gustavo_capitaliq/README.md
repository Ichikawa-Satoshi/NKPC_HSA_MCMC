# QoQ Gustavo x Capital IQ HSA report

This directory contains the comprehensive English report for the saved PPI,
Core-CPI, and prespecified oil-control QoQ estimation sequence in
`tests/gustavo_state_capitaliq_cycle`. The report follows
the landscape LaTeX format used elsewhere under `docs/report`.

It is a reporting layer over saved estimation results. Rebuilding figures and
tables does not rerun MCMC.

## Build

From the repository root:

```bash
python docs/report/qoq_gustavo_capitaliq/build_assets.py
cd docs/report/qoq_gustavo_capitaliq
latexmk -pdf -interaction=nonstopmode -halt-on-error qoq_hsa_report.tex
```

The source report is `qoq_hsa_report.tex`; the compiled document is
`qoq_hsa_report.pdf`. A delivery copy is stored at
`output/pdf/qoq_gustavo_capitaliq_hsa_report.pdf`.

## Inputs

- `tests/gustavo_state_capitaliq_cycle/results/staged_validation/`
- `tests/gustavo_state_capitaliq_cycle/results/dynamic_validation/`
- `tests/gustavo_state_capitaliq_cycle/results/core_cpi_full/`
- `tests/gustavo_state_capitaliq_cycle/results/oil_control_full/`
- the QoQ data and competition inputs loaded by the corresponding test code

The asset builder writes reproducible plots to `figures/` and LaTeX fragments to
`tables/`.
