# Experiment: Nine-cell identification design

The executable nine-cell (3×3 price × activity) identification-first design and its report. `run.py` estimates and builds the report; `finalize.py` runs the follow-up (equivalence/secondary-joint/smoke) modules. `--test-run` stamps every output NOT FOR INFERENCE.

## Bundle layout (the operating convention)

| file | role |
|---|---|
| `reporting.py`, `followup.py` | design-specific functions (report artifacts, follow-up modules); estimator shared in `nkpc_hsa.phillips` |
| `run.py` | run code |
| `config.yaml` | *(none — uses shared `configs/nine_cell_design.yaml`)* |
| `README.md` | this description |

The heavy shared engine (samplers, dataprep, the shared Phillips-curve toolkit
`nkpc_hsa.phillips`) is imported, never copied.

## Output location (design is special)

Unlike the other bundles, `design` produces a **formal compiled report**. Its
LaTeX plumbing writes into `report/` and references `results/nine_cell_design/`
by relative path, so its outputs stay at the shared `results/nine_cell_design/`
location (via `run_nine_cell_design`'s default), **not** inside the bundle. Pass
`--output-dir` to override. In this respect it behaves more like the production
pipeline than a throwaway test.

## Run

```bash
python experiments/design/run.py --quick   # smoke
python experiments/design/run.py           # full
```

## Extra entry point

```bash
python experiments/design/finalize.py --config ...   # follow-up modules
```

`reporting.py` and `followup.py` are this experiment's design-specific functions; the shared estimator is `nkpc_hsa.phillips.estimation.run_nine_cell_design`.
## Shared config

Uses the shared design definition `configs/nine_cell_design.yaml` (also the default that `nkpc_hsa.phillips.data.load_design_data` reads), not a bundle-local config.
