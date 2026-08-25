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

Unlike the other bundles, `design` produces a **formal compiled report**. The
currently audited output is under `tests/design/results/`. Pass this path
explicitly when reproducing it so the shared engine's default output location
cannot change where the bundle is written.

## Run

```bash
python tests/design/run.py --test-run --output-dir tests/design/results --compile
python tests/design/run.py --output-dir tests/design/results --compile
```

## Extra entry point

```bash
python tests/design/finalize.py \
  --config configs/nine_cell_design.yaml \
  --baseline-dir tests/design/results \
  --output-dir tests/design/results \
  --test-run
```

`reporting.py` and `followup.py` are this experiment's design-specific functions; the shared estimator is `nkpc_hsa.phillips.estimation.run_nine_cell_design`.
## Shared config

Uses the shared design definition `configs/nine_cell_design.yaml` (also the default that `nkpc_hsa.phillips.data.load_design_data` reads), not a bundle-local config.
