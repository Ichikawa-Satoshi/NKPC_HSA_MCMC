# Experiment: Q4-anchored inverse-markup bridge

Measurement-first, modular Q4-anchored inverse-markup change bridge for the N_Gustavo state: inflation never updates the competition state. Fits both an i.i.d. markup measurement error and a conservative markup-specific AR(1) state, then four QoQ E2 cells (PPI/core-CPI × inverse-markup/negative-unemployment-gap).

## Bundle layout (the operating convention)

| file | role |
|---|---|
| `functions.py` | estimation functions — shared, imported from `nkpc_hsa.phillips` |
| `run.py` | run code |
| `config.yaml` | this experiment's settings |
| `results/` | results **inside the bundle** — report, `results/tables/`, `results/figures/`, `results/draws/` (raw posterior `.npz`); git-ignored & reproducible |
| `README.md` | this description |

The heavy shared engine (samplers, dataprep, the shared Phillips-curve toolkit
`nkpc_hsa.phillips`) is imported, never copied.

## Run

```bash
python experiments/markup_measurement/run.py --quick   # smoke
python experiments/markup_measurement/run.py           # full
```

## Producer

This experiment's measurement posterior is **consumed by** `markup_full_joint` and `markup_feedback`. Its estimation functions live in the shared `nkpc_hsa.phillips.markup_measurement` because those experiments reuse them.
