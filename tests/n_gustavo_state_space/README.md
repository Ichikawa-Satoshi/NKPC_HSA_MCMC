# Experiment: N_Gustavo-only quarterly state space

The mandatory N_Gustavo-only quarterly state-space specification and its state-law sensitivity grid. Produces the quarterly competition-state posterior reused as a modular cut by other experiments.

## Bundle layout (the operating convention)

| file | role |
|---|---|
| `functions.py` | estimation functions — shared, imported from `nkpc_hsa.phillips` |
| `run.py` | run code |
| `config.yaml` | this experiment's settings *(none — fixed in code)* |
| `results/` | results **inside the bundle** — report, `results/tables/`, `results/figures/`, `results/draws/` (raw posterior `.npz`); git-ignored & reproducible |
| `README.md` | this description |

The heavy shared engine (samplers, dataprep, the shared Phillips-curve toolkit
`nkpc_hsa.phillips`) is imported, never copied.

## Run

```bash
python tests/n_gustavo_state_space/run.py --quick   # smoke
python tests/n_gustavo_state_space/run.py           # full
```

## Producer

Writes `results/posterior/...` state draws that `nolag_price_gap` (and the production pipeline) consume. This bundle has no `config.yaml`; its specification is fixed in code.
