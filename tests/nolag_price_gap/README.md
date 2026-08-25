# Experiment: No-lag price / activity-gap grid

No-lag inflation equations across prices, activity/slack gaps, and model forms. Lagged inflation is removed and persistence is carried by an AR(1) disturbance. Reuses the production N_Gustavo-only mixed-frequency state as a modular cut (not reweighted by any price series here).

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
python tests/nolag_price_gap/run.py --quick   # smoke
python tests/nolag_price_gap/run.py           # full
```

## Consumed input

Reads the N_Gustavo-only state posterior via `config.yaml: state_posterior` (existing artifact under the shared `results/`; produced by `n_gustavo_state_space`).
