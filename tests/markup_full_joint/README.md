# Experiment: Full-joint core-CPI / unemployment E2

Run the full-joint core-CPI / negative-unemployment-gap E2 model with matched CPI expectations, jointly updating the competition state and the Phillips-curve coefficients.

## Bundle layout (the operating convention)

| file | role |
|---|---|
| `functions.py` | estimation functions |
| `run.py` | run code |
| `config.yaml` | this experiment's settings |
| `results/` | results **inside the bundle** — report, `results/tables/`, `results/figures/`, `results/draws/` (raw posterior `.npz`); git-ignored & reproducible |
| `README.md` | this description |

The heavy shared engine (samplers, dataprep, the shared Phillips-curve toolkit
`nkpc_hsa.phillips`) is imported, never copied.

## Run

```bash
python experiments/markup_full_joint/run.py --quick   # smoke
python experiments/markup_full_joint/run.py           # full
```

## Consumed input

Reads an initialization posterior via `config.yaml: measurement.initialization_posterior` (existing artifact under the shared `results/`). Regenerate `markup_measurement` first to relocate it into a bundle.
