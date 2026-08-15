# Experiment: Cut-to-full-joint inflation feedback path

Evaluate the modular-cut → full-joint inflation-feedback path by importance sampling: how much the inflation likelihood should reweight the competition-state measurement posterior (feedback strength λ from 0 = modular cut to 1 = full joint).

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
python experiments/markup_feedback/run.py --quick   # smoke
python experiments/markup_feedback/run.py           # full
```

## Consumed input

Reads a stored markup-measurement posterior via `config.yaml: measurement.posterior` (an existing artifact under the shared `results/`). To regenerate end-to-end, run `markup_measurement` first and point this path at its bundle `results/`.
