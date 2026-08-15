# Experiment: Inverse-markup timing sensitivity

Zero-sum inverse-markup timing sensitivities between exact Q4 competition anchors — how the placement of quarterly markup information between annual anchors moves the slope, holding the anchors fixed.

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
python experiments/markup_interpolation/run.py --quick   # smoke
python experiments/markup_interpolation/run.py           # full
```
