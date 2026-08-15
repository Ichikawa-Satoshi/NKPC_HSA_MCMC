# Experiment: observed inverse-HHI competition channel

**Question.** Does an *observed* competition proxy (sector inverse-HHI) reproduce
the HSA slope evidence that the latent firm-count design produces — or is the
sign an artifact of a particular fast-state timing?

## Bundle layout (the operating convention)

| file | role |
|---|---|
| `functions.py` | estimation functions (design build, sampler, summaries, recovery sim) |
| `run.py` | run code — fan-out over cells/variants, builds the comparison PDF |
| `config.yaml` | data spec, screening, sampling, and simulation settings |
| `results/` | estimation results **inside the bundle** — report PDF/tex, `results/tables/`, `results/figures/`, `results/draws/` (raw per-task posterior `.npz`), `manifest.json` (git-ignored, reproducible) |
| `README.md` | this description |

The heavy shared engine (samplers, dataprep, the shared Phillips-curve toolkit)
is imported from `nkpc_hsa`, not copied here.

## Run

```bash
python experiments/observed_hhi/run.py --quick        # smoke
python experiments/observed_hhi/run.py --jobs 4       # full
python experiments/observed_hhi/run.py --no-draws     # skip raw-draw .npz
```

Outputs (report PDF, tables, figures, per-task posterior draws under
`results/draws/`, manifest) land in `experiments/observed_hhi/results/` — next to
the code, not under the shared `results/` tree. Raw draws are written per task
by the worker processes and are git-ignored (reproducible, and large).

## Notes / limitations

- No QCEW observation enters this experiment (stated in the report body).
- The state smoother is a modular cut; it is not reweighted by any price series
  used here.
- Findings emphasise **timing sensitivity**: the Cell-1 posterior mean moves
  materially across the 0–4 quarter fast-state profile, which is evidence that
  timing is consequential — not a licence to pick the lag that yields the
  preferred HSA sign.
