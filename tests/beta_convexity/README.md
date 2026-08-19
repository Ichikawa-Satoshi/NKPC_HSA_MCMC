# Experiment: hybrid (β_b, β_f) adding-up / convexity restriction

**Question (review §4.2).** The shared inflation regression puts independent,
zero-centred priors on the lagged-inflation weight `β_b` and the expectation
weight `β_f` — it imposes no hybrid-NKPC discipline. Does forcing the standard
restriction change the backward/forward split or, more importantly, the HSA slope
`δ`? This experiment adds the restriction **without touching the unconstrained
default estimator**.

Three specifications per cell:

| spec | restriction | how |
|---|---|---|
| `baseline` | none | the unconstrained conditional draw |
| `convexity` | `β_b, β_f ∈ [0,1]` | rejection from the exact Gaussian conditional |
| `adding_up` | `β_b + β_f = 1`, `β_f ∈ [0,1]` | reparameterise `β_b = 1 − β_f` (exact) |

## Bundle layout

| file | role |
|---|---|
| `functions.py` | constrained coefficient draws + `fit_hybrid_restricted` + `delta_summary` |
| `run.py` | run code: `python tests/beta_convexity/run.py` |
| `config.yaml` | sample window, model (E2), cells, sampling |
| `results/` | `restriction_comparison.csv`, `figures/delta_by_restriction.png`, `draws/`, `manifest.json` (git-ignored) |

The shared design, priors, transforms and the `CellFit` container come from
`nkpc_hsa.phillips`; only the coefficient-draw step is replaced here.

## Run

```bash
python tests/beta_convexity/run.py --quick    # smoke
python tests/beta_convexity/run.py            # full
python tests/beta_convexity/run.py --no-draws
```

## Reading the output

`restriction_comparison.csv` gives, per (cell × spec): `beta_b_mean`,
`beta_f_mean`, `beta_sum_mean`, `kappa_1_mean`, `delta_mean` with a 95% CI, and
`restriction_binding_share` (fraction of draws where the box rejected and the mean
was clipped — a high share means the data fight the restriction). If `δ` is stable
across the three specs, the unconstrained default is not driving the HSA result; if
it moves, §4.2 is material and should be reported as a robustness caveat.

## Notes / limitations

- Only the hybrid QoQ E1/E2 equation is covered (the restriction is about the
  backward/forward split). YoY and no-lag equations are out of scope.
- `adding_up` also keeps `β_f ∈ [0,1]`, so `β_b ∈ [0,1]` automatically.
- Uses the i.i.d. markup-measurement state (modular cut); inflation never updates
  the competition state.
