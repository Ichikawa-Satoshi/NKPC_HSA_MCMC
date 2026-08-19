# Tests — experiment index (テスト一覧)

Each subdirectory of `tests/` is one **self-contained experiment bundle**
(`functions.py` + `run.py` + `config.yaml` + `results/` + `README.md`). Bundles
import the shared engine from `nkpc_hsa` (sampler, dataprep, the Phillips-curve
toolkit `nkpc_hsa.phillips`) and write outputs only into their own git-ignored
`results/`. Run any one with:

```bash
python tests/<name>/run.py --quick    # smoke
python tests/<name>/run.py            # full
```

`functions.py` present = the experiment has its own estimation code; absent = it
reuses only `nkpc_hsa.phillips`. See each bundle's own `README.md` for detail.

## Standardized result PDF

Any bundle that saved its fits with `_save_fit` (`results/draws/*.npz`) can be
turned into a six-block PDF — the same blocks the production `each_result` report
carries:

```bash
python -m nkpc_hsa.reporting.bundle_report tests/<name>/results --compile
```

→ `tests/<name>/results/report/<name>_report.pdf` with: **(1)** posterior
coefficient table, **(2)** prior-vs-posterior panels for every parameter,
**(3)** the time-varying `κ_t` path, **(4)** the `qbar`/`qhat` competition-state
decomposition, **(5)** a precision / specification comparison, and **(6)** R-hat /
bulk-ESS convergence. Blocks 3–4 appear only when the bundle also saved a
`results/draws/state.npz` (`qbar`/`qhat` draws); `beta_convexity` does, as a
template. Omit `--compile` to write just the `.tex` + tables + figures.

### Describe the model in `spec.yaml` (optional)

Drop a `tests/<name>/spec.yaml` and the report renders it as a **Specification**
header (equation + data table + priors table) — you write the model once, the PDF
is generated. All fields are optional; **text fields are LaTeX** (`$...$` for
maths, escape `_`/`&`/`%`). See `tests/beta_convexity/spec.yaml` for a full
template.

```yaml
description: >
  One-paragraph intro ($\beta_b$, \S4.2, … as LaTeX).
equation: |          # omit to auto-derive from the saved coefficient names
  \pi_t = a + \beta_b\,\pi_{t-1} + \dots + \varepsilon_t
data:                # -> a "Data" table (Series / Source / Transform)
  - series: "$\\pi_t$"
    source: "CPIAUCSL / PPIACO"
    transform: "$400\\,\\Delta\\log P_t$"
priors:              # -> a "Priors" table; unlisted params fall back to the saved SD
  beta_b: "$\\mathcal{N}(0,\\,1)$"
```

Without a `spec.yaml`, the equation is still auto-derived from the coefficient
names and the priors table is filled from the SDs saved in each `.npz`.

## At a glance

| experiment | what it asks / does | own `functions.py` | config | consumes another run's posterior |
|---|---|---|---|---|
| `n_gustavo_state_space` | Estimate the mandatory N_Gustavo-only quarterly competition **state**; produces the posterior others reuse as a modular cut. | shared | code-fixed | — (producer) |
| `markup_measurement` | Q4-anchored inverse-markup **bridge** for the N_Gustavo state (i.i.d. + AR(1) markup error); fits four QoQ E2 cells. | shared | ✔ | — (producer) |
| `observed_hhi` | Does an **observed** sector inverse-HHI proxy reproduce the HSA slope, or is the sign a fast-state timing artifact? | ✔ | ✔ | — |
| `capital_iq_quarterly` | Test the observed quarterly Capital IQ effective-firm series under firm-count and revenue aggregation, with smoke/full convergence gates and a formal PDF. | shared (`observed_hhi`) | ✔ | — |
| `nolag_price_gap` | No-lag inflation equations across prices × activity gaps × model forms, lag replaced by a persistent AR(1) error. | shared | ✔ | ← `n_gustavo_state_space` |
| `markup_full_joint` | Full-**joint** core-CPI / unemployment E2 (state and coefficients updated together) with matched CPI expectations. | ✔ | ✔ | ← `markup_measurement` |
| `markup_feedback` | Sweep the modular-cut → full-joint **feedback** strength λ by importance sampling (λ=0 cut … λ=1 joint). | ✔ | ✔ | ← `markup_measurement` |
| `markup_interpolation` | Zero-sum inverse-markup **timing** sensitivities between exact Q4 anchors (where between anchors the markup info sits). | ✔ | ✔ | — |
| `beta_convexity` | Hybrid **(β_b, β_f) restriction** (convexity / adding-up) vs baseline — does the backward/forward discipline move δ? (review §4.2) | ✔ | ✔ | — |
| `design` | The executable **nine-cell** (3×3 price × activity) identification design and its formal report. | design-specific `reporting.py` / `followup.py` | shared `configs/nine_cell_design.yaml` | — |

## Notes

- **Producers vs consumers.** `n_gustavo_state_space` and `markup_measurement`
  produce competition-state posteriors that `nolag_price_gap`,
  `markup_full_joint` and `markup_feedback` read (via each `config.yaml` posterior
  path). To regenerate a consumer end-to-end, run its producer first.
- **`design` is special.** Unlike the others it builds a formal LaTeX report into
  `report/` and writes to the shared `results/nine_cell_design/`, not its own
  bundle `results/` (its report plumbing uses relative paths). It behaves more like
  the production pipeline than a throwaway test.
- **Shared config.** `configs/nine_cell_design.yaml` is the common design /
  measurement definition that `nkpc_hsa.phillips.load_design_data` reads by
  default; it is not a per-bundle file.
- **Outputs.** Every bundle writes report/tables/figures and the raw MCMC
  `draws/` under its own `results/`, git-ignored (`tests/.gitignore`,
  reproducible). `design` is the noted exception.

## Detailed one-liners

- **`n_gustavo_state_space`** — fits the N_Gustavo-only mixed-frequency state and a
  state-law sensitivity grid; the resulting `posterior/…` state draws are the
  modular-cut input other experiments reuse. (No config — the spec is fixed in code.)
- **`markup_measurement`** — measurement-first and modular: inflation never updates
  the competition state. Fits an i.i.d. markup measurement error and a conservative
  markup-specific AR(1) state, then four QoQ E2 cells (PPI/core-CPI ×
  inverse-markup/negative-unemployment-gap).
- **`observed_hhi`** — builds E0–E2 / `hsa_restricted` designs on an observed
  inverse-HHI series, sweeps fast-state definition/timing, and reports whether the
  HSA sign survives — emphasising that timing is consequential, not a licence to
  pick the lag that yields the preferred sign. No QCEW enters it.
- **`capital_iq_quarterly`** — applies the observed-HHI design to the Capital IQ
  quarterly effective-firm series, compares firm-count and revenue weighting in
  three representative cells under persistent-AR(1) and iid inflation errors,
  and writes draws, diagnostics, figures, tables, a manifest, and a formal PDF
  entirely under its own `results/` directory.
- **`nolag_price_gap`** — reuses the production N_Gustavo-only state as a cut; every
  inflation equation drops lagged inflation and carries persistence in an AR(1)
  disturbance; crosses prices, activity/slack measures, model forms and timings.
- **`markup_full_joint`** — the full-joint core-CPI / negative-unemployment E2 model
  with matched CPI expectations, updating the state and the Phillips-curve
  coefficients jointly (initialised from the markup-measurement posterior).
- **`markup_feedback`** — treats the measurement posterior as a proposal and
  reweights by the inflation likelihood raised to λ, tracing the cut→joint path with
  ESS / Pareto-k diagnostics.
- **`markup_interpolation`** — zero-sum timing sensitivities: hold the exact Q4
  competition anchors fixed and move the quarterly inverse-markup information between
  them, to see how much the slope depends on that placement.
- **`beta_convexity`** — adds `convexity` (β_b,β_f∈[0,1], rejection) and `adding_up`
  (β_b+β_f=1, exact reparameterisation) to the baseline QoQ E2 cut and reports how δ
  and κ_1 move; the shared unconstrained estimator is left unchanged. (Review §4.2.)
- **`design`** — the identification-first nine-cell design: `run.py` estimates and
  builds the report, `finalize.py` runs the follow-up (equivalence / secondary-joint
  / smoke) modules; `--test-run` stamps every output NOT FOR INFERENCE.
