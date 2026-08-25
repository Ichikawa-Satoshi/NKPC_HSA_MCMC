# Tests — experiment index (テスト一覧)

Each subdirectory of `tests/` is one experiment bundle. Most have a `README.md`,
configuration, executable scripts, and saved output under their own `results/`,
but script names and sampling profiles differ. Use the exact commands in the
bundle README rather than assuming every folder has the same `run.py` interface.

Two repository-wide records are mandatory reading:

- [`EXPERIMENT_AUDIT.md`](EXPERIMENT_AUDIT.md) records what every existing test
  actually did and what its saved results do and do not establish.
- [`TEST_BUNDLE_TEMPLATE.md`](TEST_BUNDLE_TEMPLATE.md) is the documentation
  contract for every new test folder.

For bundles that retain the common interface, run from the repository root with:

```bash
python tests/<name>/run.py --quick    # smoke
python tests/<name>/run.py            # full
```

`functions.py` present usually means the experiment has its own estimation code;
absence usually means it reuses shared code. See each bundle README for the actual
model, sampling length, and inferential status.

## Documentation contract

Every new test folder must contain, from the time it is created:

1. the research question and frozen model equations;
2. data, sample, transformations, timing, and what is held fixed;
3. priors, estimands, theory signs, and predeclared pass/fail gates;
4. mock, quick, and full sampling definitions plus exact runnable commands;
5. a results section updated after each run with numerical diagnostics;
6. an explicit conclusion stating whether the output is inferential, diagnostic,
   failed, or superseded; and
7. an output inventory, limitations, and changelog.

Mock and quick output must be labelled **not for inference**. A converged sampler
must not be described as an identified model unless the structural learning gates
also pass. Failed and legacy runs remain recorded rather than being overwritten.

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
| `hsa_ppi_identification` | Separates the fixed-preprocessing PPI signal from fixed/joint-state and five-model latent comparisons. | ✔ | ✔ | — |
| `hsa_exact_n_decomposition` | Exact `N=Nbar+Nhat`, annual Gustavo constraint, variance-share parameterization, and seven nested NKPC models. | ✔ | ✔ | — |
| `hsa_lambda_dynamic` | Joint-state static/dynamic HSA models with estimated proportionality coefficient `lambda`. | ✔ | ✔ | — |
| `hsa_theta_bridge` | Controlled cut/joint × fixed/free-lambda test of why theta appears identified under a restriction. | ✔ | ✔ | — |
| `hsa_nested_validation` | Active 24-fit PPI/core-CPI nested-validation grid using an exact-N AR(2) state. | ✔ | ✔ | — |
| `hsa_deep_identification` | Frozen identification-first screen, recovery, exact-N MA(3), QoQ, and restriction audit. | `joint_ma3.py` | ✔ | ← active exact-N inputs |
| `competition_slope_change` | Competition-only AR(2) slow/cycle block propagated into a semi-structural slope-only MA(3) NKPC, with historical slope-change and timing diagnostics. | ✔ | ✔ | — |
| `active_firm_stock_bds_bed` | External active-firm state: annual BDS firm levels plus BED establishment-flow timing, cut from inflation, followed by oracle and propagated-state free-`theta_N` recovery. | ✔ | ✔ | — |
| `gustavo_state_capitaliq_cycle` | Exact annual-Q4 Gustavo slow-state bridge, cut Capital IQ AR(2) cycle, QoQ direct/combined recovery, then an explicitly authorized post-gate varying-theta/free-dynamic/HSA-dynamic diagnostic; YoY archived. | ✔ | ✔ | — |
| `mixed_frequency_gustavo_capitaliq` | Exact total Gustavo Q4 conditions plus two Capital IQ QoQ-growth measurements in one inflation-cut mixed-frequency state; blocked measurement validation and four free-channel PPI/markup NKPC cells. | ✔ | ✔ | — |

## Notes

- **Producers vs consumers.** `n_gustavo_state_space` and `markup_measurement`
  produce competition-state posteriors that `nolag_price_gap`,
  `markup_full_joint` and `markup_feedback` read (via each `config.yaml` posterior
  path). To regenerate a consumer end-to-end, run its producer first.
- **`design` is special.** It builds a formal LaTeX report as part of its run and
  currently has saved audit output under `tests/design/results/`. Check its
  manifest before treating any other configured output path as current.
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
- **`hsa_ppi_identification`** — contains three distinct exercises: a converged
  fixed-preprocessing PPI screen, a fixed-versus-joint state comparison, and a
  five-model latent comparison. They must not be collapsed into one claim.
- **`hsa_exact_n_decomposition`** — imposes exact quarterly accounting and the
  annual Gustavo restriction; it establishes computational feasibility but not
  structural identification.
- **`hsa_lambda_dynamic`** — tests estimated lambda in static and dynamic HSA
  restrictions; its saved full run fails convergence and identification.
- **`hsa_theta_bridge`** — shows that the main identification loss occurs when the
  product `lambda*theta` is factorized, not mainly from cut versus joint feedback.
- **`hsa_nested_validation`** — the current full 24-fit nested grid; legacy 28-fit
  and AR(1) directories are retained but not part of the active result.
- **`hsa_deep_identification`** — the latest frozen audit. It found no candidate
  that passed all convergence, identification, theory-sign, and validation gates;
  consequently it did not promote a full run or compute formal marginal evidence.
- **`competition_slope_change`** — the implemented successor baseline: it treats
  effective-firm concentration as an empirical competition coordinate, estimates
  the exact competition decomposition without inflation feedback, propagates state
  draws into four slope-only MA(3) cells, and reports historical `kappa` changes.
  The full sampler passes strict convergence gates, but all `delta` intervals cross
  zero, `theta_C` remains prior-wide, and the omega allocation is prior-sensitive.
- **`active_firm_stock_bds_bed`** — builds the theory-aligned external firm-stock
  coordinate from annual BDS firm counts and quarterly BED timing without letting
  inflation update the state. Its smoke state misses the ESS gate, and recovery
  after state uncertainty is propagated detects none of the injected effects
  through `theta_N=50`; no full run or HSA restriction is therefore promoted.
- **`gustavo_state_capitaliq_cycle`** — assigns non-overlapping measurement roles:
  Gustavo alone fixes the quarterly slow-state bridge and Capital IQ alone
  supplies a conditional AR(2) cycle. The primary NKPC now uses annualized QoQ
  PPI and genuine one-quarter-ahead SPF under IID and AR(1) errors; its saved YoY
  predecessor remains archived. QoQ strengthens the positive direction but all
  direct-channel intervals include zero and the recovery ESS gate fails. Its
  separately recorded free-combined extension passes convergence and shows that
  adding `delta * nbar * x` does not remove the positive `theta_CIQ` update;
  `delta` itself remains sign-uncertain. Its 640-fit staged recovery fails the
  observed-effect promotion gate. A later explicitly authorized diagnostic runs
  varying theta, free dynamic, and HSA-restricted dynamic anyway: computation
  passes, but gamma, lambda, and both derived HSA slope coefficients remain
  sign-unidentified, and all dynamic models lose holdout ELPD to constant theta.
- **`mixed_frequency_gustavo_capitaliq`** — lets Gustavo constrain total
  `nbar+nhat` at Q4 while Capital IQ growth informs the within-year total path;
  the average quarterly allocation is a transition mean, not an observation.
  Its retained mock fails both state convergence and the prespecified blocked
  Capital IQ forecast: mean RMSE is 7.1% worse than average allocation. All four
  free `theta_N` means have the wrong HSA orientation. It is not promoted to
  quick and no HSA restriction is estimated.
