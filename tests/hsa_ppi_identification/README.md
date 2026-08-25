# HSA PPI Identification

Diagnostic study of the HSA competition channels in the **theory-near cell**:
**Capital IQ firm-weighted competition × PPI inflation × inverse-markup activity ×
SPF expectations**.

This folder contains three distinct exercises. The fixed-preprocessing run has a
converged PPI slope signal, but the free direct channel is not identified and the
joint latent-state model comparison does not establish HSA over CES. See
`../EXPERIMENT_AUDIT.md` before citing a headline coefficient.

## Estimating equation (theory-faithful, psi excluded)

```
pi_t = a + beta_b pi_{t-1} + beta_f E_t pi_{t+1}
       + (kappa_0 + kappa_1 q_t) x_t     # competition-dependent slope; kappa_1 = delta
       - theta_0 c_t                     # direct cyclical competition channel
       (+ gamma * -q_t c_t)              # varying_theta only
       + eps_t,     eps_t ~ AR(1)
```

- `q_t = 10 log(N_t)` centered; slow competition environment.
- `c_t` = cyclical competition = one-sided EWMA innovation (8q half-life), **lag1**.
- `x_t` = inverse markup (marginal-cost proxy, BN cycle).
- **No standalone competition-level term `psi q_t`** — it is an empirical control,
  not part of the structural HSA NKPC (drops out of the theory; excluded here).
- **HSA restriction** (`hsa_restricted`): single `theta` on `b_x*zeta*q_t x_t - c_t`,
  imposing `kappa_1 = b_x*zeta*theta` (from the HSA identity dkappa/dq = zeta*theta).

## Data

| symbol | series | source | construction |
|---|---|---|---|
| pi_t | `pi_ppi` | FRED PPIACO | YoY %, monthly→quarterly |
| x_t | `markup_BN_inv` | De-Loecker-type inverse markup | Beveridge–Nelson cycle |
| E pi | `Epi_spf_gdp` | Philadelphia Fed SPF | 1q-ahead GDP-deflator, annualized-log |
| N_t | `N_capitaliq_firmw` | Capital IQ company panel | firm-weighted 1/HHI over coarse SIC, industry-SA |

Samples: `full` 1989Q4–2017Q4, `primary` 1996Q1–2017Q4 (drops the 1989–1995
Capital IQ database coverage ramp), `conservative` 2000Q1–2017Q4. End = markup 2017Q4.

## Estimation

Bayesian Gibbs: AR(1)-whitened conjugate normal regression for coefficients,
inverse-gamma for the disturbance variance, Metropolis–Hastings for the AR(1)
coefficient. 12,000 iterations, 4,000 warmup, thin 4, 4 chains.

## Run

```bash
python tests/hsa_ppi_identification/run.py --quick   # smoke -> results/smoke/
python tests/hsa_ppi_identification/run.py            # full  -> results/
python tests/hsa_ppi_identification/build_report.py   # inverse-markup PDF (default primary_activity)
python tests/hsa_ppi_identification/build_report.py --activity neg_unemp_gap   # unemployment-gap PDF
python tests/hsa_ppi_identification/build_report_gustavo_capiq.py             # Gustavo x Capital IQ PDF
```

### Two estimators on the indicator-allocated N (fixed vs joint decomposition)

```bash
python tests/hsa_ppi_identification/two_models.py                 # 4 cells x {fixed, joint}
python tests/hsa_ppi_identification/build_report_two_models.py    # PDF with model write-up
```

`two_models.py` builds the quarterly competition N by allocating each annual Gustavo
change with Capital IQ weights — year-specific `ŵ^CIQ_q` where Capital IQ is
observed, average `w̄_q` where missing (`gustavo_capiq_quarterly_v2`) — then estimates
the HSA NKPC two ways: Model 1 decomposes N into N̄/N̂ with a fixed EWMA before
estimation; Model 2 estimates N̄ (trend) and N̂ (cycle) jointly with the NKPC in a
state-space sampler. Cells: {inverse markup, neg-unemployment gap} × {PPI, core CPI},
SPF GDP-deflator expectations. The report states the model and the HSA-restriction
justification (dκ/dN̄ = ζθ ⇒ δ = b_x·ζ·θ).

### Nested-model comparison (CES / Slope / Direct / Dynamic / Joint)

```bash
python tests/hsa_ppi_identification/model_comparison.py --workers 4     # 4 cells x Models 0-4, WAIC + logML
python tests/hsa_ppi_identification/build_report_model_comparison.py    # comparison PDF
```

`model_comparison.py` estimates the five nested HSA models — 0 CES (no channel), 1 Slope
(δ), 2 Direct (θ₀), 3 Dynamic (θ₀,γ), 4 Joint (δ,θ₀,γ) — jointly with the latent N̄/N̂
state-space (report_models `run_gibbs`, hybrid) on the v2 Gustavo×Capital IQ N, for the four
cells {PPI, core CPI} × {inverse markup, neg-unemp gap}, and compares them by WAIC and
Laplace–Metropolis log marginal likelihood. Cells run in parallel (one process per cell).
**Finding:** in every cleanly-mixed cell no free competition channel beats the CES baseline
(ΔWAIC, ΔlogML within noise); the one large apparent gain (core CPI / neg-unemp gap, Slope
& Joint) is a non-converged mode (Rhat 1.3–1.6, δ≈0) and is flagged, not evidence. The free
channels are individually under-identified (δ and the direct θ₀ are near-collinear), which is
why the structural restriction δ=b_x·ζ·θ produces a tighter composite coefficient.
That tightening is slope information imposed through a fixed proportionality and
is not independent identification of the direct channel. See the report §4.

`build_report_gustavo_capiq.py` uses a temporal-disaggregation competition series:
annual Gustavo level with Capital IQ's average quarterly contribution profile
(`s_q`), over the full Gustavo span 1974-2013 (steep-Phillips era included). δ and
κ₀ come out positive (κ₀ strongly with the unemployment gap); the HSA-restricted θ
is positive but marginal (a benchmarked disaggregation keeps the Gustavo within-year
variation small).

`build_report.py` produces the formal PDF: (0) estimation/data/model, (1)
coefficient table (+ activity-cell comparison + sample robustness), (2)
prior-vs-posterior, (3) decomposition (trend + nonlinear slope), (4) convergence.
`--activity` selects the focus cell; the unemployment-gap version is written to
`results/hsa_ppi_identification_report_neg_unemp_gap.pdf`.

## Outputs (in `results/`, git-ignored)

- `report.md` — coefficient tables per sample × variant.
- `tables/coefficient_summaries.csv`, `tables/hsa_channels.csv`.
- `figures/hsa_channels.png` — δ and θ across samples.
- `manifest.json` — config, convergence gate, timing.

## Correct reading of the primary sample (1996Q1–2017Q4)

Two activity cells are estimated: inverse markup and negative unemployment gap.
In PPI/inverse markup the free slope is `0.916 [0.092, 1.739]`, but free direct
`theta = 0.120 [-0.664, 0.865]`. Fixed-HSA theta is
`0.194 [0.040, 0.345]`; this is the slope signal mapped through the fixed
restriction. In PPI/unemployment gap, both the free slope and free theta intervals
include zero. Quadratic preprocessing sensitivities may describe curvature within
that fixed-state design, but they do not repair direct-channel or joint-state
identification. The sign also depends on the competition construction, so this
folder is a sensitivity result rather than the repository's preferred structural
estimate.
