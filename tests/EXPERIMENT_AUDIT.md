# Experiment audit

Audit date: 2026-08-25  
Repository baseline: `18761cf` plus the current uncommitted test bundles and saved results.

This document records what each directory under `tests/` actually tested and what
the saved output supports. It deliberately separates four questions:

1. Did the sampler converge?
2. Did the posterior learn the parameter relative to its prior?
3. Did an unrestricted 95% interval have the theoretical sign?
4. Is the run long enough and correctly specified for empirical inference?

The audit is based on each folder's executable code, configuration, README,
manifest, coefficient/convergence tables, and saved JSON summaries. Pre-existing
MCMC was not rerun for the original documentation pass. The subsequently added
`competition_slope_change` bundle was run at both smoke and full profiles. The
subsequent `active_firm_stock_bds_bed` bundle was run at mock and smoke profiles,
`gustavo_state_capitaliq_cycle` was run at its mock-only profile, and
`mixed_frequency_gustavo_capitaliq` was run at its mock profile. Each new
bundle's three unit tests passed.

Passing one question does not imply passing the others. In particular, a positive
coefficient created by a fixed HSA proportionality coefficient is not independent
identification of the direct channel, and a low WAIC from a nonconverged run is not
model evidence.

Terminology also matters. Several folders use `N` as a code-level name for a
Gustavo/Capital-IQ effective-firm or inverse-concentration state. Unless the input
is a literal active-firm count, this is an empirical competition measure `C`, not
automatically the theoretical HSA mass of active firms. The audit retains folder
variable names when describing implemented equations but does not treat that
measurement mapping as established.

## Status vocabulary

- **Empirical full run:** long saved run with usable convergence diagnostics.
- **Computational pass, identification fail:** sampling is reliable, but the key
  structural interval includes zero or remains close to its prior.
- **Exploratory diagnostic:** short, mock, screen, cut, or sensitivity calculation;
  useful for finding problems, not for reporting a structural estimate.
- **Invalid for inference:** a saved run fails its convergence gate or uses an
  intentionally incomplete design.

## Master record

| Test bundle | What was tested | Saved-run status | Result that may be carried forward |
|---|---|---|---|
| `active_firm_stock_bds_bed` | Annual BDS firm-level anchor plus quarterly BED establishment-flow timing, inflation-cut state, and free-`theta_N` oracle/propagated recovery | Smoke only; state ESS and recovery fail; not for inference | The external-N construction is operational, but the aggregate sample does not recover the direct channel after state uncertainty is propagated; do not promote HSA restrictions or a full run. |
| `gustavo_state_capitaliq_cycle` | Exact Gustavo Q4 slow state, cut Capital IQ AR(2) cycle, QoQ direct/free-combined fits, staged recovery, then a post-gate varying-theta/free-dynamic/HSA-dynamic diagnostic | Computational pass; static recovery gate fails; explicitly requested dynamic diagnostic also fails identification; not for inference | Average `theta_0` remains directionally positive, but gamma, unrestricted lambda, and both derived HSA slope terms include zero; all twelve dynamic holdout ELPDs are below constant theta. |
| `mixed_frequency_gustavo_capitaliq` | Exact total Gustavo Q4 constraints and two Capital IQ QoQ-growth measurements in one cut mixed-frequency RW/AR(2) state; blocked measurement prediction and free direct/combined QoQ NKPCs | Mock gate fails convergence and blocked validation; not for inference | Exact accounting works, but the hybrid forecasts hidden Capital IQ growth worse than both allocation benchmarks and all `theta_N` means have the wrong HSA orientation; do not promote or impose HSA restrictions. |
| `beta_convexity` | Bounds and adding-up restrictions for backward/forward inflation coefficients | Exploratory diagnostic | The restrictions move posterior means but do not identify the competition slope. |
| `capital_iq_quarterly` | Observed quarterly Capital IQ firm/revenue effective counts, fast-state and error-law sensitivity | Empirical full run; computational pass, structural robustness fail | PPI-markup slope evidence appears under EWMA construction, but the HSA/direct result is sensitive to fast-state construction and AR(1) errors. |
| `competition_slope_change` | Competition-only AR(2) decomposition, state-draw propagation, slope-only MA(3) NKPC, historical slope changes, and direct timing diagnostics | Empirical full run; computational pass, identification fail | PPI/inverse-markup gives the strongest positive slope direction, but its 95% interval crosses zero; direct timing is prior-wide and the state variance allocation is omega-prior-sensitive. |
| `design` | Executable nine-cell architecture and compliance matrix | Test-run architecture audit | The pipeline runs, but important follow-up modules are missing/data-blocked and all nine economic-effect intervals include zero. |
| `hsa_deep_identification` | Frozen identification-first search, exact-N AR(2), annual allocation, MA(3), QoQ, recovery and restrictions | Screen/mock/quick only; no promoted model | No candidate passed the complete identification, sign, convergence, and validation sequence. |
| `hsa_exact_n_decomposition` | Exact `N = Nbar + Nhat`, annual Gustavo constraint, variance-share parameterization, seven models | Empirical full run; saved bundle gate pass, identification fail | Exact identity and the recorded R-hat gate pass; `theta`, `delta`, `gamma`, and free `lambda` remain weak. |
| `hsa_lambda_dynamic` | Joint-state seven-model static/dynamic HSA with estimated `lambda` | Invalid for inference | Overall convergence fails and `lambda` is essentially prior-wide. |
| `hsa_nested_validation` | Active 24-fit nested model grid with exact-N AR(2) state | Empirical full run; bundle gate pass, below later ESS standard, identification fail | All direct-channel intervals include zero; free `lambda` is unidentified; no formal marginal likelihood was run. |
| `hsa_ppi_identification` | Preprocessed observed-N PPI signal, fixed versus joint state, and five-model comparison | Mixed: one converged preprocessing run plus joint diagnostics | A PPI-markup slope is visible with a fixed preprocessing choice; free direct `theta` and the joint latent HSA comparison are not validated. |
| `hsa_theta_bridge` | Cut versus joint state crossed with fixed `lambda=6` versus free `lambda` | Empirical full run; saved bundle gate pass, factorization fail | Cut/joint makes little difference; freeing `lambda` destroys separate identification of `theta`. |
| `markup_feedback` | Importance reweighting from modular cut to full feedback | Exploratory diagnostic | Low/moderate feedback is inspectable; the full-feedback endpoint has poor overlap and cannot stand in for a joint fit. |
| `markup_full_joint` | Core-CPI/unemployment full-joint markup state | Invalid for inference | State chains do not converge; the apparently positive coefficient cannot be interpreted. |
| `markup_interpolation` | Placement of quarterly markup information between exact Q4 anchors | Invalid for inference | Timing changes means, but state and coefficient chains are too short/nonconverged. |
| `markup_measurement` | Q4-anchored inverse-markup measurement bridge and four QoQ cells | Invalid for inference | State convergence fails and direct-channel intervals include zero. |
| `n_gustavo_state_space` | Annual Gustavo-only mixed-frequency quarterly state | Invalid for inference | Annual data alone do not stably identify the quarterly slow/cycle split in the saved run. |
| `nolag_price_gap` | No lagged inflation, AR(1) error, broad price/activity/timing grid | Exploratory diagnostic | Removing the lag/intercept-style structure does not robustly recover `theta`; preferred timing is unstable. |
| `observed_hhi` | SEC inverse-HHI proxy, timing/error-law sensitivity and recovery | Exploratory diagnostic | No primary cell has a robust nonzero effect; sign changes with timing and short coverage has low power. |

## Detailed results

### `beta_convexity`

The test compared an unrestricted hybrid NKPC, bounds
`alpha_b, alpha_f in [0,1]`, and the exact adding-up restriction
`alpha_b + alpha_f = 1` in three QoQ E2 cells. The saved run has 400 iterations,
150 warmup draws, thinning 2, and two chains, so it is a short diagnostic rather
than a production estimate. There are nine fitted rows.

The competition-slope intervals are very wide and all include zero. Examples are
PPI/inverse-markup `delta = 0.764 [-9.227, 10.905]` without the restriction and
`2.019 [-8.014, 11.402]` with adding-up; core-CPI/negative-unemployment-gap is
`-0.931 [-3.129, 1.393]` without it. The restrictions therefore do not solve
competition-channel identification.

### `capital_iq_quarterly`

This bundle uses observed quarterly Capital IQ effective-firm counts, comparing
firm-count and revenue weighting, iid and persistent AR(1) inflation errors, and
alternative fast-state constructions. The main EWMA full run used 20,000
iterations, 5,000 warmup, thinning 5, and four chains. It passed its computational
gate: maximum R-hat 1.0021 and minimum ESS 2,191.

For PPI/inverse markup, the AR(1) slope estimate was `0.329 [-0.100, 0.742]`
with firm weights and `0.416 [-0.019, 0.837]` with revenue weights. The direct
coefficient was positive under iid errors (`0.324 [0.105, 0.539]` and
`0.327 [0.082, 0.570]`) but not under AR(1) errors (`0.273 [-0.097, 0.608]` and
`0.277 [-0.149, 0.667]`). Thus the apparent direct effect is disturbance-law
sensitive.

The separate AR(2) fast-definition full run also converged well (maximum R-hat
1.0017, minimum ESS 2,371), but its HSA/composite intervals were broad: for
PPI/inverse markup, firm-weight AR(1) `delta_HSA = -0.113 [-3.323, 3.170]` and
revenue-weight AR(1) `1.160 [-1.942, 4.381]`. The correct conclusion is excellent
computation but no robust structural result across fast-state definitions.

### `design`

This is a nine-cell, three-price by three-activity pipeline test for E0/E1/E2
models. Its manifest explicitly says `is_test_run: true`; the saved sample is
1982Q1--2012Q4 with 124 observations. The measurement information ratio is
0.786 and the measurement gate is false. Persistent-error, low-frequency,
alternative-state-law, formal Chib marginal-likelihood, real-time forecast,
PPI-expectations, and markup-scale modules are incomplete or data-blocked.

All nine saved economic-effect 95% intervals include zero. Cut/joint conflicts
are flagged for `theta` in cells 1, 3, 4, 7, 8, and 9. This bundle demonstrates
the reporting and compliance architecture; it is not evidence for HSA.

### `hsa_exact_n_decomposition`

This was the first full exact-observation decomposition:

```math
N_t=\bar N_t+\hat N_t,
\qquad
\sigma_{\bar N}^2=\omega\tau^2,
\qquad
\sigma_{\hat N}^2=(1-\omega)\tau^2.
```

Annual Gustavo totals are imposed exactly and Capital IQ supplies the within-year
allocation prior. The state run used 20,000 iterations, 6,000 warmup, thinning 7,
and four chains; model runs used 12,000, 4,000, thinning 4, and four chains. The
maximum R-hat was 1.0093 and the exact-identity error was `2.22e-16`. The saved
gate is R-hat based and does not establish the later deep-audit ESS>=800 rule for
every state parameter.

The posterior allocation was `omega = 0.588 [0.365, 0.786]`, `tau = 0.199`, and
AR(1) cycle persistence `rho = 0.800 [0.547, 0.898]`. This is the run that raised
the concern that too much innovation variance was assigned to the slow state.
Structural coefficients were weak: slope-only `delta = 0.0388 [-0.107, 0.182]`,
direct-only `theta = 0.0057 [-0.196, 0.212]`, and free-static
`theta = 0.0055 [-0.197, 0.207]`. In the free-lambda static model,
`lambda = -0.004 [-14.344, 14.377]`. Dynamic `gamma` and quadratic terms were
also weak. WAIC values were approximately 712--714, with no clear improvement
over CES. Exact accounting worked; HSA identification did not.

### `hsa_lambda_dynamic`

This jointly estimated the state and seven nested static/dynamic NKPC models,
including free `lambda`. The saved full profile reports 24,000 iterations,
8,000 warmup, thinning 8, four chains, and 5,000 particles, but the overall gate
failed. Model maximum R-hats include 1.112 for slope, 1.062 for free static,
1.069 for HSA static, 1.191 for free dynamic, and 1.131 for HSA dynamic.

The substantive parameters were also unidentified: slope-only
`delta = 0.0368 [-0.117, 0.185]`, direct-only
`theta = 0.0244 [-0.182, 0.236]`, HSA-static
`lambda = 0.410 [-15.255, 15.642]`, and HSA-dynamic
`lambda = 0.129 [-15.188, 14.586]`. Neither WAIC nor approximate marginal
likelihood from this nonconverged run may be used as HSA evidence.

### `hsa_nested_validation`

The active v4 experiment contains 24 fits. PPI and core CPI are kept separate.
For each price, negative unemployment gap has CES, slope, direct, and free-combined
models; inverse markup adds fixed `lambda` 3/6/9 and a free-lambda diagnostic.
State-cut variants and obsolete B4/B5 variants were removed. The active full run
used 20,000 iterations, 6,000 warmup, thinning 7, and four chains. It passed the
bundle's saved gate with maximum R-hat 1.0081 and exact-identity error
`2.22e-16`. Its minimum state bulk ESS is 707, so it would not pass the stricter
ESS>=800 rule adopted later in `hsa_deep_identification`.

Every free/direct `theta` interval includes zero. Examples are PPI/inverse-markup
free combined `theta = 0.0048 [-0.207, 0.211]`,
`delta = 0.0977 [-0.174, 0.372]`; core-CPI/inverse-markup
`theta = 0.0090 [-0.192, 0.213]`, `delta = 0.0267 [-0.070, 0.124]`;
and PPI/unemployment-gap `theta = 0.0053 [-0.200, 0.216]`.
Fixed-lambda posterior-positive probabilities are roughly 0.71--0.76, but all
95% intervals include zero. Free lambda is prior-wide:
`0.012 [-15.587, 15.793]` for PPI/inverse markup and
`-0.025 [-12.292, 11.999]` for core CPI/inverse markup.

Formal marginal likelihood and annual-origin forecast comparison were not run,
because no candidate passed the preceding identification requirements. The
`full_v1_28_fit_legacy` and `full_v3_ar1_legacy` directories are historical and
must not be mixed with active v4 results.

### `hsa_ppi_identification`

This folder contains three logically different experiments and must not be read
as one result.

First, an observed-Capital-IQ preprocessing run used a one-sided EWMA innovation,
lag one, AR(1) inflation error, and 12,000 iterations with four chains. It converged
(maximum R-hat 1.001). In the 1996Q3--2017Q4 PPI/inverse-markup cell, the free
slope was `0.916 [0.092, 1.739]`, while free direct
`theta = 0.120 [-0.664, 0.865]`. Under fixed HSA proportionality, theta became
`0.194 [0.040, 0.345]`; that is slope information transferred through the
restriction, not independent direct-channel identification.

Second, the `two_models` comparison crossed fixed EWMA and joint decomposition.
Its restricted/composite PPI-markup theta was `0.060 [-0.013, 0.132]` fixed and
`0.075 [-0.067, 0.217]` joint; PPI-gap was `0.054 [-0.008, 0.113]` fixed and
`0.159 [-0.028, 0.346]` joint. Core-CPI intervals also included zero.

Third, the five-model latent-state comparison did not validate HSA over CES.
PPI-markup WAIC values were about 747.5--747.9, while CES had the better saved
log-marginal diagnostic. Apparent gains in PPI-gap and core-gap coincide with
R-hats as high as 1.058, 1.557, and 1.322 and therefore are invalid. The folder's
old shorthand that theta was “identified” applies only to a fixed-restriction,
preprocessed signal and should not be interpreted as free direct identification.

### `hsa_theta_bridge`

This controlled experiment crossed cut versus joint-state estimation with fixed
`lambda=6` versus free lambda. The full run passed its computational gate
(maximum R-hat 1.009; exact-identity error `2.22e-16`). Results were nearly the
same under cut and joint state. Fixed lambda gave
`theta = 0.036 [-0.022, 0.089]` under cut and
`0.037 [-0.021, 0.091]` under joint estimation, with positive probability about
0.91 but intervals still including zero.

When lambda was free, cut gave `theta = 0.005 [-0.146, 0.160]` and
`lambda = 0.15 [-14.77, 15.31]`; joint gave
`theta = 0.002 [-0.156, 0.163]` and
`lambda = 0.09 [-14.73, 14.64]`. The identified-looking object is the product
`delta=lambda*theta`, not its two factors. The main problem is factorization,
not inflation feedback into the state.

### `markup_measurement`, `markup_interpolation`, and `markup_full_joint`

All three saved runs are explicitly exploratory (300 iterations, 100 warmup,
two chains) and fail state convergence.

- `markup_measurement`: state R-hats are about 1.08 under iid errors and reach
  1.31 for the AR(1) measurement drift; direct-channel intervals include zero.
- `markup_interpolation`: moving information between exact Q4 anchors changes
  coefficient means, but state R-hats range roughly 1.06--1.34 and the measurement
  drift reaches about 1.66.
- `markup_full_joint`: `d_q` has R-hat 1.694 and ESS 3.3; `qbar` and `qhat` R-hats
  are about 1.23. The cut/joint coefficient similarity is not enough to validate
  the result.

These runs identified computational failure modes; none supplies an empirical
HSA estimate.

### `markup_feedback`

This importance-sampling experiment gradually raises the inflation likelihood
from a modular cut (`lambda_feedback=0`) to full feedback (`1`). This lambda is a
feedback weight, not the structural HSA parameter. At full feedback, raw ESS is
29 of 400, Pareto `k=0.862`, and the maximum normalized weight is 0.136. The
coefficient path (`theta` about 0.169 to 0.245 and slope about 0.067 to 0.098)
is interpretable only over the low/moderate-feedback region with adequate overlap.
The endpoint cannot replace a directly sampled joint posterior.

### `n_gustavo_state_space`

This estimates a quarterly slow/cycle state using only 32 annual Gustavo Q4
observations over 1982Q1--2013Q4. The saved run has only 180 iterations, 80
warmup, thinning 2, and two chains. State convergence is poor:
`d_q` R-hat 1.263, `phi_q` 1.791, `sigma_qbar` 1.981, and `sigma_qhat` 1.160.
State-law sensitivity estimates of theta are broad and approximately prior-wide,
for example `-0.021 [-0.813, 0.714]` at `phi=0.5`. Annual-only information does
not stably recover a quarterly decomposition in this saved test.

### `nolag_price_gap`

This is a 200-fit screen across five prices, four activity measures, four model
forms, and six fast-state timings. Lagged inflation is removed and persistence is
placed in an AR(1) disturbance. The manifest marks it as a test run and it consumes
the short, nonconverged annual-Gustavo state above. Simpler E0 models often beat
E2, and the best fast-state timing varies across cells. It therefore shows that
removing lagged inflation does not robustly rescue the direct channel; it is not
a confirmatory result.

### `observed_hhi`

This uses an observed SEC inverse-HHI proxy and runs 297 timing/error-law tasks
with 600 iterations, 200 warmup, thinning 2, and two chains. Seven of nine primary
cells meet the bundle's limited convergence screen, but no primary cell has a
nonzero effect. Cell 1 has only 23 observations and a persistent-error estimate
`theta = -2.21 [-8.23, 3.51]`. Across timing choices the mean ranges from about
-2.21 to +2.65, so even the sign is not stable. The recovery exercise detects an
effect of 0.25 only 5.3% of the time at 24 observations and 28.3% at 124
observations in cell 1. The observed-HHI design is underpowered and timing-sensitive.

### `hsa_deep_identification`

This is the latest identification-first audit. Its protocol was frozen before
reading candidate results. It compares exact-N AR(2) state laws, annual-allocation
slow innovations, valid MA(3) errors for overlapping YoY inflation, genuine QoQ
data/expectations, static/dynamic restrictions, no-intercept and inflation
adding-up sensitivities, non-overlapping Q4 observations, and simulation recovery.

The deterministic screen evaluated 1,296 successful likelihoods and found zero
identified candidates. The dynamic screen evaluated 192 rows and found zero rows
passing both theta-path and kappa-path gates. Non-overlapping Q4 discovery and
validation splits did not produce a stable candidate.

The leading annual-allocation AR(2), PPI/inverse-markup, free-channel quick run
achieved maximum R-hat 1.004 and exact identity error `2.22e-16`, but state ESS
values were below the frozen 800 gate. It estimated
`delta = 0.190 [-0.061, 0.438]`, positive probability 0.937 and posterior/prior
SD ratio 0.778; direct `theta = -0.007 [-0.204, 0.197]`, positive probability
0.471 and SD ratio 0.971. The posterior probability that kappa is positive at
at least 95% of sample dates was only 0.174.

With fixed `lambda=6`, theta was `0.0296 [-0.0095, 0.0678]`, positive probability
0.929 and SD ratio 0.777; maximum R-hat was 1.012 and state ESS values again fell
below 800. This does not pass either convergence or identification gates.

The annual-allocation AR(2) state changed the variance split in the intended
direction: in the leading free run `omega = 0.0468 [0.0014, 0.2093]`, mean slow
innovation variance 0.00090, and cycle innovation variance 0.01771. Thus AR(2)
and annual allocation stabilize the earlier excessive slow variance, but they do
not create direct-channel information.

The QoQ mock across all four cells produced theta means from about -0.003 to
-0.008, 95% intervals near `[-0.20, 0.18]`, and posterior/prior SD ratios
0.95--1.03. The simulation recovery run covered all truths but had maximum R-hat
1.067; true `theta=0.16` was recovered as
`0.091 [-0.221, 0.408]`, while true `delta=0.10` was
`0.244 [0.004, 0.500]`. This demonstrates weak direct-channel power even in a
controlled sample of the same length.

Removing the NKPC intercept and imposing `alpha_b+alpha_f=1` did not identify
theta. Free lambda remained essentially prior-wide (about
`-0.66 [-17.57, 17.89]` in the leading mock). Dense-Gaussian FFBS validation has
two passing unit tests, but algorithmic correctness does not change the empirical
identification result.

No candidate passed the frozen sequence. Therefore no production/full run was
promoted and no formal marginal likelihood or Bayes factor was computed. This is
the correct stopping rule: model evidence is not calculated for a structurally
unidentified candidate.

### `competition_slope_change`

This test implements the next admissible semi-structural baseline rather than
reopening the full HSA factorization. The competition coordinate is
`c_t = 10(log C_t - log C_ref)` with the predeclared 1984 Gustavo value as its
origin. It is an empirical effective-competition coordinate, not a structural
active-firm stock. The competition block is estimated from competition data only:

```math
c_t^{obs}=\bar c_t+\hat c_t,
\qquad
\sigma_{\bar c}^2=\omega\tau^2,
\qquad
\sigma_{\hat c}^2=(1-\omega)\tau^2.
```

The slow transition uses the annual Gustavo change allocated across quarters by
Capital IQ when coherent and otherwise shrinks to a robust average quarterly
profile. The primary cycle is stochastic AR(2); AR(1) and two alternative omega
priors are sensitivity checks. Inflation never feeds back into this block. Saved
competition-state draws, not a posterior-mean plug-in, are propagated into four
separate PPI/core-CPI by inverse-markup/unemployment-gap slope cells:

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
 +(\kappa_0+\delta\bar c_t)x_t+\varepsilon_t,
\qquad
\varepsilon_t=u_t+\psi_1u_{t-1}+\psi_2u_{t-2}+\psi_3u_{t-3}.
```

The full run used four chains, 20,000 state iterations with 6,000 warmup and
thinning 7, and 16,000 NKPC iterations with 5,000 warmup and thinning 5. The
primary computational gate passed: maximum R-hat 1.0012, minimum bulk ESS
4187.6, minimum tail ESS 4984.2, and exact-accounting error `2.22e-16`.

Economic identification remains incomplete. PPI/inverse markup has the strongest
direction, `delta = 0.196 [-0.059, 0.442]`, `P(delta>0)=0.936`, and a
posterior/prior SD ratio of 0.792. The other cells are weaker: PPI/unemployment
gap `0.020 [-0.123, 0.166]`, core-CPI/unemployment gap
`0.032 [-0.040, 0.098]`, and core-CPI/inverse markup
`0.008 [-0.083, 0.099]`. Thus no slope interval excludes zero. Because the slow
competition coordinate falls over the full sample, the main PPI/inverse-markup
historical estimand is `Delta kappa_comp = -0.450 [-1.032, 0.136]`; it too crosses
zero.

The current, lag-one, and lead-one `theta_C` diagnostics are all essentially
prior-wide. For PPI/inverse markup they are respectively
`-0.013 [-0.220, 0.187]`, `0.009 [-0.197, 0.213]`, and
`0.012 [-0.190, 0.220]`, with posterior/prior SD ratios 0.986--0.993. These are
coefficients on an empirical cyclical concentration coordinate, not estimates of
the structural active-firm `theta_N`.

The primary Beta(2,8) omega prior gives
`omega = 0.057 [0.002, 0.273]`. Changing only that prior gives
`0.330 [0.003, 0.984]` under Beta(2,2) and
`0.191 [0.000, 0.996]` under Beta(1,1). The alternative samplers converge well
enough to expose substantive prior sensitivity. The short AR(1) check does not
converge to the full standard (R-hat 1.036, minimum bulk ESS 64) and is not a
headline alternative. The honest conclusion is a clean computational pass and a
useful semi-structural slope direction, but not a fully identified structural
competition effect. No lambda, full HSA cross-equation restriction, marginal
likelihood, or causal policy counterfactual is promoted.

### `active_firm_stock_bds_bed`

This bundle implements the external-active-firm step without recycling the
effective-competition `C=1/HHI` coordinate. Annual BDS `FIRM` counts supply 45
level anchors from 1978 through 2022. BED supplies 116 quarters with matched
establishment births and deaths. Because BED measures establishments rather than
firms, its standardized net-entry flow is used only as a noisy timing measurement
with a freely estimated intercept, loading, and variance:

```math
y_y^{BDS}=\bar n_{y,Q1}+\hat n_{y,Q1},
\qquad
z_t^{BED}=a_E+\ell_E\Delta(\bar n_t+\hat n_t)+e_t^E.
```

The slow state is a random walk with drift and the cycle is a stochastic AR(2).
Their innovation variances are parameterized as
`sigma_bar^2=omega*tau^2` and `sigma_hat^2=(1-omega)*tau^2`. Only BDS and BED
enter this state posterior. Saved state draws are then propagated into the
constant-slope, free-direct-channel MA(3) NKPC:

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
 +(\kappa_0)x_t-\theta_N\hat n_t+\varepsilon_t.
```

The recorded smoke profile used four state chains with 5,000 iterations and two
recovery chains with 1,000 iterations for each of ten replicates and eleven
injected values. It is explicitly not inferential. State maximum R-hat was
`1.0226`; minimum bulk ESS was `115.1` for `omega`, below the frozen smoke gate
of 300. The state means were `omega=0.694 [0.516,0.855]`,
`tau=0.0536 [0.0395,0.0699]`, AR(2) damping
`0.666 [0.398,0.874]`, and period `16.74 [8.38,23.50]` quarters.

The observed NKPC regressions mixed adequately but did not identify the direct
channel. PPI/inverse-markup gave
`theta_N=-1.712 [-17.620,13.798]`, `P(theta_N>0)=0.418`; PPI/negative-
unemployment-gap gave `-1.293 [-16.781,15.563]`, probability `0.425`.

Recovery separates two failures. If the injected `nhat` path is treated as
perfectly known, smoke detection is 30% at `theta_N=20`, 70% at `30`, and 100%
at `50`; smaller effects are largely undetectable in the 83-quarter aggregate
sample. When the BDS/BED state posterior is propagated, detection is zero for
every injected value through `50`. The latter is the admissible design for an
empirical claim. Consequently the full profile, `delta`, structural `lambda`,
HSA restrictions, and marginal evidence are deliberately not run. This is a
predeclared stopping decision, not selection on a preferred coefficient sign.

### `gustavo_state_capitaliq_cycle`

This mock gives the two effective-firm datasets deliberately separate jobs. The
Gustavo coordinate is anchored exactly at every annual Q4 observation and its
quarterly path is drawn from a Gaussian bridge:

```math
g_y=10\log(N_y^G/N_{1993}^G),
\qquad
\bar n_{y,Q4}=g_y.
```

Capital IQ cannot update that slow posterior. Conditional on saved Gustavo draws,
each firm- or revenue-weighted Capital IQ coordinate is measured as

```math
c_{j,t}=a_j+b_j\bar n_t+\hat n_{j,t}+e_{j,t},
```

where `nhat` follows a stationary stochastic AR(2). The estimated intercept and
loading absorb different data universes and scales. Inflation is used only in
the subsequent annualized-QoQ free-cycle equation

```math
\pi_t^q=400(\log P_t-\log P_{t-1}),
\qquad
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+\kappa_0x_t-\theta_{CIQ}\hat n_{j,t}+\varepsilon_t,
```

using the genuine SPF one-quarter-ahead forecast in matching annualized-log
units. IID is primary and a persistent AR(1) error is robustness. The coefficient
is named `theta_CIQ`, not structural `theta_N`. The earlier overlapping-YoY MA(3)
result is retained under `mock_yoy_legacy` and is no longer the primary result.

The exact Gustavo anchor error is `9.99e-16`. The two cycle fits are usable for
a mock: maximum R-hat is 1.061 for firm weights and 1.040 for revenue weights.
The firm-weighted Capital IQ loading on the Gustavo slow state is
`-0.383 [-0.761,-0.061]`, while the revenue-weighted loading is
`0.027 [-0.295,0.330]`. This disagreement is a warning against treating the two
effective-firm levels as interchangeable; it is not a structural sign result.

All observed QoQ free-cycle intervals include zero, but their positive direction
is stronger than in the archived YoY run. Under primary IID PPI/inverse markup,
firm-weighted `theta_CIQ=0.718 [-0.766,2.169]`, positive probability 0.837;
revenue-weighted `0.663 [-0.820,2.083]`, probability 0.823. Persistent AR(1)
gives respectively `0.665 [-1.165,2.464]` and
`0.676 [-0.920,2.300]`. The direction survives the error-law change, but the
intervals widen and remain far from sign identification.

Recovery is more favorable than the BDS/BED external-stock test because the
Capital IQ cycle has a materially larger quarterly scale. With three replicates,
IID oracle recovery detects one of three at `theta_CIQ=1` and all replicates at
`3`; propagated recovery detects none at `1`, one of three at `3`, and all at
`10`. Under AR(1), propagated recovery detects none through `3` and all at `10`.
These are not power estimates. The lowest AR(1) recovery bulk ESS is 20.5, so the
full mock gate fails even though every observed QoQ fit has maximum R-hat at most
1.013 and theta bulk ESS above 588. No full run, `delta`, structural `lambda`,
HSA restriction, or marginal evidence was promoted from that recovery stage.

The subsequent nested diagnostic reused the same competition posterior draws and
added only the centered slow-state slope interaction:

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+[\kappa_0+\delta(\bar n_t-\overline{\bar n})]x_t
-\theta_{CIQ}\hat n_{j,t}+\varepsilon_t.
```

It used four chains, 2,000 iterations, 600 warmup iterations, and thinning by two.
All eight fits pass the computational gate: maximum R-hat is `1.0025` and minimum
bulk ESS is `1,618.8`. `theta_CIQ` passes the predeclared descriptive retention
screen in all eight cells. Under IID PPI/negative unemployment gap, firm weighting
gives `theta_CIQ=0.781 [-0.800,2.346]`, positive probability `0.829`; revenue
weighting gives `0.729 [-0.810,2.197]`, also probability `0.829`. These are at
least as favorable as the direct-only probabilities `0.788` and `0.800`.

The slope coefficient is weaker. In the same two cells,
`delta=0.166 [-0.520,0.845]`, positive probability `0.673`, and
`0.164 [-0.514,0.841]`, probability `0.678`. Every delta interval includes zero.
The largest absolute posterior `Corr(delta,theta_CIQ)` is `0.352`, so severe
posterior confounding is not the reason for the weak delta result. This diagnostic
shows that the positive direct update is not merely proxying the omitted slow
slope interaction. It does not test `delta=lambda*theta`: with an unrestricted
real static lambda, that equality is a reparameterization for nonzero theta.

The final staged validation then estimated 17 observed-data fits and 640 recovery
fits under a predeclared promotion rule. Effects were injected in standardized
units. For the empirical unemployment-gap magnitudes
`(s_delta,s_theta)=(0.06,0.11)`, propagated-state suggestive recovery is `0.100`
for delta and `0.333` for theta_CIQ. Oracle-state recovery is `0.133` and `0.333`.
Thus state uncertainty is not the main limitation. At the much larger joint
effect `(0.40,0.40)`, propagated recovery rises to `0.767` and `0.833`, but delta
still misses the `0.80` promotion threshold and theta coverage falls to `0.70`.

The recovery computation itself passes: maximum R-hat is `1.0395`, minimum bulk
ESS is `100.6`, both observed-size coverages are `1.00`, and null false-positive
rates are zero. Thirteen initially short fits were rerun under the predeclared
longer four-chain convergence rule; no retry was selected by coefficient outcome.

Full-sample PSIS-LOO slightly favors free combined, but every comparison has
Pareto-k above one and is therefore unreliable. The fixed 2010Q1-2013Q4 holdout
ELPD is lower for free combined in all four cells. Savage-Dickey BF01 values of
`1.38` to `1.57` mildly favor `delta=0`. The observed-effect recovery gate failed
for both parameters, so free dynamic and HSA-restricted dynamic were not run.
This is the required stopping result, not an incomplete estimation stage.

At explicit user direction, a separate post-gate diagnostic subsequently ran
the varying-theta, free-dynamic, and HSA-restricted-dynamic models. It used

```math
\theta_t=\theta_0+\gamma\bar n_t^c,
\qquad
\kappa_t^{free}=\kappa_0+\delta_1\bar n_t^c+\delta_2q_t^{(2)},
```

and imposed

```math
\kappa_t^{HSA}=\kappa_0+\lambda\theta_0\bar n_t^c
+\frac{\lambda\gamma}{2}q_t^{(2)}.
```

The 27 observed fits and 300 recovery fits pass their computational gates:
maximum observed R-hat is `1.0011`, minimum observed bulk ESS is `4,074.5`,
maximum recovery R-hat is `1.0129`, and minimum recovery bulk ESS is `498.5`.
Identification does not pass. In the primary varying-theta fits,
`P(theta_0>0)` is `0.804`--`0.830`, but all theta intervals include zero and all
gamma intervals lean negative while including zero. At injected standardized
`gamma=0.10`, propagated-state suggestive recovery is `0.533` and strong
recovery is only `0.067`; strong recovery reaches `0.800` only at `gamma=0.40`.

Every HSA lambda and both derived slope intervals include zero. Firm-weighted
PPI/unemployment gives `lambda=0.402 [-5.288,5.952]`; its AR(1) robustness fit is
`0.401 [-5.975,6.292]`. All twelve dynamic model/cell combinations lose
2010Q1--2013Q4 holdout ELPD relative to constant theta. Thus the explicit run
confirms the original stopping diagnosis: dynamic computation is sound, but
neither time-varying theta nor the HSA cross-equation restrictions are identified.

### `mixed_frequency_gustavo_capitaliq`

This follow-up does not assign Gustavo mechanically to the slow state. It imposes
Gustavo as an exact annual-Q4 condition on total competition,

```math
n_t=\bar n_t+\hat n_t,
\qquad
g_y=(\bar n_t+\hat n_t)_{yQ4},
```

and uses two Capital IQ QoQ changes as noisy measurements of total competition
growth:

```math
\Delta c_{j,t}=a_j+b_j\Delta(\bar n_t+\hat n_t)+e_{j,t}.
```

The slow state is a random walk around the previously estimated average quarterly
allocation drift; the cycle is a stable damping/period AR(2). Innovations use the
requested variance-share parameterization
`sigma_bar^2=omega*tau^2` and `sigma_hat^2=(1-omega)*tau^2`. Inflation is absent
from this likelihood. Capital IQ is reindexed before differencing, so sparse
annual pre-overlap values are not mistaken for quarterly changes.

An initial code check exposed a genuine likelihood degeneracy: treating exact
Gustavo values as zero-noise Gaussian densities rewarded `tau -> 0`, because the
annual-change drift already matched consecutive Q4 totals. The retained
implementation instead conditions exactly on each Gustavo equality and evaluates
only the conditional Capital IQ measurement density. The resulting MAP is
interior (`omega=0.254`, `tau=0.276`) and saved Q4 identity error is
`8.9e-16`.

The saved mock nevertheless fails its predeclared gate. The state has maximum
R-hat `2.390` and minimum bulk ESS `2.5`, driven by AR(2) damping; `tau` and both
Capital IQ loadings also mix poorly. Measurement-only blocked validation hides
both Capital IQ series in three two-year blocks. Mean RMSE is `0.988`, versus
`0.922` for average allocation and `0.852` for equal allocation. Thus the hybrid
is 7.1% worse than the declared average-allocation benchmark, even before its
computational failure is considered.

Four annualized-QoQ PPI/inverse-markup/SPF cells propagated the cut state into
direct-only and free-static-combined equations, each without and with current and
lagged oil controls. Their regression coefficients mix adequately, but every
`theta_N` mean has the wrong HSA orientation. Direct/no-oil is
`-1.694 [-9.746,6.726]`, `P(theta_N>0)=0.331`; direct/oil is
`-2.141 [-7.979,4.787]`, probability `0.215`. In free combined/no-oil,
`theta_N=-1.291 [-9.070,6.555]`, probability `0.383`, while
`delta=0.350 [-2.461,2.874]`, probability `0.612`. Oil controls materially improve
descriptive WAIC, showing that omitted cost shocks matter, but do not rescue the
competition sign.

The run is retained as a failed mock. It is not promoted to quick, simulation
recovery, or HSA restrictions. The next admissible diagnostic is measurement-only:
normalize one Capital IQ loading, compare reduced AR(1)/AR(2) cycles, and perform
leave-one-measure-out plus the same blocked forecast. If no reduced measurement
model beats the allocation benchmarks, this hybrid should be abandoned rather
than rescued through inflation feedback.

## Overall conclusion

Across the repository, the robust result is negative but informative:

1. Exact `N=Nbar+Nhat` accounting can be implemented and sampled reliably.
2. An AR(2) cycle plus annual-allocation slow law substantially reduces the
   excessive slow-state innovation share.
3. A competition slope can appear in selected PPI/inverse-markup preprocessing
   specifications, but it is not robust to the state/error construction.
4. Earlier structural free `theta_N` estimates are weak. The newer QoQ empirical
   `theta_CIQ` is materially updated toward the theoretical direction and survives
   a free slow-slope interaction, but its 95% intervals still include zero and it
   is not yet the theory-aligned active-firm coefficient. Fixed `lambda` can still
   transfer slope information into theta rather than independently identify it.
5. Free `lambda` is not separately identified near `theta=0`, because the static
   likelihood primarily sees the product `lambda*theta`.
6. No honest specification currently satisfies all requested conditions:
   identification of every claimed parameter, unrestricted theoretical signs,
   convergence, validation, and better formal marginal evidence than CES.

This conclusion should be used as the baseline for future tests. A new result
supersedes it only when the new folder records its design before estimation and
passes the documentation and evidence gates in `TEST_BUNDLE_TEMPLATE.md`.
