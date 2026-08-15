# Independent end-to-end technical audit

Date: 2026-08-10  
Repository: `NKPC_HSA_MCMC`  
Audited revision label after fixes: `2026-08-independent-audit-v1`

## Executive verdict

The main data builder is reproducible and most low-level probability calculations are
implemented correctly. The baseline linear-Gaussian FFBS, parameter scaling, conjugate
variance updates, and conditional Particle-Gibbs kernel all survived independent tests.

The reported economic results are nevertheless **not presently trustworthy as a joint
posterior analysis**. The principal annual-Q4 HSA posterior is strongly multimodal in
the AR(2)/state decomposition. The selected audit run has scalar/path diagnostics as bad
as `Rhat=1.83` and bulk ESS about 3. Every annual-Q4 HSA cell in the saved report grid
fails the repository's joint convergence rule. Four-quarter-inflation residuals also
have strong serial correlation under both CES and HSA steady. A positive marginal
`delta` trace is not sufficient to validate the latent states or the derived `kappa_t`
path.

Three material code defects were found and fixed:

1. SEC fiscal-Q4 revenue could subtract a nine-month fact carrying a different XBRL
   revenue concept from the annual fact.
2. The opt-in full-covariance dynamic smoother used an invalid ordinary RTS backward
   conditional despite correlated transition and measurement innovations.
3. The predictive/report pipeline did not use the estimator's sample or observation
   pattern, omitted terms from the implemented equation, and labeled full-posterior
   filtering-density transformations as proper prequential/WAIC/LOO scores.

The SEC data were rebuilt from all 69 raw archives after the first fix. The report is
intentionally not rebuilt: all 138 base report runs carry an older estimation revision,
and the current builder correctly refuses them.

## Reproducibility snapshot

- Main processed-data SHA-256:
  `1eb4fd42f33411d38c8f02ee3322eab3514b7466a6ec2928f47d189b76d52fd9`.
- Corrected SEC-quarterly SHA-256:
  `88838ee8584ba199c7591c74d34cf2c20f24654bf20976e0cce168c9c112f0e2`.
- Corrected SEC model-ready SHA-256:
  `c813b85743bd2677ba8e94b291902480881217ba86da8c504e8a00e6465f0cf3`.
- Clean audit test outcome: **158 passed**, seven numerical/dependency warnings,
  173.73 seconds.
- Selected production settings: two chains, 12,000 iterations, 4,000 burn-in, thin 5.
  The exact dispatcher is `scripts/run_audit_selected.py`.
- Diagnostic PG pilot: two chains, 1,200 iterations, 400 burn-in, thin 2, at 128 and
  512 particles. This pilot is not a production posterior.

Audit outputs are under
`results/audit/2026-08-independent-audit-v1/`. The key machine-readable files are
`selected_scalar_diagnostics.csv`, `selected_path_diagnostics.csv`,
`inflation_residual_diagnostics.csv`, and `particle_gibbs_diagnostics.csv`.

## Actual production call chain

The production route, reconstructed from executable calls rather than prose, is:

1. `scripts/01_build_data.py:9-15` calls
   `nkpc_hsa.dataprep.build.build_processed_dataset`.
2. `src/nkpc_hsa/dataprep/build.py:208-226` calls the legacy
   `func_data_build.build_dataset`, then adds HP output, labor share, and the legacy BED
   stock before writing `model_ready.csv`.
3. `src/nkpc_hsa/dataprep/func_data_build.py:273-319` loads every source, aligns all
   series at quarter end, concatenates, and constructs one-quarter lags.
4. `scripts/13_estimate_cpi_ppi_report.py` resolves `configs/models.yaml`, loads
   `model_ready.csv` with a datetime index, forms the requested job grid, and calls
   `run_model`.
5. `src/nkpc_hsa/inference/wrappers.py:130-230` applies the declared sample before
   complete-case selection. Lines 291-389 construct annual-Q4, interpolated, or observed
   competition measurements.
6. `src/nkpc_hsa/inference/wrappers.py:580-675` dispatches CES, HSA steady, HSA dynamic,
   HSA full, and HSA constant-theta. HSA full is routed to Particle Gibbs; the other HSA
   variants use exact joint FFBS where linear-Gaussian.
7. The wrapper stacks chains, writes NetCDF plus metadata/config/data hashes, and the
   diagnostics/report scripts read those run directories.
8. `scripts/12_build_cpi_ppi_report.py` selects only the current estimation revision,
   intended observation design, unrestricted model, and minimum iteration count.
   `scripts/build_report.py` then sequences the remaining report artifacts.

This route is not identical to every exploratory script in the repository. In
particular, QCEW and SEC extensions run through `scripts/15_build_extension_data.py` and
`scripts/16_estimate_extensions.py`; they change both the competition variable and, for
SEC, the sample.

## Data audit

### Main model-ready frame

The saved frame has 454 unique, continuous quarters from 1913Q1 through 2026Q2 and no
duplicate dates or internal missing quarters. Rebuilding it in memory from the raw
sources reproduced every saved numeric value to tolerance `1e-12`.

| Estimation variable | Raw source and production column | Construction | Audit result |
|---|---|---|---|
| Headline CPI inflation | `raw/inflation/CPIAUCSL.csv`, `CPIAUCSL` | Monthly levels averaged by quarter, then `100*(q/q[-4]-1)` (`func_data_build.py:142-160,274-281`) | Numerically verified. Simple percentage change, not log change. |
| Core CPI inflation | `raw/inflation/CPILFESL.csv`, `CPILFESL` | Quarterly mean, then `100*(log q-log q[-4])` | Numerically verified. This differs from the headline/PPI transformation. |
| PPI inflation | `raw/inflation/PPIACO.csv`, `PPIACO` | Quarterly mean, then simple four-quarter percentage change | Numerically verified. Main-sample simple-versus-log difference reaches about 1.08 percentage points. |
| Expected inflation | `raw/inflation/Clev_Fed_Inflation_Expectation.csv`, raw ` Epi` | Multiply the decimal by 100, then quarterly mean (`func_data_build.py:163-168`) | Values match the official `EXPINF1YR` series. It is a model-based one-year expectation, not an external survey series. |
| Unemployment gap | `raw/unemp_gap/NROU.csv`, `NROU`; `UNRATE.csv`, `UNRATE` | Quarterly mean; `NROU-UNRATE` (`func_data_build.py:217-240`) | Sign is correct for tightness: positive in booms. `UNRATE` is seasonally adjusted. |
| BN output gap | `raw/output_gap/BN_filter_GDPC1_quaterly.csv`, `cycle` | Supplied BN cycle | Units are 100-log points and align with HP output. |
| HP output gap | same real-GDP source, original level | HP(1600) of `100*log(real GDP)` (`build.py:48-65`) | Independently reproduced against `statsmodels` to about `1e-10`. |
| HP labor-share gap | `raw/laborshare/PRS85006173.csv` | HP(1600) of `100*log(index)` (`build.py:72-117`) | Independently reproduced. |
| Inverse-markup gap | `raw/markup/BN_markup_inv.csv`, `cycle` | Supplied BN cycle (`func_data_build.py:202-214`) | Source/column wiring verified. |
| Main competition level | `raw/competition/BN_N_Gustavo_26.csv`, `original_series` | Annual value, PCHIP for the comparison frame; annual-Q4 observations in the main likelihood | Raw value equals `10000/HHI`, consistent with a 0--10,000 HHI convention. |

Official descriptions used to verify semantics: the [Cleveland Fed inflation-expectation
page](https://www.clevelandfed.org/indicators-and-data/inflation-expectations), the
[FRED EXPINF1YR series](https://fred.stlouisfed.org/data/EXPINF1YR), the BLS QCEW
[ownership codes](https://www.bls.gov/cew/classifications/ownerships/ownership-titles.htm),
[industry codes](https://www.bls.gov/cew/classifications/industry/industry-titles.htm),
and [open-data field descriptions](https://www.bls.gov/cew/additional-resources/open-data/csv-data-slices.htm),
and the SEC [Financial Statement Data Sets](https://www.sec.gov/file/financial-statement-data-sets).

### Data concerns

**D1 — expectation timing and interpretation. MODEL / IDENTIFICATION PROBLEM; DATA
PROVENANCE PROBLEM.** `Epi` is the within-quarter mean of a forward one-year,
model-based expectation. The left side is trailing four-quarter inflation, while the
code/report writes the regressor as `E_t pi_{t+1}`. These are not the same horizon.
Quarterly averaging also uses information arriving within the quarter. Historical
structural interpretations of `alpha` are affected. The report and config were corrected
to stop calling the series an SPF/external survey, but the horizon mismatch remains.

**D2 — NROU vintage. DATA PROBLEM (provenance, modest numerical size).** Current-vintage
FRED CPI, core CPI, PPI, unemployment, and expectations match the local files over the
main sample. Local NROU differs in every main-sample quarter from the current FRED
vintage: mean absolute difference 0.0213 percentage point, maximum 0.0419 in 2012Q4.
The repository records no vintage, so this cannot be distinguished from accidental
staleness. The magnitude is too small to explain the HSA convergence failure, but exact
historical replication depends on it.

**D3 — annual timing. MODEL / IDENTIFICATION PROBLEM.** The source is an annual
aggregate, not a literal Q4 snapshot. The production convention places one observation
in Q4 (`wrappers.py:324-375`); the corrected PCHIP knots place year `t` at year-end
(`func_data_build.py:41-84`). The code is internally consistent, but Q4 is an identifying
timing assumption. PCHIP is not an independent high-frequency proxy: it manufactures
within-year values from the same 31 annual observations.

**D4 — inflation robustness changes two inputs. DATA / MODEL PROBLEM.** Headline and
PPI use simple percentage changes while core uses log changes. A core-versus-PPI result
therefore changes the price index, the expectation match, and the transformation. It
cannot be attributed solely to the competition channel or price index.

### QCEW

`src/nkpc_hsa/dataprep/qcew.py:27-109` correctly selects `US000`, private ownership code
5, total-industry code 10, quarters 1--4, and the published `qtrly_estabs` count. The
1982Q1--2012Q4 series is complete (124 observations). It is a count of establishments,
not firms. The source is not seasonally adjusted; mean quarter-to-quarter 100-log changes
differ by 0.517 point across calendar quarters, and the average within-year Q4/Q1 increase
is 1.91%. The joint AR(2) state has no seasonal component, so it can interpret this
recurring pattern as an economic cycle. All 30 saved QCEW extension cells fail the joint
convergence rule; the primary run has `rho_N1 Rhat=1.124`, ESS 11, and state-path
`Rhat=1.135`, ESS 10. QCEW conclusions are not trustworthy.

### SEC inverse HHI

The normalization is correct: market shares sum within SIC3 markets, `hhi` is a fraction
in `(0,1]`, `hhi_10000=10000*hhi`, and effective firms is `1/hhi`
(`sec_hhi.py:446-455` and its validator). The proxy nevertheless covers public SEC
filers, treats SIC3 as a market, and collapses heterogeneous markets. It is not an
economy-wide firm count.

The empirical equation is linear in log inverse HHI. With expenditure weights, the
coherent aggregate is the revenue-weighted geometric mean `inv_hhi_logrevw`; the legacy
extension default and comparison script use the firm-count-weighted mean. These are
visibly different series. The selected corrected audit run therefore uses the log-
revenue-weighted aggregate.

**D5 — corrected SEC Q4 bug. DATA PROBLEM; IMPLEMENTATION PROBLEM.** The old extraction
collapsed revenue concepts before pairing annual and nine-month facts. Across the
2012--2013 raw archives, 143 of 9,356 available annual/nine-month pairs (1.53%) mismatched
concepts, and 128 of those entered a positive derived Q4 value. The fix retains one fact
per `(adsh,qtrs,tag)` (`sec_hhi.py:317-321`), chooses the latest filing while retaining
concepts (`393-405`), and merges annual/Q3 facts on `(cik,fy,tag)` (`413-427`). Rebuilding
changes all 56 headline quarterly observations: mean absolute change 0.01735 effective
firms; maximum 0.20022 in 2018Q4 (old 4.62738, corrected 4.42716). All historical SEC
posteriors may be affected and were invalidated by the new revision.

The SEC sample is 2012Q1--2025Q1 (`T=53`) because the extension deliberately removes the
main 2012Q4 endpoint. A result comparing main firms with SEC inverse HHI changes proxy,
sample length, macro regime, public-filer coverage, and market aggregation at once. It is
not a one-variable robustness comparison.

## Exact samples

Every one of the 11 production data specifications resolves to 124 complete quarters,
1982Q1--2012Q4, under the configured window (`configs/models.yaml:7-8,72-83`). Annual-Q4
HSA observes 31 firm-count values and has 93 missing firm-count rows; the inflation row
remains present every quarter. The PCHIP comparison treats all 124 derived quarterly
values as observed. The TNIC specifications would have only 97 observations and are not
in the production run list. QCEW has 124 complete quarters. Corrected SEC core CPI has 53.

## Equations reconstructed from code

Let `y_t = pi_t-Epi_t`, `a_t=pi_{t-1}-Epi_t`, and
`zeta_t=x_t-phi*x_{t-1}`. Stored kappa-family parameters are in physical units.

- CES: `y_t = alpha*a_t + kappa*x_t + lambda*zeta_t + eta_t`.
- HSA steady:
  `y_t = alpha*a_t + (kappa0+delta*Nbar_t)*x_t + lambda*zeta_t + eta_t`.
- HSA dynamic:
  `y_t = alpha*a_t + kappa*x_t - theta*Nhat_t + e_t`.
- HSA constant theta:
  `y_t = alpha*a_t + (kappa0+delta*Nbar_t)*x_t - theta*Nhat_t
  + lambda*zeta_t + eta_t`.
- HSA full:
  `y_t = alpha*a_t + (kappa0+delta*Nbar_t)*x_t
  -(theta0+gamma*Nbar_t)*Nhat_t + lambda*zeta_t + eta_t`.
- Competition measurement:
  `Nobs_t=Nbar_t+Nhat_t+nu_t` when observed.
- States:
  `Nhat_t=rho1*Nhat_{t-1}+rho2*Nhat_{t-2}+u_t` and
  `Nbar_t=Nbar_{t-1}+n+epsilon_t`.

The equation is empirically restricted so lagged and expected inflation weights are
exactly `alpha` and `1-alpha`. This vertical long-run normalization is imposed by code,
not derived from the HSA state decomposition, and should be stated as an identifying
restriction rather than a direct theory implication.

### Scaling

`KAPPA_SCALE=100` is correct and consistent. Kappa, kappa0, and delta priors/draws are
multiplied internally because regressors are `x/100` and `x*Nbar/100`, then divided by
100 for storage (for example `hsa_steady/model.py:703-742,958-983`). Theta, theta0, and
gamma use unscaled regressors (`-Nhat`, `-Nhat*Nbar`) and are not divided on storage
(`hsa_full_pg/model.py:452-479,550-570`). No double- or missing-scaling defect was found.

The transformed competition state is `(100*log N - sample center)/10`, so one state unit
is ten log points. This matters when reading delta and the state-innovation variances.

## State-space likelihood

For the three-state linear models the exact order is
`s_t=(Nhat_t,Nhat_{t-1},Nbar_t)`:

```
F = [[rho1,rho2,0], [1,0,0], [0,0,1]]
c = [0,0,n]
Q = diag(sigma_u^2,0,sigma_eps^2)
H_N = [1,0,1]
H_pi,t = [h_Nhat,t,0,h_Nbar,t]
R_t = diag(sigma_N^2,sigma_eta^2) when N is observed,
      [sigma_eta^2] otherwise.
m_0 = [0,0,0], P_0 = 10 I.
```

This matches `joint_ffbs.py:253-356`. Missing observations drop only `H_N`. State
ordering, variance versus standard-deviation use, and current-parameter loadings were
verified. A dense precision-matrix posterior independently matches FFBS means and
standard deviations for both complete and annual-Q4 patterns. The deterministic lag
coordinate is eigenvalue-clipped to `1e-10`; this is numerically negligible.

The initial `10 I` prior is implemented consistently but economically arbitrary. Its
sensitivity, especially for early state paths with only annual N observations, has not
been established and remains an open robustness requirement.

**S1 — corrected full-covariance smoother. IMPLEMENTATION PROBLEM.** In the dynamic
model's optional `covariance_structure="full"`, transition innovation `w_t` and
measurement error `v_t` are contemporaneously correlated. The forward filter included
the cross-covariance, but the old backward pass used the ordinary RTS kernel. Given
`s_{t+1}`, `y_{t+1}` still informs `s_t` through `Cov(w_{t+1},v_{t+1})`, so that kernel
did not draw from the smoothing posterior. The corrected code conditions jointly on
`(s_{t+1},y_{t+1})` (`hsa_dynamic/model.py:947-1048`) and matches an independently
assembled dense Gaussian posterior. Baseline `e_zeta_only` results were not affected
because their relevant cross-covariance is zero; historical opt-in full-covariance
results were affected.

## Gibbs and Particle-Gibbs audit

The Gaussian coefficient blocks use posterior covariance
`(X'X/sigma2+V0^-1)^-1` and the matching mean. The inverse-gamma helper draws
`1/Gamma(shape=a, scale=1/b)`, i.e. IG(shape, scale), and every variance update uses
`a+n/2`, `b+SSR/2`. Transition updates correctly use `T-1`; measurement updates use
only finite N rows. The inverse-Wishart update uses `nu+T` and `S+E'E`. Restricted
covariance updates sample directly on the restricted space rather than drawing full and
zeroing elements. AR(2) stationarity rejection leaves the current state unchanged after
exhausting proposals, a valid self-transition when the current point is stationary.

Derived `kappa_t` and `theta_t` are recomputed after the current state draw before
storage (for full PG, `hsa_full_pg/model.py:527-570`). No old-state storage bug was found.

Particle Gibbs pins the reference path in slot zero, preserves its ancestry, uses the
bootstrap state transition, weights both inflation and finite N observations including
the bilinear `-gamma*Nbar*Nhat` term, samples a terminal particle, and traces ancestry
(`hsa_full_pg/model.py:55-233`). At `gamma=0`, its invariant one-step distribution
matches the independent exact FFBS benchmark under annual missingness. It has no ancestor
sampling, so validity does not imply good mixing.

**G1 — lost PG diagnostics. IMPLEMENTATION / MCMC PROBLEM.** The sampler returned ESS
and moved-path diagnostics as a plain mapping, while the wrapper only extracted ordinary
`{"draws":...}` summaries. Historical NetCDFs therefore contain no PG path-degeneracy
evidence. `wrappers.py:392-410` now persists `pg_ess_mean`, `pg_ess_min`, and
`pg_moved_frac`.

The audit pilot shows why this matters:

| Particles | Mean conditional ESS | Mean minimum ESS | Mean moved fraction | State outcome |
|---:|---:|---:|---:|---|
| 128 | 98.9 | 1.03 | 0.249 | `Nhat` max Rhat 1.65, ESS 3.85 |
| 512 | 397.2 | 3.45 | 0.609 | `Nhat` max Rhat 1.40, ESS 4.58 |

More particles improve conditional path movement, but do not resolve the posterior's
AR/state multimodality. The 1,200-iteration runs are diagnostics only.

## Identification, priors, and posterior geometry

There is a near-ridge with a direct mathematical interpretation. For a constant shift
`c`, `Nbar+c`, `Nhat-c` preserves `Nobs`; changing `kappa0` to `kappa0-delta*c` also
preserves the time-varying slope. The zero-intercept AR(2) penalizes the shift by a term
proportional to `1-rho1-rho2`. When the AR sum approaches one, that penalty becomes weak.
Annual measurement supplies only 31 direct observations, leaving persistent and
near-alternating decompositions as separate posterior regions.

In the selected current-revision main HSA-steady run:

- chain means for `(rho1,rho2)` are `(0.180,-0.886)` and `(1.077,-0.116)`;
- scalar maxima are `Rhat=1.836`, minimum ESS 3;
- `Nbar` max Rhat is 1.254, minimum ESS 6.0;
- `Nhat` max Rhat is 1.828, minimum ESS 2.93;
- `kappa_t` max Rhat is 1.045, minimum ESS 39;
- `delta` is 0.0245, 95% interval `[0.0112,0.0405]`, but its apparently benign
  marginal trace coexists with the failed joint state posterior.

The trace/state figure `figures/audit_hsa_trace_geometry.png` displays the two regions.
This is a MODEL / IDENTIFICATION and MCMC / COMPUTATIONAL problem, not a genuine
economic cycle finding.

The saved historical annual-Q4 report grid confirms the problem: CES has max Rhat about
1.002 and minimum ESS 2,644, while every annual HSA steady, constant-theta, dynamic, and
full cell fails the joint `Rhat<=1.01` and ESS>=400 rule. Historical headline tables that
print scalar delta intervals but do not condition publication on state-path diagnostics
are unreliable.

Prior sensitivity is material. In the older core-CPI annual sweep, delta moves from
about 0.010 under the tight prior (interval crosses zero), to 0.026 baseline, to 0.034
under the weak prior; headline CPI shows the same pattern. Those files are from an older
revision and cannot be promoted to current results, but they demonstrate lack of prior
invariance. The weak IG state priors also have shape 1 and hence no finite prior mean.

The corrected SEC log-aggregate run converges numerically, but delta is 0.0016 with
95% interval `[-0.0362,0.0404]`, `P(delta>0)=0.528`, and posterior/prior SD ratio 0.974.
This is a likelihood-uninformative result, not evidence that the economic effect is
zero.

Synthetic recovery separates code from identification:

- With 240 quarters and a moderate signal, true delta 0.075 produced posterior mean
  0.089 and interval `[-0.008,0.191]`; posterior state means correlated 0.974 with true
  Nbar and 0.827 with true Nhat. The likelihood still supplied limited coefficient
  information.
- With 180 quarters and deliberately strong signal, true delta 0.22 was recovered at
  0.245, interval `[0.210,0.284]`, while true alpha and AR parameters were also covered.
  Yet kappa0 had `Rhat=2.19`, directly exposing the level-shift ridge. The estimator can
  recover delta when identification is strong; the production geometry, not a simple
  algebra error, is the limiting issue.

## Inflation residuals

Four-quarter inflation observed every quarter overlaps by three quarters. The i.i.d.
inflation likelihood is strongly contradicted by posterior-mean residual diagnostics:

| Current selected run | Residual ACF(1) | Ljung--Box Q(4) | p-value | Q(12) | p-value |
|---|---:|---:|---:|---:|---:|
| CES core | 0.375 | 33.08 | 1.15e-6 | 53.43 | 3.45e-7 |
| HSA steady core | 0.245 | 21.99 | 2.02e-4 | 41.13 | 4.66e-5 |

The MA(3) companion is a useful sensitivity because of the overlap, but overlap alone
does not prove that the hybrid-equation residual is exactly MA(3); that requires a model
of the underlying quarterly shock and aggregation. Existing MA3 results are one-chain,
900-iteration smoke tests from an older revision and cannot support conclusions. Delta
and theta sensitivity to a fully converged derived residual model remains unresolved.

## Reporting audit

All 138 base `results/runs` metadata files carry revision `2026-08-theta-centred`, not the
current audit revision. The current report builder correctly reports 138 missing required
runs and stops. The source report/PDF currently on disk is therefore historical output,
not a current validated report.

Additional stale-artifact paths existed:

- prior-decomposition CSVs carry revision `2026-08-unrate-sa-v1` or lacked embedded
  provenance;
- `conditional_ml.csv` had no revision/frequency/run-length fields;
- the predictive table could select/reformat stale output;
- `scripts/11_additional_report_evidence.py` did not require annual-Q4 and could select a
  newer PCHIP run by run-id ordering.

The consumers now require current revision/design/provenance
(`make_fit_comparison_table.py:34-44`, `make_conditional_ml_table.py:33-49`,
`prior_decomposition_rho_delta.py:180-208`), and additional evidence explicitly selects
annual-Q4. All three stale files now cause a nonzero exit rather than silently producing
LaTeX.

**R1 — invalid historical fit comparison. IMPLEMENTATION / REPORTING PROBLEM.** The old
predictive script loaded `DATE` as a normal column, bypassing the configured sample
window; used all 124 interpolated N values even for annual-Q4 rather than 31 finite
observations; initialized the state at `N[0]` instead of `m0=0,P0=10I`; omitted the
bilinear Taylor intercept; and omitted entry/activity-shock terms from the plug-in full
equation. The corrected construction is `predictive_comparison.py:85-117,135-199,202-245`.

Even corrected, parameter draws come from the full-sample posterior. The FF-LPD is not a
genuine prequential score, and applying WAIC/PSIS-LOO formulas to forward-filtering
conditional densities is not standard WAIC/LOO. The full model also uses an EKF
linearization rather than the exact nonlinear predictive density. The report now labels
these columns as in-sample heuristics (`make_fit_comparison_table.py:48-63`). Historical
claims that the score ordering was proper model-selection evidence are invalid.

## Finding disposition and historical impact

| ID | Classification | Fixed? | Historical estimates/results affected? |
|---|---|---|---|
| D1 expectation horizon/description | DATA; MODEL / IDENTIFICATION | Description fixed; horizon open | Alpha and structural NKPC interpretation, all models |
| D2 undocumented NROU vintage | DATA PROBLEM | Open | Small numerical effect on unemployment-gap cells |
| D3 annual-Q4 timing/PCHIP | MODEL / IDENTIFICATION | Code correct; assumption open | All main HSA states and PCHIP comparisons |
| D4 price-index transform confound | DATA / MODEL | Report clarified | CPI/PPI robustness attribution |
| D5 SEC concept mismatch | DATA; IMPLEMENTATION | Yes; data rebuilt | All old SEC estimates |
| S1 correlated-noise backward kernel | IMPLEMENTATION | Yes; dense test | Opt-in full-covariance dynamic runs only |
| G1 PG diagnostics discarded | IMPLEMENTATION; MCMC | Yes | Not posterior target; prevented auditing all old full runs |
| Main Nbar/Nhat ridge | MODEL / IDENTIFICATION; MCMC | No | Every HSA family and derived state path |
| Residual serial correlation | MODEL | No | Posterior uncertainty and potentially delta/theta means |
| QCEW seasonality/meaning | DATA; MODEL; MCMC | No | All QCEW extension claims |
| SEC prior dominance/sample confound | IDENTIFICATION | No | SEC interpretation and proxy comparisons |
| R1 predictive score/model mismatch | IMPLEMENTATION; REPORTING | Code/labels fixed; genuine OOS open | Entire historical fit-comparison table and prose |
| Stale auxiliary artifacts | IMPLEMENTATION; REPORTING | Consumers now fail closed | Prior-decomposition, Chib, fit-comparison report sections |

## What is verified and what can be trusted

### Verified

- Main raw-to-processed transformations, quarter alignment, lags, HP filters, and the
  configured 1982Q1--2012Q4 sample.
- Main HHI inverse normalization and the corrected SEC HHI normalization.
- Kappa-family versus theta-family scaling.
- Linear-Gaussian state ordering, `F,Q,H,R`, missing-row logic, and exact FFBS.
- Conjugate Gaussian, inverse-gamma, restricted covariance, and AR stationarity updates.
- Full-Sigma smoother after the audit fix.
- Conditional SMC mechanics and gamma-zero invariant-distribution comparison.
- CES MCMC numerical convergence for the selected cell.
- Corrected SEC log-aggregate HSA MCMC numerical convergence for its selected cell.

### Numerically converged but economically limited

- The selected CES posterior is computationally reliable conditional on its likelihood,
  but its i.i.d. inflation-residual assumption is rejected by residual diagnostics.
- The corrected SEC cell converges, but delta is prior-dominated and the proxy/sample are
  not comparable to the main historical firm-count design.

### Not trustworthy for posterior/economic conclusions

- Main annual-Q4 HSA steady state paths, `kappa_t`, and joint posterior.
- HSA constant-theta, dynamic, and full results in the historical annual grid.
- All QCEW extension results.
- Historical SEC results generated from the pre-fix Q4 data.
- Historical predictive/WAIC/LOO ordering, stale Chib table, and stale prior-decomposition
  prose.
- Claims that a positive marginal delta or a Bayes factor is an economic finding before
  joint convergence, residual specification, and prior sensitivity are resolved.

## Required next steps before a defensible production report

1. Reparameterize/identify the level decomposition explicitly—for example center Nbar
   over the sample or anchor its initial level, and parameterize the AR(2) in roots or
   persistence/frequency coordinates. Derive the implied prior before choosing one.
2. Run at least four dispersed chains and make state-path, kappa-path, and theta-path
   diagnostics report-blocking, not merely scalar diagnostics.
3. Derive a quarterly shock/aggregation model for overlapping four-quarter inflation,
   then estimate a converged residual-specification sensitivity.
4. Align the expectation horizon and information set with the inflation definition, or
   relabel the equation as semi-structural and test lagged/beginning-quarter timing.
5. Treat QCEW establishments as a distinct measurement with seasonal dynamics; do not
   call them firms.
6. Use the corrected SEC data and the log-revenue-weighted aggregate, while reporting
   public-filer/SIC3 limitations and never attributing its different sample to the proxy
   alone.
7. Recompute the full current-revision grid only after steps 1--4. Then rerun the Chib and
   prior-decomposition estimations, and use rolling re-estimation or exact K-fold/leave-
   future-out evaluation for predictive claims.

Until those steps are complete, the defensible conclusion is narrow: the repository
contains a correctly wired and largely correctly sampled family of models, but the main
HSA production posterior is not jointly identified/mixed well enough, and its residual
model is not adequate enough, to support the reported economic conclusions.
