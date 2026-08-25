# Recorded results

Recorded: 2026-08-25  
Status: screen/mock/selected quick diagnostics; **not for structural inference**.

## Frozen decision rule

The predeclared gates in `SPECIFICATION.md` require maximum R-hat at most 1.01,
minimum bulk and tail ESS at least 800, exact-identity error at most `1e-10`, a
95% interval excluding zero in the theory direction, theory-sign probability at
least 0.975, posterior/prior SD ratio at most 0.75, satisfactory design residual
variation, positive theta/kappa paths, and successful recovery/validation. Model
evidence is evaluated only after all preceding gates pass.

## 1. Deterministic and dynamic screens

`results/screen/manifest.json` records 1,296 rows and 1,296 successful likelihood
evaluations, but zero `screen_identified` rows and zero eligible joint-state
passes. BIC/AIC values in this stage are diagnostics, not formal evidence.

The dynamic screen has 192 rows. No free or restricted dynamic specification
passes both the theta-path and kappa-path requirements. The Q4-only non-overlap
screen also fails to produce a result that is stable between the discovery and
validation samples; the validation segment has very few annual observations.

The fixed-state free-channel diagnostic shows why the PPI/inverse-markup result
looked promising but is insufficient. The S0-current slope is
`delta = 0.548 [0.255, 0.841]`, while
`theta = 2.45 [-1.24, 6.14]`. Under S1-current,
`delta = 0.498 [0.212, 0.785]`, while
`theta = -1.24 [-5.27, 2.80]`. The slope signal does not identify the direct
channel and the latter is state-definition sensitive.

## 2. Leading exact-N MA(3) quick runs

The leading candidate is annual-allocation AR(2), PPI/inverse markup, with a free
static slope and direct channel.

| Quantity | Posterior summary | Gate reading |
|---|---:|---|
| `delta` | `0.190 [-0.061, 0.438]` | P(positive)=0.937; SD ratio=0.778; fails identification |
| `theta` | `-0.007 [-0.204, 0.197]` | P(positive)=0.471; SD ratio=0.971; no learning |
| `omega` | `0.0468 [0.0014, 0.2093]` | Slow share is much smaller than in the earlier AR(1) state |
| slow innovation variance | `0.00090` | Stabilized |
| cycle innovation variance | `0.01771` | Most innovation allocated to cycle |
| max R-hat | `1.0040` | Passes R-hat alone |
| min bulk ESS | `337.6` | Fails the 800 gate |
| min tail ESS | `426.8` | Fails the 800 gate |
| exact identity error | `2.22e-16` | Passes |
| P(kappa positive for >=95% dates) | `0.174` | Fails path-sign gate |

The fixed-`lambda=6` version gives
`theta = 0.0296 [-0.0095, 0.0678]`, P(positive)=0.929, and posterior/prior SD
ratio 0.777. Its maximum R-hat is 1.0124 and minimum state ESS is below 800.
It fails convergence, interval, sign-probability, shrinkage, and kappa-path gates.
The tighter theta is caused by the imposed `delta=6*theta` mapping; it is not a
separate direct-channel discovery.

## 3. QoQ mock

The exact-N annual-allocation AR(2) QoQ mock used all four declared empirical
cells. Free-channel theta estimates are:

| Cell | theta | P(positive) | Post/prior SD | Max R-hat |
|---|---:|---:|---:|---:|
| PPI x negative unemployment gap | `-0.006 [-0.179, 0.161]` | 0.490 | 0.952 | 1.056 |
| PPI x inverse markup | `-0.003 [-0.180, 0.181]` | 0.503 | 1.032 | 1.055 |
| Core CPI x negative unemployment gap | `-0.007 [-0.198, 0.175]` | 0.460 | 0.983 | 1.038 |
| Core CPI x inverse markup | `-0.008 [-0.201, 0.181]` | 0.463 | 1.024 | 1.051 |

This is a short mock and none of the four cells learns theta.

## 4. Simulation recovery

The mock recovery exercise covers all injected truths, but maximum R-hat is 1.067
and coverage alone is not identification. For true `theta=0.16`, the posterior is
`0.091 [-0.221, 0.408]`; the sign is not recovered. For true `delta=0.10`, the
posterior is `0.244 [0.004, 0.500]`. For true `omega=0.20`, it is
`0.132 [0.024, 0.419]`. The test shows that at the empirical sample length the
sampler has substantially less power for the direct channel than for the slow
slope.

## 5. Restriction and specification diagnostics

- Free lambda in the leading mock is about
  `-0.66 [-17.57, 17.89]`; it is not separately identified.
- Removing the intercept gives theta about `-0.023 [-0.238, 0.190]` for PPI/gap
  and `-0.018 [-0.227, 0.176]` for PPI/markup.
- Imposing `alpha_b+alpha_f=1` gives theta about
  `-0.010 [-0.183, 0.191]` for PPI/gap and
  `-0.009 [-0.222, 0.207]` for PPI/markup.
- The estimated YoY MA(3) coefficients are large and near the invertibility
  boundary, consistent with strong overlap. Replacing this with iid or AR(1)
  residuals would be a misspecification, not an identification remedy.
- `pytest -q tests/hsa_deep_identification/test_joint_ma3.py` returns two passing
  dense-Gaussian FFBS validation tests.

## 6. Final gate table

| Gate | Best observed result | Decision |
|---|---|---|
| Exact `N=Nbar+Nhat` accounting | Error `2.22e-16` | Pass |
| Slow/cycle variance stabilization | `omega` mean about 0.047 | Pass as a design improvement |
| Sampler implementation validation | 2 FFBS tests pass | Pass |
| Full MCMC convergence | State ESS below 800; fixed-HSA R-hat 1.012 | Fail |
| Free `theta` identification | Interval includes zero; SD ratio 0.971 | Fail |
| Free `delta` identification | Interval includes zero; P+=0.937 | Fail |
| Free `lambda` identification | Prior-wide interval | Fail |
| Positive theta and kappa paths | Kappa path probability 0.174 in leading free run | Fail |
| Discovery/validation stability | No stable candidate | Fail |
| Simulation recovery | Direct-channel sign not recovered; mock nonconvergence | Fail |
| Formal HSA vs CES marginal evidence | Not run because prior gates failed | Not eligible |

## Conclusion

The test succeeded in diagnosing and improving the competition-state law: annual
allocation plus an AR(2) cycle avoids the earlier excessive slow innovation
variance. It did not solve the economic identification problem. The data and
current design do not separately identify the free direct channel or the HSA
factorization. No candidate is promoted to a full run, and no claim that HSA fits
better than CES is supported by this folder.
