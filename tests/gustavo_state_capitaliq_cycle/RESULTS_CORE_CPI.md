# Core CPI QoQ M0--M4 validation

Recorded: 2026-08-25  
Status: **computational pass; structural-sign and dynamic-HSA identification fail**

## Design

Core CPI is estimated separately from PPI on 1993Q2--2013Q4 (`T=83`). Monthly
FRED CPILFESL is averaged within quarter and transformed as
`400 * Delta log(P)`. The expectation is the Philadelphia Fed SPF CPI3
one-quarter-ahead headline-CPI forecast in annualized log units. It is an
explicit proxy for unavailable historical Core-CPI expectations.

The Gustavo slow-state and firm-/revenue-weighted Capital IQ AR(2) cycle draws
are reused byte-for-byte and remain cut from inflation. The run estimates M0
constant theta, M1 free static combined, M2 varying theta, M3 free dynamic, and
M4 HSA-restricted dynamic. Every model has full and pre-2010 training fits for
both activities and both Capital IQ cycles. The firm-weighted unemployment-gap
cell also has a persistent-AR(1) robustness fit.

## Observed coefficients

In M0, the Core-CPI direct loading leans negative in all four cells:

| Cycle / activity | theta_CIQ | P(theta>0) | Post/prior SD |
|---|---:|---:|---:|
| Firm / unemployment gap | -0.016 [-0.111, 0.078] | 0.378 | 0.464 |
| Revenue / unemployment gap | -0.016 [-0.101, 0.069] | 0.348 | 0.433 |
| Firm / inverse markup | -0.025 [-0.118, 0.067] | 0.300 | 0.449 |
| Revenue / inverse markup | -0.024 [-0.112, 0.063] | 0.294 | 0.442 |

M1 does not change that direction. In the firm-weighted unemployment-gap cell,
`delta=-0.016 [-0.067,0.037]` and
`theta_CIQ=-0.029 [-0.133,0.072]`. In the firm-weighted inverse-markup cell,
`delta=0.009 [-0.227,0.241]` and
`theta_CIQ=-0.026 [-0.124,0.077]`.

M2 produces the opposite time-variation direction from PPI. Firm-weighted Core
CPI x unemployment gap gives
`theta_0=-0.026 [-0.121,0.069]` and
`gamma=0.030 [-0.025,0.087]`, with `P(gamma>0)=0.860`. Revenue weighting gives
`gamma=0.025 [-0.027,0.078]`, with probability `0.829`. Every interval includes
zero.

All M3 target intervals include zero. Under M4, all lambda and derived slope
intervals also include zero. Firm-weighted unemployment gap gives
`lambda=0.564 [-8.287,9.120]`; firm-weighted inverse markup gives
`lambda=1.056 [-10.042,11.710]`.

## Recovery

The full run contains 480 static and 300 varying-theta recovery fits. At the
observed standardized static effects `(s_delta,s_theta)=(0.06,0.11)`, the
propagated-state suggestive recovery rates are `0.000` and `0.433`; oracle-state
rates are `0.067` and `0.500`. At `s_gamma=0.10`, propagated-state suggestive
recovery is `0.367` and strong recovery is `0.067`.

Recovery maximum R-hat is `1.0207` and minimum bulk ESS is `331.9`. The weak
recovery is therefore substantive rather than a simulation-convergence failure.

## Prediction

Relative to M0, M1 holdout ELPD changes range from `-0.037` to `0.321`, while
all M1 LOO and WAIC changes are negative. M2 has only a negligible positive
holdout difference (`0.005`) in one cell. M3 and M4 lose substantial holdout
ELPD in both unemployment-gap cells. No extension improves prediction
consistently across activity measures, cycle definitions, and scoring rules.

## Computational gate

- Observed fits: 45.
- Maximum observed R-hat: `1.0036`.
- Minimum observed bulk ESS: `1,220.2`.
- Recovery fits: 780.
- Measurement draws: identical hashes to the PPI run.
- Smoke run: completed separately and excluded from reported posteriors.

## Interpretation

Core CPI supplies real posterior learning but does not validate HSA. Its average
direct loading has the opposite direction from PPI, and its time-variation
coefficient leans in the opposite direction as well. Because all intervals span
zero and recovery is weak at observed effects, the cross-price difference is a
transportability warning, not evidence for a negative Core-CPI structural
coefficient or a positive time-varying HSA channel.

Reproduce the full run from the repository root:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/run_core_cpi_validation.py --mode full --workers 4
```
