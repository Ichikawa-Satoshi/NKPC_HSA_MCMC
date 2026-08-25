# QoQ oil-control extension: exact specification and results

## Status

This is a full inference run. It was executed after the current-plus-one-lag oil
specification was fixed. No lag search, sign restriction, or outcome-dependent
choice of oil transformation was performed.

- Revision: `gustavo-state-capitaliq-cycle-qoq-oil-control-v1`
- Sample: 1993Q2--2013Q4 (`T=83`)
- Observed fits: 90
- Recovery fits: 560
- Maximum observed R-hat: `1.0046`
- Minimum observed bulk ESS: `1106.3`
- Maximum recovery R-hat: `1.0180`
- Minimum recovery bulk ESS: `344.4`
- Computational gate: pass

Saved numerical results are under `results/oil_control_full/`.

## Prespecified control

The input is the repository's FRED-derived real WTI-to-CPI index,
`raw/others/WTISPLC_CPIAUCSL.csv`. Its arbitrary level normalization cancels in
the log difference. Define

\[
q_t^o=400\{\log R_t^o-\log R_{t-1}^o\}.
\]

Every M0--M4 inflation equation receives exactly two additional terms,

\[
\beta_{o,0}q_t^o+\beta_{o,1}q_{t-1}^o.
\]

Both oil coefficients have zero-mean Gaussian priors whose one-standard-
deviation regressor effect has prior SD equal to one inflation SD. The same oil
series, transformation, lags, and prior rule are used for PPI and Core CPI. The
competition measurement posterior remains cut from inflation.

The oil control is strongly associated with PPI inflation (`corr=0.761`) but
not Core-CPI inflation (`corr=0.031`). Its correlation with the firm-weighted
posterior-mean Capital-IQ cycle is only `-0.040`, with the Gustavo slow state
`-0.059`, and with the negative unemployment gap `0.019`. Therefore the
control mainly removes PPI input-cost variation rather than projecting out the
competition regressors themselves.

## PPI results

For the firm-weighted unemployment-gap M0 cell,

\[
\beta_{o,0}=0.112[0.091,0.133],\qquad
\beta_{o,1}=0.012[-0.023,0.045].
\]

The contemporaneous oil pass-through is tightly positive; the extra lag is not
separately identified. The direct competition loading changes from the no-oil
estimate

\[
\theta=0.674[-0.845,2.200],\quad P(\theta>0)=0.816,
\]

to

\[
\theta=0.691[-0.391,1.813],\quad P(\theta>0)=0.890.
\]

The posterior mean is essentially unchanged, while the posterior/prior SD ratio
falls from `0.549` to `0.403`. Oil therefore removes residual noise rather than
explaining away the positive direct-channel estimate. The 95% interval still
crosses zero, so this remains suggestive rather than strong evidence.

In M1, which lets the slope and direct channels coexist,

\[
\delta=0.072[-0.538,0.710],\qquad
\theta=0.775[-0.443,2.016],\quad P(\theta>0)=0.896.
\]

The no-oil counterparts were `delta=0.173[-0.517,0.861]` and
`theta=0.784[-0.832,2.388]`. Thus the direct loading survives the addition of
the slope channel and oil control, but the slope interaction remains weak and
moves closer to zero.

M2 gives

\[
\theta_0=0.731[-0.374,1.853],\qquad
\gamma=-0.098[-0.714,0.518].
\]

There is no evidence that the positive PPI direct loading varies with the slow
competition state. M3 remains weakly identified. In M4,

\[
\lambda=0.425[-4.024,5.513],
\]

and both HSA-derived slope coefficients cross zero. Oil control does not solve
the multiplicative identification problem in the restricted dynamic model.

The PPI conclusions are stable to a persistent-AR(1) residual. For M0,
`theta=0.690[-0.501,1.916]`; the contemporaneous oil coefficient remains
`0.112[0.089,0.134]`.

## Core-CPI results

For the firm-weighted unemployment-gap M0 cell,

\[
\beta_{o,0}=-0.0001[-0.0021,0.0020],\qquad
\beta_{o,1}=-0.0010[-0.0030,0.0010].
\]

Neither oil term is learned as an economically meaningful Core-CPI control.
The direct loading is

\[
\theta=-0.015[-0.115,0.078],\quad P(\theta>0)=0.392,
\]

against `-0.016[-0.111,0.078]` without oil. The Core-CPI competition results
are therefore unchanged. M2 still has a positive directional gamma,
`0.029[-0.027,0.086]` with `P(gamma>0)=0.844`, but the interval crosses zero.
The M4 lambda and derived slope coefficients remain unidentified.

## Prediction and fit

Relative to the otherwise identical no-oil model, PPI WAIC improves by about
38--39 ELPD units in every M0--M4 cell. PSIS-LOO points in the same direction,
but maximum Pareto-k values remain above one, so LOO magnitudes are not treated
as reliable. Holdout performance depends on the activity proxy:

- With inverse markup, oil improves holdout ELPD by `3.53` to `4.91` and lowers
  RMSE by `0.58` to `1.01` across M0--M4.
- With the unemployment gap, M0 improves holdout ELPD by `1.55` but raises RMSE
  by `0.21`; richer competition models are mixed and M3 deteriorates.

Core-CPI WAIC deteriorates by about `1.3`--`1.6` units. Holdout ELPD improves by
only `0.25`--`0.62`, while RMSE changes by less than `0.015`. This is too small
and internally mixed to justify oil as a useful Core-CPI control in this sample.

## Recovery interpretation

With propagated competition-state uncertainty and standardized observed-scale
effects `(delta,theta)=(0.06,0.11)`, suggestive recovery rates are:

| Outcome | delta | theta |
|---|---:|---:|
| PPI | 0.05 | 0.40 |
| Core CPI | 0.05 | 0.55 |

For a standardized gamma of `0.10`, propagated-state suggestive recovery is
`0.60` for PPI and `0.35` for Core CPI. These rates remain below a conventional
power target. The observed PPI posterior is more positive after oil control,
but the recovery exercise still says that aggregate quarterly data cannot
reliably distinguish effects of the observed structural magnitude across
repeated samples.

## Conclusion

Oil control materially improves the PPI inflation equation and narrows the PPI
direct-channel posterior without changing its mean. It therefore strengthens
the descriptive statement that the PPI direct loading is positive in this
sample. It does not make the 95% interval exclude zero, does not recover the
slope channel, does not support time variation, and does not identify lambda.
Core-CPI results are essentially unaffected. The admissible interpretation is
stronger residual control for PPI, not successful validation of the full HSA
restriction.

## Reproduction

```bash
PYTHONPATH=src:. python \
  tests/gustavo_state_capitaliq_cycle/run_oil_control_validation.py \
  --mode full --workers 4
```
