# Identification-first HSA NKPC search protocol

## 1. Objective and non-negotiable rules

The objective is to determine whether an economically admissible HSA NKPC can
simultaneously deliver identified parameters, theory-consistent unrestricted
posterior signs, reliable MCMC convergence, and greater integrated model evidence
than CES. The data are allowed to reject this conjunction.

The following practices are prohibited:

- imposing a sign restriction and then presenting the resulting sign as data evidence;
- using an i.i.d. error for overlapping year-over-year inflation;
- selecting sample endpoints or priors after observing a preferred sign;
- reporting a plug-in, harmonic-mean, cut-state Laplace, or unstable bridge number
  as a formal marginal likelihood;
- allowing inflation to rewrite Gustavo anchors or Capital IQ allocation weights;
- promoting a model that fails simulation recovery, convergence, or identification.

## 2. Data held fixed

Competition is the Gustavo annual effective-firm count allocated within the year
using Capital IQ quarterly information. Every sampled quarterly path matches the
Gustavo Q4 benchmark exactly. The allocation posterior is measurement-only.

The four confirmatory empirical cells are:

1. PPI x negative unemployment gap;
2. PPI x inverse markup;
3. Core CPI x negative unemployment gap;
4. Core CPI x inverse markup.

PPI and Core CPI are never pooled. The full Gustavo endpoint-matched sample is
1974Q4--2013Q4. Alternative start dates are diagnostic stability checks only and
cannot define the winner.

## 3. Common structural equations

The exact competition identity is

```math
q_t=\bar q_t+\hat q_t.
```

CES is

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
+\kappa_0x_t+\varepsilon_t.
```

The static HSA candidate is

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
+\left(\kappa_0+\lambda\theta\bar q_t\right)x_t
-\theta\hat q_{t-j}+\varepsilon_t,
```

where `j` is either zero or one and is compared as a declared timing sensitivity.
The confirmatory calibration grid is `lambda in {3,6,9}`. Free lambda is an
identification diagnostic, not a confirmatory model, because the likelihood is
locally singular near `theta=0`.

## 4. Admissible competition-state candidates

### S0: quarterly local level plus AR(2) cycle

```math
\bar q_t=\bar q_{t-1}+\eta^b_t,
```

```math
\hat q_t=2r\cos(2\pi/P)\hat q_{t-1}-r^2\hat q_{t-2}+\eta^h_t.
```

This is the current exact-N benchmark.

### S1: annual-allocation slow innovation plus AR(2) cycle

For year `y`, let `m_q` be the measurement-only average quarterly allocation
profile and let `g_y` be the annual slow innovation:

```math
\bar q_{yq}-\bar q_{y,q-1}=m_q g_y+\eta^{b,small}_{yq},
```

The quarterly deviations are mean zero and their variance is estimated. They
are **not** constrained to sum to zero within a year. Such a constraint, when
combined with the exact Gustavo Q4 total, would mechanically impose an equal
Q4 cycle state in adjacent years and would therefore manufacture the split.
Gustavo constrains total `q`; it does not separately observe `bar q`. This law
separates the externally measured average annual allocation from deviations in
the Capital IQ quarterly allocation without pretending that the slow state is
observed.

### S2: local-linear trend plus AR(2) cycle

```math
\bar q_t=\bar q_{t-1}+d_{t-1}+\eta^b_t,
\qquad d_t=d_{t-1}+\eta^d_t.
```

This standard smooth-trend alternative prevents a local-level random walk from
absorbing every quarterly movement. Its additional variance must pass recovery
and posterior-learning gates.

All candidates retain exact `q=bar q+hat q`. Deterministic HP, band-pass, or EWMA
states may be used only as transparent screening diagnostics, never as the final
joint posterior.

## 5. Inflation frequency and disturbance candidates

- Year-over-year inflation uses an invertible MA(3) disturbance because four
  overlapping quarterly price changes enter each observation.
- Quarter-over-quarter annualized inflation uses a non-overlapping quarterly
  likelihood. I.i.d. and AR(1) errors may be compared; AR(1) is retained when its
  persistence is learned or predictive diagnostics require it.
- A year-over-year i.i.d. likelihood is an intentionally rejected placebo.

QoQ expectations must use the one-quarter-ahead annualized-log SPF series. A
frequency-mismatched expectation series is not admissible.

## 6. Frozen gates

### MCMC convergence

- rank-normalized maximum R-hat <= 1.01;
- minimum bulk ESS >= 800 and minimum tail ESS >= 800;
- no state or coefficient chain isolated in a separate mode;
- exact identity error <= 1e-10.

### Identification

For each structural coefficient that the model claims to estimate:

- central 95% interval excludes zero in the theory direction;
- unrestricted posterior theory-sign probability >= 0.975;
- posterior SD / prior SD <= 0.75;
- design condition number <= 30 after prior-scale standardization;
- the direct HSA regressor retains at least 20% residual variation after
  projection on the remaining NKPC regressors.

For state parameters, the posterior interval must not cover more than 90% of a
bounded prior support, and simulation recovery must cover the truth without a
boundary pile-up.

The intercept is not assigned a theory sign. Positive variance parameters are
not counted as sign evidence merely because their support is positive.

### Economic signs

The unrestricted posterior must support

```math
\alpha_b>0,\qquad \alpha_f>0,\qquad \theta>0,
\qquad \kappa_t=\kappa_0+\lambda\theta\bar q_t>0
```

over at least 95% of sample dates with posterior probability at least 0.95.
For a free-lambda diagnostic, `lambda>0` must independently pass the same
identification and sign gates; otherwise lambda is reported as unidentified.

### Model evidence

HSA is favored over CES only if all earlier gates pass and:

- formal log Bayes factor `log m(HSA)-log m(CES) > log(3)`;
- repeated bridge/Chib calculations differ by <= 0.25 log units;
- annual-origin predictive ELPD improves by more than two standard errors.

Marginal likelihood comparisons use the same cell, dates, inflation frequency,
allocation distribution, state priors, and compatible coefficient priors.

## 7. Search discipline

Candidate screening uses simulation recovery and the 1974Q4--1999Q4 discovery
subsample. A candidate architecture is frozen before inspecting 2000Q1--2013Q4
validation performance. The final full-sample estimate is reported only after
this validation. All screened candidates, including failures, remain in the
manifest.
