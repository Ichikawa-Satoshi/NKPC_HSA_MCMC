# Gustavo state and Capital IQ cycle mock

## Question

Can the competition decomposition be stabilized by assigning distinct jobs to
the two effective-firm series: Gustavo determines the slow state, while Capital
IQ determines the quarterly cycle? This is a mock identification diagnostic, not
a production HSA estimate.

## Modular measurement order

Define the Gustavo coordinate

```math
g_y=10\{\log N_y^G-\log N_{1993}^G\}.
```

The quarterly slow state is a Gaussian bridge with exact Q4 endpoints:

```math
\bar n_t=\bar n_{t-1}+\mu+\eta_t^{\bar n},
\qquad
\bar n_{y,Q4}=g_y.
```

Gustavo alone determines `bar n`. Capital IQ and inflation cannot update it.
Conditional quarterly increments within each annual interval are drawn from the
Gaussian transition law subject to their sum equalling the observed annual
Gustavo change.

For Capital IQ series `j`, define its own-origin coordinate

```math
c_{j,t}=10\{\log N^{CIQ}_{j,t}-\log N^{CIQ}_{j,1993Q4}\}.
```

Capital IQ is then used only in the second, cut measurement block:

```math
c_{j,t}=a_j+b_j\bar n_t+\hat n_{j,t}+e_{j,t},
\qquad e_{j,t}\sim N(0,\sigma_{e,j}^2),
```

```math
\hat n_{j,t}=2r_j\cos(2\pi/P_j)\hat n_{j,t-1}
-r_j^2\hat n_{j,t-2}+\eta_{j,t},
\qquad \eta_{j,t}\sim N(0,\sigma_{h,j}^2).
```

The intercept and loading prevent the Capital IQ effective-firm level from being
equated mechanically with the Gustavo level. The likelihood integrates over a
fixed Monte Carlo sample from `p(bar n | Gustavo)`; there is no reverse feedback.
Both firm-weighted and revenue-weighted Capital IQ series are reported.

## Primary QoQ mock NKPC

The primary inflation observation is annualized quarter-on-quarter log inflation,

```math
\pi_t^q=400(\log P_t-\log P_{t-1}),
```

and the expectation is the genuine SPF one-quarter-ahead GDP-deflator forecast
transformed to the same annualized-log units. For each Capital IQ cycle and each
PPI activity cell, estimate

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+\kappa_0x_t-\theta_{CIQ}\hat n_{j,t}+\varepsilon_t,
```

with IID innovations as the primary law,

```math
\varepsilon_t=u_t,
```

and a persistent AR(1) robustness law,

```math
\varepsilon_t=\rho\varepsilon_{t-1}+u_t.
```

MA(3) is not used for QoQ inflation. The earlier overlapping-YoY run is retained
under `results/mock_yoy_legacy/` and is not the primary result.

This first stage contains no `delta`, structural `lambda`, HSA restriction, or
marginal-likelihood comparison. The free direct coefficient is named
`theta_CIQ`, not `theta_N`, because Capital IQ remains an effective-firm
concentration measure. Recovery for the firm-weighted primary cycle compares
propagated-state and oracle-known-cycle modes under both IID and AR(1) errors.

## Nested free-combined diagnostic

The next recorded stage holds the measurement posterior and every QoQ data
transformation fixed and adds only the slow-state slope interaction:

```math
\bar n^c_t=\bar n_t-\frac{1}{T}\sum_{s=1}^T\bar n_s,
```

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+\left(\kappa_0+\delta\bar n_t^c\right)x_t
-\theta_{CIQ}\hat n_{j,t}+\varepsilon_t.
```

Centering occurs inside each propagated slow-state draw. The Capital IQ cycle and
the paired Gustavo slow draw enter together. The competition measurement block
remains cut from inflation.

The priors for the original coefficients, including `theta_CIQ`, are unchanged.
The zero-mean Gaussian prior for `delta` is standardized so that one standard
deviation of `delta * nbar_centered * x` has the same configured prior effect
scale as the other competition coefficient. The run uses four chains, 2,000
iterations, 600 warmup iterations, and thinning by two.

Before estimation, `theta_CIQ` was classified as retained when its posterior
positive probability was at least `.70`, its posterior/prior SD ratio was at most
`.75`, and `abs(Corr(delta,theta_CIQ))` was below `.80`. These are descriptive
channel-separability thresholds, not structural identification thresholds.

The model estimates `delta` and `theta_CIQ` freely. It therefore asks whether the
slow slope and cyclical direct channels can coexist without the former explaining
away the latter. It does not impose

```math
\delta=\lambda\theta.
```

Moreover, if `lambda` is an unrestricted real parameter, this static equality is
only a reparameterization whenever `theta` is nonzero. A genuine static HSA test
therefore requires external discipline on `lambda`; alternatively, a dynamic HSA
model must supply overidentifying cross-equation restrictions.

## Staged recovery and promotion rule

Before estimating any dynamic restriction, simulate the free-combined model on
the actual sample and regressors:

```math
\pi_t^{sim}=a+\alpha_b\pi_{t-1}^{sim}+\alpha_fE_t\pi_{t+1}^q
+(\kappa_0+\delta^*\bar n_t^c)x_t-\theta^*\hat n_t+\varepsilon_t.
```

Effects are defined in comparable inflation-standard-deviation units:

```math
s_\theta=\frac{\theta SD(\hat n)}{SD(\pi)},
\qquad
s_\delta=\frac{\delta SD(\bar n^c x)}{SD(\pi)}.
```

The observed unemployment-gap magnitudes are approximated by
`s_theta=0.11` and `s_delta=0.06`. The primary firm-weighted IID recovery uses 30
replications in both propagated-state and oracle-state modes. Additional AR(1)
and inverse-markup checks use 10 replications. Null, one-channel, observed-size,
moderate, and large joint effects are recorded.

Suggestive detection requires sign probability at least `.80` and a
posterior/prior SD ratio at most `.75`. Strong detection additionally requires
sign probability `.975` and a 95% interval excluding zero. Promotion to dynamic
HSA requires, for both coefficients at the propagated observed-size scenario:

- suggestive recovery at least `.80`;
- interval coverage at least `.80`;
- null false-positive rate at most `.10`;
- maximum R-hat at most `1.10`; and
- minimum bulk ESS at least `100`.

The nested empirical comparison uses full-sample PSIS-LOO and manually computed
WAIC, a fixed 2010Q1-2013Q4 one-step-ahead holdout, and a Savage-Dickey density
ratio for `delta=0`. PSIS results with Pareto-k above `.70` are descriptive only.

If the recovery gate passes, the intended dynamic comparison is

```math
\theta_t=\theta_0+\gamma\bar n_t^c,
\qquad
\kappa_t=\kappa_0+\delta_1\bar n_t^c+\delta_2(\bar n_t^c)^2
```

against

```math
\delta_1=\lambda\theta_0,
\qquad
\delta_2=\frac{\lambda\gamma}{2}.
```

The gate did fail, so the automatic workflow stopped both dynamic models. The
user subsequently gave an explicit instruction to run the branch anyway. That
separate run is classified as a post-gate weak-identification diagnostic and
cannot be used to promote HSA merely because a restricted posterior is narrower.

## Explicitly authorized dynamic diagnostic

Define the within-draw centered slow state and centered quadratic term as

```math
\bar n_t^c=\bar n_t-T^{-1}\sum_s\bar n_s,
\qquad
q_t^{(2)}=(\bar n_t^c)^2-T^{-1}\sum_s(\bar n_s^c)^2.
```

Centering `q_t^(2)` changes only the constant in `kappa_t`; it leaves the HSA
derivative identity unchanged. The three estimated models are:

```math
\text{varying theta:}\quad
\theta_t=\theta_0+\gamma\bar n_t^c,
\qquad \kappa_t=\kappa_0,
```

```math
\text{free dynamic:}\quad
\theta_t=\theta_0+\gamma\bar n_t^c,
\qquad
\kappa_t=\kappa_0+\delta_1\bar n_t^c+\delta_2q_t^{(2)},
```

```math
\text{HSA-restricted dynamic:}\quad
\theta_t=\theta_0+\gamma\bar n_t^c,
\qquad
\kappa_t=\kappa_0+\lambda\theta_0\bar n_t^c
+\frac{\lambda\gamma}{2}q_t^{(2)}.
```

Hence the cross-equation restrictions tested are

```math
\delta_1=\lambda\theta_0,
\qquad
\delta_2=\frac{\lambda\gamma}{2}.
```

`lambda` has the sign-unrestricted prior `N(0,10^2)`. All other competition
priors use the same zero-mean standardized effect scale of `0.20` as the staged
workflow. Primary IID fits use 5,000 iterations, 1,500 warmup iterations,
thinning by two, and four chains. Firm-weighted PPI x negative unemployment gap
is also fit with a stationary persistent AR(1) error.

The varying-theta recovery fixes `s_theta=0.11` and injects standardized
`s_gamma` in `{0,0.05,0.10,0.20,0.40}`. Each scenario uses 30 replications in
both propagated-state and oracle-state modes. This recovery tests whether
`gamma` can be identified before interpreting the HSA product restrictions; it
does not by itself test recovery of `lambda`.

## Status rule

Every result from this folder is **mock / not for inference**. R-hat and ESS test
the implementation only. Economic identification additionally requires a
sign-consistent 95% interval, sign probability at least `.975`, and a
posterior/prior SD ratio at most `.75` in simulation recovery.
