# External active-firm stock from BDS levels and BED timing

## Frozen purpose

This test does not tune the effective-competition coordinate and does not test an
HSA cross-equation restriction. It asks, in order:

1. Can annual BDS firm levels and quarterly BED establishment-flow timing recover
   an externally anchored quarterly active-firm state without inflation feedback?
2. Given that state and the actual aggregate NKPC geometry, what is the minimum
   detectable free direct coefficient `theta_N`?
3. Only if recovery succeeds, what does the real-data free-`theta_N` fit say?

## Measurement distinction

BDS `FIRM` is the annual firm-count level and is associated with the March/Q1
reference period. BED births and deaths count establishments, not firms. The BED
series therefore never enters as an observed firm stock and is never cumulatively
equated to BDS firms.

Define the firm coordinate

```math
n_t=10\{\log N_t-\log N_{1993}^{BDS}\}.
```

At annual BDS dates,

```math
y^{BDS}_{y}=\bar n_{y,Q1}+\hat n_{y,Q1}.
```

The published BDS count is the level anchor, conditional on the published BDS
universe. The filter uses only a fixed `0.005` numerical-error scale; no free
annual measurement-error variance can absorb the level restriction.

Let `z_t^{BED}` be standardized quarterly establishment net entry. It is a noisy
timing measurement of the change in the latent firm stock:

```math
z_t^{BED}=a_E+\ell_E\Delta n_t+e_t^E,
\qquad e_t^E\sim N(0,\sigma_E^2).
```

The free loading `ell_E` and error variance absorb the establishment-to-firm
mapping. Its sign is not fixed. Before BED coverage this measurement row is
absent, so BED cannot manufacture timing.

The slow transition is

```math
\bar n_t=\bar n_{t-1}+\mu+\eta_t^{\bar n}.
```

The cycle is stochastic AR(2):

```math
\hat n_t=2r\cos(2\pi/P)\hat n_{t-1}-r^2\hat n_{t-2}
          +\eta_t^{\hat n}.
```

Innovation variances use

```math
\sigma_{\bar n}^2=\omega\tau^2,
\qquad
\sigma_{\hat n}^2=(1-\omega)\tau^2.
```

The entire state posterior is estimated from BDS/BED only. Inflation never
updates `nbar`, `nhat`, `omega`, `ell_E`, or any state-law parameter.

## First NKPC and recovery target

The only empirical NKPC estimated in this bundle is

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
      +\kappa_0x_t-\theta_N\hat n_t+\varepsilon_t,
```

with overlapping-inflation residual

```math
\varepsilon_t=u_t+\psi_1u_{t-1}+\psi_2u_{t-2}+\psi_3u_{t-3}.
```

There is no `delta`, `lambda`, time-varying `kappa`, or HSA restriction.
Competition-state draws are mixed into the NKPC sampler rather than replaced by
a posterior mean.

For each injected value in `{0, .05, .10, .20, .30, 1, 3, 10, 20, 30, 50}`, recovery uses the actual
sample dates, expectations, activity regressor, state path geometry, inflation
persistence, and MA(3) disturbance law. A replicate detects the effect only when
all three hold:

- the 95% interval excludes zero in the injected direction;
- `P(theta_N>0) >= .975`; and
- posterior/prior SD ratio is at most `.75`.

The wide upper grid is deliberate: the external BDS/BED decomposition may imply
a much smaller quarterly `nhat` scale than the effective-competition coordinate.
Two modes are always reported. `propagated_state` integrates over the external-N
posterior and is the admissible design for empirical inference. `oracle_state`
holds the injected `nhat` path known and diagnoses the regression/sample power
that would remain if the N state were measured perfectly.
The minimum detectable effect is the smallest injected value with at least 80%
replicate detection in the full profile. Mock and smoke recovery rates are
computational checks and are not inferential power estimates.
