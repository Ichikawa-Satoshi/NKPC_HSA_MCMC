# Competition-linked structural slope change

## Research question

Does a competition state constructed without inflation feedback explain a
historical change in the structural NKPC slope? The main estimand is

```math
\Delta\kappa_{\mathrm{comp}}(t_0,t_1)
=\delta\left(\bar c_{t_1}-\bar c_{t_0}\right),
```

not the direct HSA coefficient or the free proportionality coefficient.

## Competition coordinate and allocation

The empirical competition coordinate is

```math
c_t^{obs}=10\left(\log C_t-\log C_{ref}\right).
```

`C_ref` is fixed to the 1984 Gustavo value before estimation. It is a coordinate
origin, not a theoretical steady state. Annual Gustavo changes are distributed
using Capital IQ quarterly movements where usable and a robust average quarterly
profile otherwise. Extreme annual ratios are coherence-shrunk using competition
data only.

## Competition-only state block

```math
c_t^{obs}=\bar c_t+\hat c_t
```

with no additional observation error. The slow transition mean uses the external
average annual allocation:

```math
\bar c_t=\bar c_{t-1}+m_{q(t)}\Delta g_{y(t)}+\eta_t^{\bar c}.
```

There is no within-year zero-sum restriction on the slow innovation. The primary
cycle is

```math
\hat c_t=2r\cos(2\pi/P)\hat c_{t-1}-r^2\hat c_{t-2}
+\eta_t^{\hat c},\qquad 0<r<1,
```

and AR(1) is retained as a state-law robustness check. Innovation variances are

```math
\sigma_{\bar c}^2=\omega\tau^2,
\qquad
\sigma_{\hat c}^2=(1-\omega)\tau^2.
```

The state block never uses inflation.

## Primary slope-only NKPC

For every saved competition-state draw, the primary model is

```math
\kappa_t=\kappa_0+\delta\bar c_t,
```

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
+\kappa_t x_t+\varepsilon_t.
```

There is no independent random-walk innovation in `kappa_t` and no standalone
competition-level control. State uncertainty is propagated by drawing from the
competition-only posterior inside the NKPC mixture sampler; a posterior-mean
plug-in is not used.

Overlapping YoY inflation uses

```math
\varepsilon_t=u_t+\psi_1u_{t-1}+\psi_2u_{t-2}+\psi_3u_{t-3},
\qquad u_t\sim N(0,\sigma_u^2).
```

## Direct-channel diagnostic

The diagnostic extension is

```math
\pi_t=\cdots+(\kappa_0+\delta\bar c_t)x_t
-\theta_C\hat c_{t-j}+\varepsilon_t,
```

where `j=0` is the benchmark, `j=1` is timing robustness, and a one-quarter lead
is a placebo. The coefficient is called `theta_C`: it loads on a cyclical
competition index and is not the structural active-firm coefficient `theta_N`.

## Interpretation gates

Convergence, parameter learning, and economic signs are separate. A full result
requires maximum R-hat at most 1.01 and bulk/tail ESS at least 800. Structural
learning additionally requires a 95% interval excluding zero in the declared
direction, sign probability at least 0.975, and posterior/prior SD ratio at most
0.75. A fixed-restriction coefficient is not part of this test.

