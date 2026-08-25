# Competition-linked structural slope change: recorded result

Profile: `full`  
Revision: `competition-slope-change-v1`  
Inferential status: **computational gate passed**

This file is generated from `manifest.json` and CSV results. Numerical values are not manually transcribed.

## Competition-only state

```math
c_t^{obs}=\bar c_t+\hat c_t,\qquad \sigma_{\bar c}^2=\omega\tau^2,\quad \sigma_{\hat c}^2=(1-\omega)\tau^2.
```

| Quantity | Posterior mean | 95% interval |
|---|---:|---:|
| `omega` | 0.0574 | [0.0021, 0.2728] |
| `tau` | 0.1354 | [0.1012, 0.1852] |
| `slow_innovation_variance` | 0.0011 | [0.0000, 0.0057] |
| `cycle_innovation_variance` | 0.0177 | [0.0092, 0.0325] |
| `damping_or_rho` | 0.4601 | [0.3184, 0.6569] |
| `cycle_period` | 9.4070 | [6.0801, 17.3188] |

### State-law and omega-prior sensitivity

| Variant | omega | Maximum R-hat | Minimum bulk ESS |
|---|---:|---:|---:|
| `ar1_baseline` | 0.038 [0.003, 0.130] | 1.036 | 64 |
| `ar2_omega_balanced` | 0.330 [0.003, 0.984] | 1.004 | 487 |
| `ar2_omega_uniform` | 0.191 [0.000, 0.996] | 1.008 | 321 |
| `ar2_baseline` | 0.057 [0.002, 0.273] | 1.001 | 5826 |

The baseline AR(2) sampler converges, but the slow/cycle variance allocation is not data-dominated: changing the omega prior materially changes its posterior. The short AR(1) sensitivity also fails the full convergence threshold and is not a competing headline estimate.

## Primary slope-only NKPC

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+(\kappa_0+\delta\bar c_t)x_t+\varepsilon_t,
\qquad \varepsilon_t=u_t+\psi_1u_{t-1}+\psi_2u_{t-2}+\psi_3u_{t-3}.
```

| Cell | delta | P(delta>0) | Post/prior SD | R-hat | Bulk ESS |
|---|---:|---:|---:|---:|---:|
| Core CPI x inverse markup | 0.008 [-0.083, 0.099] | 0.573 | 0.287 | 1.000 | 9068 |
| Core CPI x unemployment gap | 0.032 [-0.040, 0.098] | 0.813 | 0.449 | 1.000 | 8119 |
| PPI x inverse markup | 0.196 [-0.059, 0.442] | 0.936 | 0.792 | 1.000 | 8250 |
| PPI x unemployment gap | 0.020 [-0.123, 0.166] | 0.609 | 0.941 | 1.000 | 8644 |

## Main economic estimand

```math
\Delta\kappa_{comp}=\delta(\bar c_{t_1}-\bar c_{t_0}).
```

| Cell | Window | Delta kappa | P(Delta kappa>0) |
|---|---|---:|---:|
| PPI x inverse markup | full_sample | -0.450 [-1.032, 0.136] | 0.064 |
| PPI x inverse markup | post_1982 | -0.484 [-1.106, 0.148] | 0.064 |
| PPI x inverse markup | capital_iq_coverage | -0.936 [-2.125, 0.284] | 0.064 |
| PPI x unemployment gap | full_sample | -0.046 [-0.383, 0.283] | 0.391 |
| PPI x unemployment gap | post_1982 | -0.050 [-0.417, 0.306] | 0.391 |
| PPI x unemployment gap | capital_iq_coverage | -0.096 [-0.807, 0.592] | 0.391 |
| Core CPI x unemployment gap | full_sample | -0.073 [-0.228, 0.093] | 0.187 |
| Core CPI x unemployment gap | post_1982 | -0.078 [-0.246, 0.098] | 0.187 |
| Core CPI x unemployment gap | capital_iq_coverage | -0.152 [-0.471, 0.192] | 0.187 |
| Core CPI x inverse markup | full_sample | -0.018 [-0.229, 0.191] | 0.427 |
| Core CPI x inverse markup | post_1982 | -0.020 [-0.247, 0.207] | 0.427 |
| Core CPI x inverse markup | capital_iq_coverage | -0.038 [-0.476, 0.401] | 0.427 |

## Direct competition-index diagnostic

```math
\pi_t=\cdots+(\kappa_0+\delta\bar c_t)x_t-\theta_C\hat c_{t-j}+\varepsilon_t.
```

| Cell | Timing | theta_C | P(theta_C>0) | Post/prior SD |
|---|---|---:|---:|---:|
| PPI x inverse markup | current | -0.013 [-0.220, 0.187] | 0.450 | 0.986 |
| PPI x inverse markup | lag1 | 0.009 [-0.197, 0.213] | 0.538 | 0.988 |
| PPI x inverse markup | lead1 | 0.012 [-0.190, 0.220] | 0.543 | 0.993 |
| PPI x unemployment gap | current | -0.013 [-0.220, 0.189] | 0.458 | 1.000 |
| PPI x unemployment gap | lag1 | 0.008 [-0.198, 0.216] | 0.530 | 0.999 |
| PPI x unemployment gap | lead1 | 0.012 [-0.191, 0.217] | 0.549 | 0.990 |

## Computational gate

- Maximum primary R-hat: 1.0012 (required <= 1.01).
- Minimum primary bulk ESS: 4187.6 (required >= 800).
- Minimum primary tail ESS: 4984.2 (required >= 800).
- Exact-identity error: 2.220e-16.

## What this test shows

- The state decomposition is estimated without inflation feedback and state uncertainty is propagated rather than plugged in.
- The data can be evaluated directly in terms of historical competition-induced slope changes.
- Computation is stable, but no delta interval excludes zero and the omega allocation is prior-sensitive. The result is therefore suggestive at most, not a structurally identified competition effect.

## What this test does NOT show

- A zero-crossing theta_C interval does not imply theta_C=0.
- theta_C is not the active-firm coefficient theta_N.
- A positive delta does not establish full HSA or a causal competition-policy counterfactual.
- No free lambda or fixed-lambda HSA restriction is estimated here.
