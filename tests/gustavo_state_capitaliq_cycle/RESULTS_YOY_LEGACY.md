# Gustavo state x Capital IQ cycle: recorded mock result

Status: **MOCK - NOT FOR INFERENCE**

## Measurement design

```math
\bar n_{y,Q4}=10\log(N_y^G/N_{1993}^G),
\qquad c_t^{CIQ}=a_C+b_C\bar n_t+\hat n_t^{CIQ}+e_t.
```

## State parameters

| Variant | Parameter | Mean and 95% interval |
|---|---|---:|
| gustavo | `mu` | -0.015 [-0.045, 0.015] |
| gustavo | `sigma_bar` | 0.179 [0.146, 0.222] |
| revenue_weighted | `intercept` | 3.442 [2.541, 4.122] |
| revenue_weighted | `loading` | 0.027 [-0.295, 0.330] |
| revenue_weighted | `damping` | 0.756 [0.600, 0.879] |
| revenue_weighted | `period` | 19.436 [14.752, 23.643] |
| revenue_weighted | `sigma_cycle` | 0.363 [0.212, 0.568] |
| revenue_weighted | `sigma_measurement` | 0.660 [0.528, 0.814] |
| firm_weighted | `intercept` | 2.860 [2.038, 3.704] |
| firm_weighted | `loading` | -0.383 [-0.761, -0.061] |
| firm_weighted | `damping` | 0.779 [0.617, 0.892] |
| firm_weighted | `period` | 20.204 [15.715, 23.735] |
| firm_weighted | `sigma_cycle` | 0.310 [0.181, 0.538] |
| firm_weighted | `sigma_measurement` | 0.462 [0.352, 0.576] |

## Free cycle coefficient

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+\kappa_0x_t-\theta_{CIQ}\hat n_t^{CIQ}+\varepsilon_t.
```

| Cycle | Cell | theta_CIQ | P(theta_CIQ>0) | Post/prior SD | R-hat |
|---|---|---:|---:|---:|---:|
| firm_weighted | ppi_inverse_markup | 0.247 [-0.776, 1.211] | 0.671 | 0.644 | 1.005 |
| firm_weighted | ppi_negative_unemployment_gap | 0.243 [-0.848, 1.246] | 0.682 | 0.669 | 0.999 |
| revenue_weighted | ppi_inverse_markup | 0.247 [-0.693, 1.170] | 0.703 | 0.647 | 1.006 |
| revenue_weighted | ppi_negative_unemployment_gap | 0.231 [-0.723, 1.172] | 0.694 | 0.643 | 1.000 |

## Recovery

| Mode | True theta_CIQ | Detection rate | Mean estimate |
|---|---:|---:|---:|
| oracle_state | 0.00 | 0.000 | 0.034 |
| oracle_state | 0.10 | 0.000 | -0.342 |
| oracle_state | 0.30 | 0.000 | 0.147 |
| oracle_state | 1.00 | 0.000 | 0.899 |
| oracle_state | 3.00 | 1.000 | 2.329 |
| oracle_state | 10.00 | 1.000 | 8.934 |
| propagated_state | 0.00 | 0.000 | 0.198 |
| propagated_state | 0.10 | 0.000 | 0.365 |
| propagated_state | 0.30 | 0.000 | 0.360 |
| propagated_state | 1.00 | 0.000 | 0.226 |
| propagated_state | 3.00 | 0.333 | 1.510 |
| propagated_state | 10.00 | 1.000 | 4.673 |

## Gate

- Maximum R-hat: `1.0936` (required <= `1.2`).
- Minimum bulk ESS: `21.9` (required >= `50.0`).
- Exact Gustavo anchor error: `9.99e-16`.
- Computational mock gate passed: `False`.
- Mock recovery rates are pipeline diagnostics, not power estimates.
- No delta, lambda, HSA restriction, marginal likelihood, or causal interpretation is estimated.
