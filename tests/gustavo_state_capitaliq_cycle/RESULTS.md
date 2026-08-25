# Gustavo state x Capital IQ cycle: recorded QoQ mock result

Inflation: `400 * quarterly log difference`; expectation: genuine SPF one-quarter-ahead annualized-log forecast.  
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
| firm_weighted | `intercept` | 2.860 [2.038, 3.704] |
| firm_weighted | `loading` | -0.383 [-0.761, -0.061] |
| firm_weighted | `damping` | 0.779 [0.617, 0.892] |
| firm_weighted | `period` | 20.204 [15.715, 23.735] |
| firm_weighted | `sigma_cycle` | 0.310 [0.181, 0.538] |
| firm_weighted | `sigma_measurement` | 0.462 [0.352, 0.576] |
| revenue_weighted | `intercept` | 3.442 [2.541, 4.122] |
| revenue_weighted | `loading` | 0.027 [-0.295, 0.330] |
| revenue_weighted | `damping` | 0.756 [0.600, 0.879] |
| revenue_weighted | `period` | 19.436 [14.752, 23.643] |
| revenue_weighted | `sigma_cycle` | 0.363 [0.212, 0.568] |
| revenue_weighted | `sigma_measurement` | 0.660 [0.528, 0.814] |

## Free cycle coefficient

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+\kappa_0x_t-\theta_{CIQ}\hat n_t^{CIQ}+\varepsilon_t.
```

| Cycle | Cell | theta_CIQ | P(theta_CIQ>0) | Post/prior SD | R-hat |
|---|---|---:|---:|---:|---:|
| firm_weighted / iid | ppi_inverse_markup | 0.718 [-0.766, 2.169] | 0.837 | 0.540 | 1.001 |
| firm_weighted / iid | ppi_negative_unemployment_gap | 0.637 [-0.840, 2.186] | 0.788 | 0.566 | 1.003 |
| firm_weighted / persistent_ar1 | ppi_inverse_markup | 0.665 [-1.165, 2.464] | 0.771 | 0.669 | 0.999 |
| firm_weighted / persistent_ar1 | ppi_negative_unemployment_gap | 0.651 [-1.144, 2.321] | 0.772 | 0.638 | 1.002 |
| revenue_weighted / iid | ppi_inverse_markup | 0.663 [-0.820, 2.083] | 0.823 | 0.566 | 0.999 |
| revenue_weighted / iid | ppi_negative_unemployment_gap | 0.604 [-0.683, 2.040] | 0.800 | 0.533 | 1.004 |
| revenue_weighted / persistent_ar1 | ppi_inverse_markup | 0.676 [-0.920, 2.300] | 0.792 | 0.610 | 1.004 |
| revenue_weighted / persistent_ar1 | ppi_negative_unemployment_gap | 0.637 [-0.950, 2.266] | 0.783 | 0.604 | 1.013 |

## Recovery

| Error | Mode | True theta_CIQ | Detection rate | Mean estimate |
|---|---|---:|---:|---:|
| iid | oracle_state | 0.00 | 0.000 | 0.397 |
| iid | oracle_state | 0.10 | 0.000 | -0.524 |
| iid | oracle_state | 0.30 | 0.000 | -0.059 |
| iid | oracle_state | 1.00 | 0.333 | 0.979 |
| iid | oracle_state | 3.00 | 1.000 | 2.285 |
| iid | oracle_state | 10.00 | 1.000 | 7.538 |
| iid | propagated_state | 0.00 | 0.000 | -0.409 |
| iid | propagated_state | 0.10 | 0.000 | 0.612 |
| iid | propagated_state | 0.30 | 0.000 | 0.582 |
| iid | propagated_state | 1.00 | 0.000 | 0.853 |
| iid | propagated_state | 3.00 | 0.333 | 1.890 |
| iid | propagated_state | 10.00 | 1.000 | 6.854 |
| persistent_ar1 | oracle_state | 0.00 | 0.000 | -0.650 |
| persistent_ar1 | oracle_state | 0.10 | 0.000 | 0.144 |
| persistent_ar1 | oracle_state | 0.30 | 0.000 | 1.150 |
| persistent_ar1 | oracle_state | 1.00 | 0.000 | 0.944 |
| persistent_ar1 | oracle_state | 3.00 | 0.333 | 1.633 |
| persistent_ar1 | oracle_state | 10.00 | 1.000 | 6.850 |
| persistent_ar1 | propagated_state | 0.00 | 0.000 | -0.093 |
| persistent_ar1 | propagated_state | 0.10 | 0.000 | 0.243 |
| persistent_ar1 | propagated_state | 0.30 | 0.000 | 0.177 |
| persistent_ar1 | propagated_state | 1.00 | 0.000 | 0.786 |
| persistent_ar1 | propagated_state | 3.00 | 0.333 | 1.759 |
| persistent_ar1 | propagated_state | 10.00 | 1.000 | 5.405 |

## Gate

- Maximum R-hat: `1.0698` (required <= `1.2`).
- Minimum bulk ESS: `20.5` (required >= `50.0`).
- Exact Gustavo anchor error: `9.99e-16`.
- Computational mock gate passed: `False`.
- Mock recovery rates are pipeline diagnostics, not power estimates.
- No delta, lambda, HSA restriction, marginal likelihood, or causal interpretation is estimated.
