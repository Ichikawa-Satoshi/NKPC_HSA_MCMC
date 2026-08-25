# Staged free-combined validation result

Status: **STOPPED BY PREDECLARED RECOVERY GATE - DYNAMIC HSA NOT RUN**

## Staged question

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+[\kappa_0+\delta\bar n_t^c]x_t-\theta_{CIQ}\hat n_t+\varepsilon_t.
```

Can `delta` and `theta_CIQ` be recovered jointly at empirically relevant standardized effects before imposing any HSA linkage?

## Promotion-gate result

| Parameter | Suggestive recovery | Required | Coverage | Null false positive |
|---|---:|---:|---:|---:|
| `delta` | 0.100 | 0.800 | 1.000 | 0.000 |
| `theta_CIQ` | 0.333 | 0.800 | 1.000 | 0.000 |

Recovery convergence: maximum R-hat `1.0395`, minimum bulk ESS `100.6`.

## Primary recovery by injected standardized effect

| Scenario | Parameter | Standardized true effect | Suggestive rate | Strong rate | Coverage |
|---|---|---:|---:|---:|---:|
| `null` | `delta` | 0.00 | 0.000 | 0.000 | 1.000 |
| `null` | `theta_CIQ` | 0.00 | 0.000 | 0.000 | 1.000 |
| `direct_observed` | `delta` | 0.00 | 0.000 | 0.000 | 1.000 |
| `direct_observed` | `theta_CIQ` | 0.11 | 0.500 | 0.033 | 1.000 |
| `slope_observed` | `delta` | 0.06 | 0.067 | 0.000 | 1.000 |
| `slope_observed` | `theta_CIQ` | 0.00 | 0.000 | 0.000 | 0.967 |
| `both_observed` | `delta` | 0.06 | 0.100 | 0.033 | 1.000 |
| `both_observed` | `theta_CIQ` | 0.11 | 0.333 | 0.067 | 1.000 |
| `both_moderate` | `delta` | 0.20 | 0.267 | 0.000 | 0.933 |
| `both_moderate` | `theta_CIQ` | 0.20 | 0.433 | 0.033 | 0.933 |
| `both_large` | `delta` | 0.40 | 0.767 | 0.033 | 0.800 |
| `both_large` | `theta_CIQ` | 0.40 | 0.833 | 0.233 | 0.700 |

## Direct-only versus free-combined model comparison

Positive ELPD differences favor free combined; `BF01 > 1` favors `delta=0`.

| Cycle | Activity | Delta LOO ELPD | Delta holdout ELPD | Delta holdout RMSE | BF01(delta=0) | Max Pareto k |
|---|---|---:|---:|---:|---:|---:|
| firm_weighted | ppi_inverse_markup | 0.482 | -0.232 | 0.071 | 1.531 | 1.550 |
| firm_weighted | ppi_negative_unemployment_gap | 0.723 | -0.046 | -0.128 | 1.406 | 1.581 |
| revenue_weighted | ppi_inverse_markup | 0.547 | -0.218 | 0.063 | 1.567 | 1.480 |
| revenue_weighted | ppi_negative_unemployment_gap | 0.013 | -0.241 | 0.057 | 1.377 | 1.398 |

All PSIS comparisons have influential observations (`Pareto k > 1`), so LOO is descriptive. Holdout ELPD is lower for free combined in all four cells. The Savage-Dickey diagnostic mildly favors `delta=0`.

## Decision

The gate failed (`delta`: 0.10, `theta_CIQ`: 0.333 versus 0.80 required). Oracle-state recovery is similarly weak, so state uncertainty is not the main bottleneck. Dynamic free and HSA-restricted models were not estimated. This prevents a restriction from manufacturing precision that the unrestricted channels do not possess.
