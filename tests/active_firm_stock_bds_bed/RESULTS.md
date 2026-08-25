# BDS/BED external active-firm test: recorded result

Profile: `smoke`  
Revision: `active-firm-stock-bds-bed-v1`  
Status: **NOT FOR INFERENCE**

## State model

```math
y_y^{BDS}=\bar n_{y,Q1}+\hat n_{y,Q1},\qquad z_t^{BED}=a_E+\ell_E\Delta n_t+e_t^E.
```

| Parameter | Mean and 95% interval | R-hat | Bulk ESS |
|---|---:|---:|---:|
| `mu` | 0.025
[0.018, 0.031] | 1.002 | 996 |
| `tau` | 0.054
[0.039, 0.070] | 1.019 | 259 |
| `omega` | 0.694
[0.516, 0.855] | 1.023 | 115 |
| `damping` | 0.666
[0.398, 0.874] | 1.008 | 390 |
| `period` | 16.744
[8.381, 23.500] | 1.004 | 529 |
| `bed_intercept` | -0.197
[-0.385, -0.009] | 1.009 | 354 |
| `bed_loading` | 11.245
[5.658, 17.102] | 1.022 | 225 |
| `sigma_bed` | 0.833
[0.690, 0.988] | 1.021 | 308 |
| `slow_innovation_variance` | 0.002
[0.001, 0.003] | derived | derived |
| `cycle_innovation_variance` | 0.001
[0.000, 0.002] | derived | derived |

## Free theta_N real-data diagnostic

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}+\kappa_0x_t-\theta_N\hat n_t+\varepsilon_t.
```

| Cell | theta_N | P(theta_N>0) | Post/prior SD | R-hat |
|---|---:|---:|---:|---:|
| ppi_inverse_markup | -1.712
[-17.620, 13.798] | 0.418 | 0.628 | 1.001 |
| ppi_negative_unemployment_gap | -1.293
[-16.781, 15.563] | 0.425 | 0.649 | 1.000 |

## Recovery

| Mode | True theta_N | Replicates | Detection rate | Mean estimate | Mean sign probability | Mean post/prior SD |
|---|---:|---:|---:|---:|---:|---:|
| oracle_state | 0.00 | 10 | 0.000 | -0.292 | 0.468 | 0.467 |
| oracle_state | 0.05 | 10 | 0.000 | -2.313 | 0.393 | 0.496 |
| oracle_state | 0.10 | 10 | 0.000 | -3.215 | 0.388 | 0.461 |
| oracle_state | 0.20 | 10 | 0.000 | 1.209 | 0.570 | 0.486 |
| oracle_state | 0.30 | 10 | 0.100 | 2.502 | 0.558 | 0.417 |
| oracle_state | 1.00 | 10 | 0.000 | 2.616 | 0.638 | 0.386 |
| oracle_state | 3.00 | 10 | 0.000 | 1.826 | 0.580 | 0.406 |
| oracle_state | 10.00 | 10 | 0.300 | 9.286 | 0.818 | 0.495 |
| oracle_state | 20.00 | 10 | 0.300 | 12.788 | 0.890 | 0.456 |
| oracle_state | 30.00 | 10 | 0.700 | 26.174 | 0.984 | 0.371 |
| oracle_state | 50.00 | 10 | 1.000 | 40.806 | 0.999 | 0.390 |
| propagated_state | 0.00 | 10 | 0.000 | 1.170 | 0.553 | 0.633 |
| propagated_state | 0.05 | 10 | 0.000 | -1.236 | 0.433 | 0.572 |
| propagated_state | 0.10 | 10 | 0.000 | 0.113 | 0.506 | 0.604 |
| propagated_state | 0.20 | 10 | 0.000 | -0.718 | 0.453 | 0.599 |
| propagated_state | 0.30 | 10 | 0.000 | -0.378 | 0.492 | 0.603 |
| propagated_state | 1.00 | 10 | 0.000 | -0.216 | 0.494 | 0.609 |
| propagated_state | 3.00 | 10 | 0.000 | -1.216 | 0.451 | 0.626 |
| propagated_state | 10.00 | 10 | 0.000 | 0.940 | 0.540 | 0.605 |
| propagated_state | 20.00 | 10 | 0.000 | 2.342 | 0.604 | 0.620 |
| propagated_state | 30.00 | 10 | 0.000 | 3.999 | 0.655 | 0.582 |
| propagated_state | 50.00 | 10 | 0.000 | 4.185 | 0.656 | 0.557 |

## Interpretation

- Computational gate passed: `False`.
- Minimum detectable theta recorded by this profile: `None`.
- Mock and smoke recovery rates are not inferential power estimates.
- No delta, lambda, HSA restriction, or model-evidence comparison is estimated here.
