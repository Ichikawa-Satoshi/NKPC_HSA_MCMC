# Varying-theta and HSA-restricted dynamic diagnostic

Status: **COMPUTATIONAL PASS; DYNAMIC IDENTIFICATION FAIL; NOT FOR INFERENCE**

This branch was estimated at the user's request after the staged recovery gate had failed. It is a weak-identification diagnostic, not a reversal of that stopping decision.

## Models

```math
\theta_t=\theta_0+\gamma\bar n_t^c,
```

```math
\kappa_t^{free}=\kappa_0+\delta_1\bar n_t^c+\delta_2 q_t^{(2)},
\qquad q_t^{(2)}=(\bar n_t^c)^2-\overline{(\bar n^c)^2},
```

```math
\kappa_t^{HSA}=\kappa_0+\lambda\theta_0\bar n_t^c+\frac{\lambda\gamma}{2}q_t^{(2)}.
```

Thus the HSA restrictions are `delta_1=lambda*theta_0` and `delta_2=lambda*gamma/2`. Centering the quadratic term changes only the intercept of `kappa_t`; it does not change the derivative restriction.

The paired Gustavo slow-state and Capital IQ cycle draws are held fixed and cut from inflation in every model.

## Full-sample primary IID coefficients

Each entry is posterior mean followed by the 95% interval.

### `varying_theta`

| Parameter | Firm / unemployment | Revenue / unemployment | Firm / inverse markup | Revenue / inverse markup |
|---|---:|---:|---:|---:|
| theta_0 | 0.710<br>[-0.865, 2.262] | 0.627<br>[-0.830, 2.077] | 0.749<br>[-0.813, 2.283] | 0.666<br>[-0.754, 2.099] |
| gamma | -0.165<br>[-1.030, 0.707] | -0.126<br>[-0.953, 0.701] | -0.215<br>[-1.083, 0.645] | -0.170<br>[-0.960, 0.610] |

### `free_dynamic`

| Parameter | Firm / unemployment | Revenue / unemployment | Firm / inverse markup | Revenue / inverse markup |
|---|---:|---:|---:|---:|
| theta_0 | 0.894<br>[-0.845, 2.598] | 0.750<br>[-0.857, 2.327] | 0.613<br>[-0.993, 2.296] | 0.518<br>[-0.988, 2.040] |
| gamma | -0.241<br>[-1.257, 0.746] | -0.171<br>[-1.104, 0.766] | -0.191<br>[-1.064, 0.693] | -0.157<br>[-0.957, 0.663] |
| delta_1 | 0.102<br>[-0.615, 0.851] | 0.115<br>[-0.595, 0.838] | -0.076<br>[-3.463, 3.262] | -0.130<br>[-3.383, 3.238] |
| delta_2 | -0.177<br>[-1.156, 0.803] | -0.138<br>[-1.134, 0.830] | 0.813<br>[-1.530, 3.137] | 0.828<br>[-1.516, 3.191] |

### `hsa_restricted_dynamic`

| Parameter | Firm / unemployment | Revenue / unemployment | Firm / inverse markup | Revenue / inverse markup |
|---|---:|---:|---:|---:|
| theta_0 | 0.406<br>[-0.619, 2.035] | 0.334<br>[-0.620, 1.837] | 0.375<br>[-0.664, 1.832] | 0.332<br>[-0.653, 1.678] |
| gamma | -0.090<br>[-0.949, 0.721] | -0.084<br>[-0.927, 0.719] | -0.175<br>[-0.996, 0.668] | -0.170<br>[-0.922, 0.639] |
| lambda | 0.402<br>[-5.288, 5.952] | 0.523<br>[-5.307, 6.244] | -0.411<br>[-13.298, 12.196] | -0.518<br>[-12.947, 12.155] |
| delta_1 (derived) | 0.149<br>[-0.644, 1.033] | 0.132<br>[-0.648, 1.013] | -0.028<br>[-3.901, 3.895] | 0.022<br>[-3.796, 3.808] |
| delta_2 (derived) | -0.016<br>[-0.705, 0.615] | -0.010<br>[-0.691, 0.651] | 0.427<br>[-1.181, 3.065] | 0.396<br>[-1.204, 2.957] |

## Predictive comparison

Differences below are relative to the constant-theta model. Positive ELPD favors the dynamic model; negative RMSE favors it. PSIS-LOO is descriptive because every cell has at least one Pareto-k above 0.7.

| Cycle | Activity | Model | Delta LOO ELPD | Delta WAIC ELPD | Delta holdout ELPD | Delta holdout RMSE |
|---|---|---|---:|---:|---:|---:|
| firm_weighted | ppi_inverse_markup | varying_theta | -0.088 | -0.413 | -0.828 | -0.108 |
| firm_weighted | ppi_inverse_markup | free_dynamic | -0.532 | -0.531 | -1.988 | 0.108 |
| firm_weighted | ppi_inverse_markup | hsa_restricted_dynamic | -0.944 | -0.483 | -1.145 | -0.137 |
| firm_weighted | ppi_negative_unemployment_gap | varying_theta | 0.242 | -0.522 | -0.777 | 0.137 |
| firm_weighted | ppi_negative_unemployment_gap | free_dynamic | -1.034 | -1.752 | -1.407 | -0.111 |
| firm_weighted | ppi_negative_unemployment_gap | hsa_restricted_dynamic | -0.270 | -1.213 | -0.636 | -0.274 |
| revenue_weighted | ppi_inverse_markup | varying_theta | -0.031 | -0.146 | -1.341 | -0.084 |
| revenue_weighted | ppi_inverse_markup | free_dynamic | -0.135 | -0.661 | -2.444 | 0.082 |
| revenue_weighted | ppi_inverse_markup | hsa_restricted_dynamic | -0.925 | -0.560 | -1.603 | -0.133 |
| revenue_weighted | ppi_negative_unemployment_gap | varying_theta | -1.117 | -0.736 | -1.118 | 0.225 |
| revenue_weighted | ppi_negative_unemployment_gap | free_dynamic | -2.195 | -0.915 | -1.850 | 0.012 |
| revenue_weighted | ppi_negative_unemployment_gap | hsa_restricted_dynamic | -1.167 | -0.562 | -0.774 | -0.276 |

All twelve dynamic holdout-ELPD differences are negative. No dynamic specification improves held-out predictive density relative to constant theta.

## Varying-theta recovery

The primary propagated-state recovery uses 30 replications at each standardized gamma. Suggestive detection requires `P(gamma>0)>=0.80` and posterior/prior SD at most 0.75; strong detection additionally requires a positive 95% interval.

| Standardized gamma | Suggestive rate | Strong rate | Coverage |
|---:|---:|---:|---:|
| 0.00 | 0.000 | 0.000 | 1.000 |
| 0.05 | 0.367 | 0.000 | 1.000 |
| 0.10 | 0.533 | 0.067 | 1.000 |
| 0.20 | 0.667 | 0.167 | 1.000 |
| 0.40 | 1.000 | 0.800 | 0.967 |

Convergence passes: observed maximum R-hat `1.0011`, observed minimum bulk ESS `4074.5`, recovery maximum R-hat `1.0129`, and recovery minimum bulk ESS `498.5`.

## Persistent-AR(1) robustness

For the firm-weighted PPI x negative unemployment-gap cell, free-dynamic `gamma=-0.231 [-1.338,0.866]`; HSA-restricted `gamma=-0.096 [-1.113,0.847]` and `lambda=0.401 [-5.975,6.292]`. Allowing persistent inflation errors does not change the identification conclusion.

## Interpretation

The actual-data `theta_0` posterior remains directionally positive in the varying-theta model, but every 95% interval includes zero. Every `gamma` posterior also includes zero and leans negative. At an observed-scale standardized gamma of 0.10, propagated-state strong recovery is only 0.067. The sample therefore cannot distinguish a modest time-varying direct coefficient from a constant one.

The HSA restriction does not solve this problem. Every unrestricted `lambda` interval spans both signs and zero, and every derived slope interval spans zero. Because the unrestricted free-dynamic channels are themselves weak, posterior narrowing inside the HSA parameterization is not independent structural identification.

The dynamic branch is computationally valid but empirically unsupported. Retain the constant-theta direct-channel result only as suggestive directional evidence; do not claim time-varying theta or the HSA cross-equation restrictions are identified.
