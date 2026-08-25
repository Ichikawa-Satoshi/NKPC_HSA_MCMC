# QoQ free-combined slope and direct-channel diagnostic

Status: **MOCK DIAGNOSTIC - NOT A STRUCTURAL HSA ESTIMATE**

## Estimated equation

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+\left[\kappa_0+\delta(\bar n_t-\overline{\bar n})\right]x_t
-\theta_{CIQ}\hat n_{j,t}+\varepsilon_t.
```

The saved Gustavo slow-state and Capital IQ AR(2) cycle draws are reused without inflation feedback. `delta` and `theta_CIQ` are free; no relation `delta=lambda*theta_CIQ` is imposed.

## Direct-only versus free combined

| Cycle | Error | PPI activity | Direct theta_CIQ | Combined theta_CIQ | P(theta>0) | delta | P(delta>0) | Corr(delta,theta) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| firm weighted | iid | inverse markup | 0.718 [-0.766, 2.169] | 0.653 [-0.900, 2.150] | 0.797 | 0.259 [-2.850, 3.220] | 0.572 | -0.095 |
| firm weighted | iid | negative unemployment gap | 0.637 [-0.840, 2.186] | 0.781 [-0.800, 2.346] | 0.829 | 0.166 [-0.520, 0.845] | 0.673 | 0.352 |
| firm weighted | persistent ar1 | inverse markup | 0.665 [-1.165, 2.464] | 0.686 [-1.039, 2.377] | 0.774 | 0.234 [-3.193, 3.653] | 0.550 | -0.105 |
| firm weighted | persistent ar1 | negative unemployment gap | 0.651 [-1.144, 2.321] | 0.763 [-1.126, 2.506] | 0.802 | 0.119 [-0.642, 0.867] | 0.626 | 0.259 |
| revenue weighted | iid | inverse markup | 0.663 [-0.820, 2.083] | 0.650 [-0.830, 2.136] | 0.819 | 0.284 [-2.684, 3.235] | 0.570 | -0.118 |
| revenue weighted | iid | negative unemployment gap | 0.604 [-0.683, 2.040] | 0.729 [-0.810, 2.197] | 0.829 | 0.164 [-0.514, 0.841] | 0.678 | 0.346 |
| revenue weighted | persistent ar1 | inverse markup | 0.676 [-0.920, 2.300] | 0.649 [-1.044, 2.306] | 0.782 | 0.245 [-3.146, 3.547] | 0.561 | -0.073 |
| revenue weighted | persistent ar1 | negative unemployment gap | 0.637 [-0.950, 2.266] | 0.702 [-1.097, 2.444] | 0.794 | 0.107 [-0.669, 0.866] | 0.609 | 0.278 |

## Complete coefficient table: primary IID

Each cell reports posterior mean and 95% equal-tail interval.

| Parameter | Firm / inverse markup | Firm / unemployment gap | Revenue / inverse markup | Revenue / unemployment gap |
|---|---:|---:|---:|---:|
| intercept | -1.983 [-16.634, 13.763] | 0.741 [-2.146, 3.610] | -2.573 [-16.911, 12.480] | 0.656 [-2.282, 3.637] |
| alpha_b | 0.337 [0.131, 0.539] | 0.337 [0.141, 0.535] | 0.334 [0.128, 0.538] | 0.340 [0.139, 0.541] |
| alpha_f | 0.422 [-0.495, 1.412] | 0.426 [-0.537, 1.378] | 0.416 [-0.536, 1.390] | 0.427 [-0.557, 1.371] |
| kappa_0 | -8.697 [-53.489, 36.889] | 0.188 [-1.101, 1.453] | -10.483 [-53.212, 34.500] | 0.117 [-1.067, 1.335] |
| delta | 0.259 [-2.850, 3.220] | 0.166 [-0.520, 0.845] | 0.284 [-2.684, 3.235] | 0.164 [-0.514, 0.841] |
| theta_CIQ | 0.653 [-0.900, 2.150] | 0.781 [-0.800, 2.346] | 0.650 [-0.830, 2.136] | 0.729 [-0.810, 2.197] |

## Diagnostics and conclusion

- Maximum R-hat: `1.0040` (required <= `1.05`).
- Minimum bulk ESS: `1652.9` (required >= `200.0`).
- The predeclared theta-retention diagnostic passes in `8` of `8` cells.
- Adding the slow-state slope channel does not remove the positive direct-channel update. Under IID unemployment-gap specifications, `P(theta_CIQ>0)=0.829` for both weightings.
- `delta` is not sign-identified: every 95% interval includes zero. The free-combined run supports channel separability, not the HSA cross-equation restriction.
- With unrestricted real `lambda`, the static equality `delta=lambda*theta` is a reparameterization whenever `theta` is nonzero; it is not a fit restriction.
