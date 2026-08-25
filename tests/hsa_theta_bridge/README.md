# HSA theta bridge validation

This is a controlled 2x2 diagnostic. Every cell uses the same PPI inflation,
negative unemployment gap, SPF expectations, Gustavo annual constraints,
Capital IQ quarterly-allocation posterior, exact identity
`N = Nbar + Nhat`, variance-share parameterization, coefficient priors, and
AR(1) inflation disturbance.

Only two switches change:

| state split | lambda | label |
|---|---:|---|
| cut (inflation excluded from Nbar/Nhat split) | 6 | `cut_fixed6` |
| cut | estimated, sign unrestricted | `cut_free` |
| joint split (inflation conditions Nbar/Nhat) | 6 | `joint_fixed6` |
| joint split | estimated, sign unrestricted | `joint_free` |

The bridge conditions all four cells on the same Capital IQ-updated quarterly
allocation posterior mean. This removes allocation Monte Carlo variation from
the comparison. The allocation remains external in all cells;
"joint" here means joint inference for the exact slow/cycle split conditional on
the same allocation path. It does not let inflation rewrite Gustavo anchors
or Capital IQ allocation weights.

Run a short test and then the full validation:

```bash
python tests/hsa_theta_bridge/run.py --quick
python tests/hsa_theta_bridge/run.py
```

## Full-run result

All values below use 4 chains. The overall maximum rank-normalized R-hat is
1.009 and the maximum exact-identity error is 2.22e-16.

| cell | theta mean [95% interval] | P(theta>0) | lambda mean [95% interval] | delta=lambda*theta mean | sd(mean Nhat path) |
|---|---:|---:|---:|---:|---:|
| `cut_fixed6` | 0.036 [-0.022, 0.089] | 0.908 | 6 fixed | 0.215 | 0.111 |
| `cut_free` | 0.005 [-0.146, 0.160] | 0.528 | 0.15 [-14.77, 15.31] | 0.166 | 0.111 |
| `joint_fixed6` | 0.037 [-0.021, 0.091] | 0.907 | 6 fixed | 0.221 | 0.113 |
| `joint_free` | 0.002 [-0.156, 0.163] | 0.508 | 0.09 [-14.73, 14.64] | 0.173 | 0.113 |

The cut-to-joint switch barely changes either theta or the state split. The
fixed-to-free lambda switch makes theta sign-symmetric, while the product
delta=lambda*theta remains shifted positive (P(delta>0) about 0.86). Thus the
new loss of theta identification is primarily the factorization of the slope
coefficient into two unrestricted parameters, not the modular cut. The older
measurement-error model also assigned much more variation to Nhat, explaining
why its fixed-lambda theta signal was stronger than either exact-N fixed-lambda
cell here.
