# Quarterly establishment identification experiment

## Purpose

Under the production `annual_q4` design, the inverse-HHI competition measure is
observed only once per year. In HSA steady the inflation equation has zero loading
on `Nhat`, so identifying that state there has no direct economic payoff. The
target specification is therefore HSA const-theta, where `-theta*Nhat` is part of
inflation. The experiment adds an independent quarterly signal for that state and
estimates `theta` jointly with its AR(2) process.

It is deliberately separate from the production report. Its data spec is
`unemployment_gap_core_establishment`, its model is HSA const-theta, and its fixed
sample is 1993Q2–2012Q4 (`T=79`).

## Quarterly establishment stock

The restored BED files contain quarterly establishment flows, not the total stock:

- `...120007LQ5`: seasonally adjusted establishment births;
- `...120008LQ5`: seasonally adjusted establishment deaths.

Source values are in thousands. The annual BDS `ESTAB` count for 1993
(5,682,098) is treated as a Q1 level anchor. For quarter-end dates from 1993Q2,

```text
net_entry_t = 1000 * (births_t - deaths_t)
E_t         = E_{t-1} + net_entry_t
```

The first reconstructed stock is therefore 5,703,098 in 1993Q2. The data builder
keeps births, deaths, net entry, and the reconstructed stock as separate columns.

BDS and BED are different statistical programs. Cumulated BED net entry reaches
7,108,098 in 2012Q4, while annual BDS reports 6,713,567 for 2012. The experiment
does not silently force those levels to agree. Most of that discrepancy is slow
moving and is removed by the cycle filter, but benchmarking the two sources is a
future robustness check.

## Establishment decomposition

Within the selected 79-quarter sample, the stock is transformed to the same
ten-log-point convention used for competition:

```text
E_model_t = (100*log(E_t) - sample_mean)/10
E_model_t = Ebar_t + Ehat_obs_t
```

`Ebar` is an HP trend with quarterly `lambda=1600`; `Ehat_obs` is the residual.
The filter is run after applying the sample window, so data after 2012Q4 cannot
affect the endpoint.

## Additional observation equation

The existing state remains

```text
Nhat_t = rho_1*Nhat_{t-1} + rho_2*Nhat_{t-2} + u_t
Nbar_t = n + Nbar_{t-1} + epsilon_t
Nobs_t = Nhat_t + Nbar_t + nu_t              (annual Q4 only)
pi_t   = ... + (kappa_0 + delta*Nbar_t)*x_t - theta*Nhat_t + e_t
```

The experiment adds a quarterly row to the same exact Kalman/FFBS update:

```text
Ehat_obs_t = lambda_E*Nhat_t + omega_t
omega_t ~ Normal(0, sigma_E^2)
```

Both `lambda_E` and `sigma_E^2` have conjugate Gibbs updates. `theta` is updated
in the inflation regression conditional on the same `Nhat` path. The baseline priors
are `lambda_E ~ Normal(1, 1)` and `sigma_E^2 ~ IG(2, 0.01)`. No sign constraint is
imposed. Annual inverse HHI continues to identify the scale and economic meaning
of `N`, while the establishment cycle supplies quarterly timing information.

### Current identification warning

With both `lambda_E` and `sigma_E` free, the pilot posterior has two persistent
regions. Starting the state from the annual-N decomposition leads to an
"ignore E" region (`lambda_E` near zero and `sigma_E` near the sample standard
deviation of `Ehat`). Starting from `Nhat = Ehat/lambda_E` leads to a linked
region (`lambda_E` around 0.31 and `sigma_E` around 0.02 in the 400-iteration
smoke run). The corresponding `theta` draws also differ materially.

The experimental sampler initializes from the linked path so that the intended
model can be exercised, but this is not evidence that the link is identified.
A production interpretation requires resolving the two regions, for example by
an economically justified positive-loading restriction and an informative
measurement-error calibration, or by replacing the proxy equation with a joint
transition model for `Nhat` and `Ehat`.

When `Ehat_data` is absent, the original two-row FFBS branch is executed without
additional random draws. Existing production models and their seeded results are
therefore unchanged. HSA steady also supports the added observation as a pure
state-identification diagnostic, but it is not the experiment run by script 14
because `theta=0` there by construction.

## Running it

```bash
python scripts/01_build_data.py
python scripts/14_estimate_establishment_augmented.py --quick --chains 1 --no-save
python scripts/14_estimate_establishment_augmented.py
```

Saved experimental runs go to `results/experiments/establishment_augmented/`, not
`results/runs/`, so the report builder cannot select them accidentally.

The quick run is only a pipeline check. It must not be interpreted as a converged
posterior or substituted into the report.
