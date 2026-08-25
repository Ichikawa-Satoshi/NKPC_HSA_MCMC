# Mixed-frequency Gustavo × Capital IQ test

> **Current status: mock only — saved output is NOT FOR INFERENCE.**

## 1. Question and status

- Research question: can exact annual-Q4 Gustavo firm-stock levels and genuinely
  observed Capital IQ quarterly changes identify a stable slow/cycle competition
  decomposition without constructing artificial quarterly observations?
- Null/benchmark comparison: the measurement-only state is compared in blocked
  Capital IQ prediction with equal annual allocation and the previously estimated
  average quarterly allocation.
- Why needed: Capital IQ is short, while assigning Gustavo to the slow state alone
  prevents annual observations from constraining total competition. The older
  allocated-quarterly construction looked longer but did not add observed
  direct-channel variation.
- Current status: `mock only`.
- Replaces/supersedes: nothing yet. It is a candidate successor to the mechanical
  quarterly allocation used in `hsa_exact_n_decomposition` and differs from the
  separate-role diagnostic in `gustavo_state_capitaliq_cycle`.

The competition block is modularly cut: inflation never updates its posterior.

## 2. Model and equations

The latent log competition coordinate is

```math
n_t=\bar n_t+\hat n_t.
```

The slow and cycle states follow

```math
\bar n_t=\bar n_{t-1}+m_{q(t)}+\eta^b_t,
\qquad
\hat n_t=\phi_1\hat n_{t-1}+\phi_2\hat n_{t-2}+u_t,
```

```math
\sigma_{\bar n}^2=\omega\tau^2,
\qquad
\sigma_{\hat n}^2=(1-\omega)\tau^2.
```

Here `m_q(t)` is the pre-estimated average within-year allocation of the known
Gustavo annual change. It is a transition mean, not a quarterly observation.
The AR(2) coefficients are parameterized by cycle damping `r` and period `P`:

```math
\phi_1=2r\cos(2\pi/P),\qquad \phi_2=-r^2.
```

At every observed Q4, Gustavo is imposed as an exact conditioning restriction:

```math
g_y=(\bar n_t+\hat n_t)_{t=yQ4}.
```

It is not entered as a zero-variance Gaussian density. Doing so would make the
likelihood degenerate as `tau -> 0`. Capital IQ firm-weighted and revenue-weighted
quarterly log changes are the noisy measurement equations:

```math
\Delta c_{j,t}=a_j+b_j\Delta(\bar n_t+\hat n_t)+e_{j,t},
\qquad e_{j,t}\sim N(0,\sigma_j^2).
```

Missing Capital IQ quarters remain missing. Conditional on competition-state
draws, the two exploratory NKPCs are

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+\kappa_0x_t-\theta_N\hat n_t+\varepsilon_t,
```

and

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+(\kappa_0+\delta\bar n_t^c)x_t-\theta_N\hat n_t+\varepsilon_t,
\qquad \varepsilon_t\sim N(0,\sigma_\pi^2).
```

The oil-control cells add current and lagged annualized QoQ real-oil-price
changes. This folder does not estimate `lambda` and does not impose an HSA
restriction. The admissible sequence is measurement validation, free `theta_N`,
free `(delta, theta_N)`, then and only then HSA restriction testing.

## 3. Data and frozen transformations

- Price: PPI, annualized QoQ `400 Delta log P`.
- Activity: inverse of markup.
- Expectations: genuine SPF one-quarter-ahead annualized-log forecast.
- Gustavo: annual effective-firm count, Q4 only, transformed to 10 log points
  relative to 1993Q4.
- Capital IQ: firm-weighted and revenue-weighted effective-firm counts; each is
  transformed to 10 log points relative to 1993Q4 and differenced only after
  reindexing to the complete quarterly grid. Sparse pre-quarterly Q4 observations
  are therefore not mistaken for QoQ changes.
- State sample: 1974Q4–2013Q4, 157 quarters; 40 Gustavo Q4 constraints and 84
  observed QoQ changes for each Capital IQ measure.
- NKPC sample: 1993Q2–2013Q4, 83 observations.
- Timing: current `hat n_t` and current centered `bar n_t`.
- Missing data: Capital IQ is omitted from the measurement update where absent;
  no interpolation is recorded as data.
- Centering: `bar n_t` is centered inside every propagated state draw for the
  slope interaction. `theta_N` uses the unstandardized 10-log-point coordinate.
- Immutable hashes: stored in each `results/<profile>/manifest.json`.

These choices were frozen before the mock result. The oil cells are predeclared
robustness checks rather than a response to a coefficient sign.

## 4. What changes and what is held fixed

| Dimension | Values tested | Held fixed? |
|---|---|---|
| Price | PPI | yes |
| Activity | inverse markup | yes |
| State law | exact-Q4 mixed-frequency RW + AR(2) | yes |
| Inflation error | IID | yes |
| Channel | direct only; free static combined | changes |
| Oil | absent; current and lagged controls | changes |
| HSA restriction | none | yes |
| Timing | current | yes |

The blocked measurement backtest hides both Capital IQ measures in 1998–99,
2004–05, and 2010–11, refits without inflation, and predicts the hidden changes.

## 5. Estimands, priors, and expected signs

| Parameter | Meaning | Prior | Theory sign | Identification criterion |
|---|---|---|---|---|
| `theta_N` | direct competition channel | centered Normal, scale calibrated before fit | positive | P(>0) >= .80 and post/prior SD <= .75 |
| `delta` | slow competition slope channel | centered Normal, scale calibrated before fit | positive under the maintained HSA orientation | same |
| `omega` | slow share of total innovation variance | Beta(2,8) | none | convergence plus visible prior update |
| `tau` | total state innovation SD | half-Normal | positive support only | convergence and non-boundary posterior |
| `r` | AR(2) damping | transformed Beta(4,3), [0.25,.95] | none | convergence |
| `P` | AR(2) period | truncated transformed N(12,4^2), [6,24] | none | convergence |
| `b_j` | Capital IQ loading | N(1,1.5^2) | positive expected, not imposed | posterior learning |

Positive support for standard deviations is not evidence for an HSA sign.

## 6. Sampling profiles

| Profile | State iterations / warmup / thin / chains | NKPC iterations / warmup / thin / chains | Purpose |
|---|---|---|
| mock | 1200 / 400 / 2 / 2 | 1000 / 350 / 2 / 2 | code-path check only |
| quick | 5000 / 1500 / 5 / 4 | 5000 / 1500 / 5 / 4 | diagnostic screening |
| full | 20000 / 6000 / 7 / 4 | 12000 / 4000 / 4 / 4 | inference only if all gates pass |

Seed: 20260901. The state parameters use an adaptive random-walk proposal during
warmup and freeze adaptation afterward. State paths are drawn conditionally by a
simulation smoother and projected to machine-exact Q4 totals.

## 7. Gates declared before estimation

- Mock: max rank R-hat <= 1.20 and min bulk ESS >= 50.
- Quick: max rank R-hat <= 1.05 and min bulk ESS >= 400.
- Full: max rank R-hat <= 1.01 and min bulk ESS >= 800.
- Exact Q4 identity error <= 1e-8.
- Mixed-frequency mean blocked RMSE must be no worse than average allocation.
- A structural coefficient is suggestive only if P(theory sign) >= .80 and its
  posterior/prior SD ratio <= .75; .975 is the strong sign threshold.
- WAIC is descriptive until convergence and identification pass. No marginal
  likelihood or HSA model evidence is computed in the mock.

## 8. Exact commands

From the repository root:

```bash
PYTHONPATH=src:. python tests/mixed_frequency_gustavo_capitaliq/run.py --profile mock
PYTHONPATH=src:. python tests/mixed_frequency_gustavo_capitaliq/run.py --profile quick
PYTHONPATH=src:. python tests/mixed_frequency_gustavo_capitaliq/run.py --profile full
PYTHONPATH=src:. pytest -q tests/mixed_frequency_gustavo_capitaliq/test_functions.py
PYTHONPATH=src:. python tests/mixed_frequency_gustavo_capitaliq/build_report.py --profile mock
```

Do not run `full` merely because it exists; promote only after the quick gates.

## 9. Output inventory

For profile `p`, outputs are under `results/p/`:

- `manifest.json`: run identity, hashes, samples, gates and report path;
- `draws/state.npz`: parameter and `nbar`/`nhat`/total draws;
- `draws/*.npz`: four NKPC fits;
- `tables/state_parameters.csv`, `state_paths.csv`;
- `tables/coefficients.csv`, `prior_posterior.csv`, `convergence.csv`;
- `tables/backtest.csv`, `backtest_summary.csv`;
- `tables/model_comparison.csv`;
- `report/mixed_frequency_gustavo_capitaliq_p.pdf`;
- `RESULTS.md`: numerical run summary.

## 10. Results

### Mock run retained on 2026-08-25

The `mock` profile completed one measurement-state fit, three blocked
measurement refits, and four NKPC cells in 64 seconds. It is **not for inference**.
The manifest is `results/mock/manifest.json` and the complete numerical summary
is `results/mock/RESULTS.md`.

| Cell/model | Parameter | Mean | 95% interval | P(>0) | Post/prior SD | R-hat | Bulk ESS |
|---|---|---:|---|---:|---:|---:|---:|
| direct, no oil | `theta_N` | -1.694 | [-9.746, 6.726] | .331 | .556 | .998 | 593 |
| direct, oil | `theta_N` | -2.141 | [-7.979, 4.787] | .215 | .394 | 1.000 | 623 |
| combined, no oil | `delta` | .350 | [-2.461, 2.874] | .612 | .625 | 1.004 | 623 |
| combined, no oil | `theta_N` | -1.291 | [-9.070, 6.555] | .383 | .547 | 1.007 | 548 |
| combined, oil | `delta` | .243 | [-2.065, 2.416] | .589 | .543 | 1.003 | 681 |
| combined, oil | `theta_N` | -2.328 | [-8.330, 4.044] | .225 | .417 | .999 | 683 |

| Gate | Threshold | Observed | Pass? |
|---|---|---|---|
| Convergence | max R-hat <= 1.20; min bulk ESS >= 50 | 2.390; 2.5 | no |
| Exact Q4 identity | error <= 1e-8 | 8.9e-16 | yes |
| Blocked measurement RMSE | no worse than average allocation | -7.1% relative improvement | no |
| Structural sign/learning | P(>0) >= .80 and post/prior SD <= .75 | all `theta_N` P(>0) <= .383 | no |

The test establishes that the exact-conditioning formulation is operational and
removes the zero-variance likelihood degeneracy. It does not establish a usable
mixed-frequency competition state or an HSA direct channel. The hybrid predicts
hidden Capital IQ growth worse than average and equal allocation, its AR(2)
damping is not converged, and every `theta_N` posterior leans negative. It does
not change the preferred staged specification and is not promoted to quick.

## 11. Limitations and next admissible step

The Capital IQ overlap supplies only 84 quarterly changes in the state/NKPC
overlap. Exact annual totals cannot by themselves identify intra-year variation.
State/measurement variance separation may remain weak, and `delta` and `theta_N`
may be collinear once both are included. The loadings are estimated, so common
factor scale must be checked rather than assumed. WAIC does not test the HSA
cross-equation restriction.

If the blocked prediction and free-`theta_N` quick gates pass, run a predeclared
simulation-recovery grid on the realized state path. Only after recovery should
the free combined model be compared with an HSA-restricted static model. If the
measurement backtest fails, retain the failure and do not use inflation feedback
to rescue the state.

## 12. Changelog

| Date | Change | Reason | Affects interpretation? |
|---|---|---|---|
| 2026-08-25 | Created mixed-frequency exact-Q4 bundle | use observed timing without artificial quarterly N | yes; mock only |
| 2026-08-25 | Treat Gustavo as an exact condition, not a zero-noise density | avoid unbounded likelihood at zero state variance | yes; corrects degeneracy |
