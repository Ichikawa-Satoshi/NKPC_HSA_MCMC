# Gustavo state x Capital IQ cycle

This isolated QoQ mock assigns the slow competition state to annual Gustavo
effective-firm counts and the quarterly cycle to Capital IQ. The two series are
linked by an estimated intercept and loading, not equated as levels. Inflation
never updates either measurement block.

The prespecified PPI/Core-CPI oil-control extension is documented in
`RESULTS_OIL_CONTROL.md`. It adds current and one-quarter-lagged annualized QoQ
changes in the repository's real WTI/CPI index to every M0--M4 equation while
leaving the cut competition states unchanged. Full results are saved in
`results/oil_control_full/` and are reproduced by
`run_oil_control_validation.py`.

Run from the repository root:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/run.py --workers 4
```

Regenerate the saved English report:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/build_report.py
```

Run the nested free-combined diagnostic using the already saved competition
posterior draws:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/run_combined.py --workers 4
```

Regenerate only its English report:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/build_combined_report.py
```

Run the staged recovery, nested comparison, holdout, and automatic HSA promotion
gate:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/run_staged_validation.py --workers 4
```

The command reuses a completed run only when all three configuration hashes
match. Force every observed and recovery fit to rerun with:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/run_staged_validation.py --workers 4 --refit
```

Regenerate only the staged English report:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/build_staged_report.py
```

Run the explicitly authorized varying-theta, free-dynamic, and
HSA-restricted-dynamic weak-identification diagnostic:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/run_dynamic_validation.py --workers 4
```

Completed fits are reused only when the base and dynamic configuration hashes
match. Force all 27 observed fits and 300 recovery fits to rerun with `--refit`.
Regenerate only the English dynamic report with:

```bash
PYTHONPATH=src:. python tests/gustavo_state_capitaliq_cycle/build_dynamic_report.py
```

Run unit tests:

```bash
PYTHONPATH=src:. pytest -q tests/gustavo_state_capitaliq_cycle/test_functions.py
```

Primary outputs are under `tests/gustavo_state_capitaliq_cycle/results/mock_qoq/`.
The earlier overlapping-YoY output is preserved under `results/mock_yoy_legacy/`
and its result record is `RESULTS_YOY_LEGACY.md`.
`RESULTS.md` is regenerated from the recorded mock tables. This bundle has no
full profile and must not be used for empirical inference.

The nested result is recorded separately under `results/free_combined_qoq/` and
in `RESULTS_COMBINED.md`; it does not overwrite the direct-only recovery run.

## Recorded QoQ mock result (2026-08-25)

- Exact Gustavo Q4 anchor error: `9.99e-16`.
- Firm-weighted Capital IQ cycle maximum R-hat: `1.061`.
- Revenue-weighted Capital IQ cycle maximum R-hat: `1.040`.
- The full mock gate fails because one AR(1) recovery replicate has bulk ESS `20.5`,
  below the frozen mock requirement of `50`.
- All eight observed `theta_CIQ` intervals include zero. Under primary IID,
  posterior-positive probabilities range from `0.788` to `0.837`; under AR(1),
  they range from `0.771` to `0.792`.
- For firm-weighted PPI x inverse markup, primary IID gives
  `0.718 [-0.766,2.169]`; AR(1) gives `0.665 [-1.165,2.464]`.
- Under IID, oracle recovery detects injected `theta_CIQ=3` in all three mock
  replicates, while propagated recovery detects one of three. Both detect all
  replicates at `10`. Three replicates are not a power estimate.

The firm-weighted measurement loading on the Gustavo slow state is negative,
`-0.383 [-0.761,-0.061]`, whereas the revenue-weighted loading includes zero.
This is evidence that the two effective-firm series should not be treated as the
same observed level; it is not evidence for a structural negative mapping.

## Recorded free-combined diagnostic (2026-08-25)

The added model is

```math
\pi_t^q=a+\alpha_b\pi_{t-1}^q+\alpha_fE_t\pi_{t+1}^q
+[\kappa_0+\delta(\bar n_t-\overline{\bar n})]x_t
-\theta_{CIQ}\hat n_t+\varepsilon_t.
```

It reuses the preceding measurement draws byte-for-byte. Four chains with 2,000
iterations and 600 warmup iterations pass the computational gate: maximum R-hat
is `1.0025` and minimum bulk ESS is `1,618.8`.

- `theta_CIQ` passes the predeclared retention diagnostic in all eight cells.
- Under IID PPI x negative unemployment gap, firm weighting gives
  `0.781 [-0.800,2.346]`, `P(theta_CIQ>0)=0.829`; revenue weighting gives
  `0.729 [-0.810,2.197]`, also with probability `0.829`.
- The corresponding `delta` positive probabilities are `0.673` and `0.678`, but
  all `delta` intervals include zero.
- The largest absolute posterior correlation between `delta` and `theta_CIQ` is
  `0.352`; adding the slow slope regressor does not explain away the direct update.

This checks channel coexistence and separability. It does not impose
`delta=lambda*theta_CIQ` and is not a structural HSA test.

## Recorded staged validation (2026-08-25)

The staged run completed 17 observed-data fits and 640 simulation-recovery fits.
At the observed standardized unemployment-gap effects
`(s_delta,s_theta)=(0.06,0.11)`, propagated-state suggestive recovery is `0.100`
for `delta` and `0.333` for `theta_CIQ`, against a predeclared `0.80` promotion
threshold. Oracle-state rates are `0.133` and `0.333`, so removing measurement
state uncertainty does not resolve the problem.

Recovery computation passes: maximum R-hat is `1.0395`, minimum bulk ESS is
`100.6`, interval coverage is `1.00`, and null false-positive rates are zero for
both target coefficients. The failure is statistical detection, not MCMC failure.

The direct-only/free-combined comparison also does not favor adding `delta`:

- free combined has lower 2010Q1-2013Q4 holdout ELPD in all four cells;
- the Savage-Dickey `BF01` ranges from `1.38` to `1.57`, mildly favoring
  `delta=0`;
- PSIS-LOO is not decisive because every comparison has a Pareto-k value above
  one.

The predeclared gate therefore stopped the dynamic branch. Neither free dynamic
nor HSA-restricted dynamic was estimated.

## Recorded dynamic weak-identification diagnostic (2026-08-25)

At the user's explicit request, the dynamic branch was subsequently run as a
diagnostic despite the failed promotion gate. This does not reverse that gate.
The run estimates varying theta, free dynamic, and HSA-restricted dynamic models
for two Capital IQ cycles and two PPI activity cells, plus the firm-weighted
unemployment-gap AR(1) robustness fits. It completed 27 observed fits and 300
varying-theta recovery fits.

- Observed maximum R-hat is `1.0011` and minimum bulk ESS is `4,074.5`.
- In the varying-theta IID model, `P(theta_0>0)` is `0.804`--`0.830`, but all
  four intervals include zero. Every `gamma` interval includes zero and leans
  negative.
- In the HSA-restricted dynamic model every `lambda`, derived `delta_1`, and
  derived `delta_2` interval includes zero. For the firm-weighted unemployment
  cell, `lambda=0.402 [-5.288,5.952]`.
- At standardized `gamma=0.10`, propagated-state suggestive recovery is `0.533`
  and strong recovery is only `0.067`. Strong recovery reaches `0.800` only at
  `gamma=0.40`.
- Relative to constant theta, all twelve dynamic specifications have lower
  2010Q1--2013Q4 holdout ELPD. PSIS-LOO remains descriptive because all cells
  have influential observations.

The dynamic sampler works, but the data do not identify time variation in theta
or the HSA cross-equation restrictions. See `RESULTS_DYNAMIC.md` for the full
coefficient and comparison tables.

## Output inventory

- `SPECIFICATION.md`: frozen modular equations and interpretation rules.
- `RESULTS.md`: generated numerical mock record.
- `results/mock_qoq/manifest.json`: data hash, settings, sample, and full mock gate.
- `results/mock_qoq/draws/`: fixed measurement, QoQ NKPC, and recovery draws.
- `results/mock_qoq/tables/`: parameters, coefficients, paths, and recovery tables.
- `results/mock_qoq/report/gustavo_state_capitaliq_cycle_qoq_report.pdf`: English report.
- `RESULTS_COMBINED.md`: exact numerical record for the free-combined extension.
- `results/free_combined_qoq/`: free-combined draws, tables, manifest, and report.
- `RESULTS_STAGED.md`: exact staged recovery, comparison, and stopping record.
- `results/staged_validation/`: observed fits, all recovery tables, promotion
  gate, manifest, and staged English report.
- `RESULTS_DYNAMIC.md`: exact varying-theta and dynamic-HSA diagnostic record.
- `results/dynamic_validation/`: 27 observed fits, 300 recovery fits, comparison
  tables, manifest, and the dynamic English report.
- `RESULTS_CORE_CPI.md`: exact Core-CPI QoQ M0--M4 result and interpretation.
- `results/core_cpi_smoke/`: short 45-fit plumbing check; not used for inference.
- `results/core_cpi_full/`: 45 full observed fits, 780 recovery fits, coefficients,
  predictive comparisons, manifest, and saved posterior draws.
- `../../docs/report/qoq_gustavo_capitaliq/`: comprehensive English
  report in the common `docs/report` format. It combines the staged and dynamic
  QoQ evidence, exact equations, source and sample documentation, coefficient
  tables, prior/posterior learning, state and time-varying coefficient paths,
  recovery, prediction, convergence, and failure diagnosis.

To rebuild the comprehensive report from the saved estimation outputs:

```bash
cd /Users/satoshi/GitHub/NKPC_HSA_MCMC
python docs/report/qoq_gustavo_capitaliq/build_assets.py
cd docs/report/qoq_gustavo_capitaliq
latexmk -pdf -interaction=nonstopmode -halt-on-error qoq_hsa_report.tex
```

## Changelog

- `v1`, 2026-08-25: added the exact Gustavo Gaussian bridge, modular Capital IQ
  AR(2) measurement, two PPI cells, oracle/propagated recovery, unit tests, and
  the overlapping-YoY mock, now archived as legacy.
- `v2`, 2026-08-25: held the competition measurement draws byte-for-byte fixed,
  replaced YoY/MA(3) with annualized QoQ plus genuine one-quarter-ahead SPF,
  added IID primary and AR(1) robustness fits, and reran recovery.
- `v3`, 2026-08-25: added the separately recorded free-combined QoQ diagnostic,
  within-draw slow-state centering, four-chain convergence gates, direct-versus-
  combined tables, and an English nested-model report.
- `v4`, 2026-08-25: added standardized joint recovery with 30 primary
  replications, oracle/propagated modes, 2010Q1 holdout prediction, PSIS-LOO,
  Savage-Dickey nested evidence, and an automatic dynamic-HSA stopping gate.
- `v5`, 2026-08-25: at explicit user direction, ran varying-theta, free-dynamic,
  and HSA-restricted-dynamic models as a post-gate weak-identification diagnostic;
  added dynamic recovery, AR(1) robustness, predictive comparisons, and an
  English seven-page report.
- `v6`, 2026-08-25: added a unified `docs/report`-format account of the full QoQ
  sequence, including the reasons each model is estimated, exact nesting logic,
  all PPI and Core-CPI cells, source and period details, diagnostic plots, predictive
  comparisons, and a consolidated interpretation of weak identification.
- `v7`, 2026-08-25: added a separately estimated Core-CPI QoQ branch using the
  matched SPF headline-CPI CPI3 expectation proxy, the same cut competition
  draws, all M0--M4 models, static and dynamic recovery, holdout comparisons,
  AR(1) robustness, and integration into the comprehensive report.
