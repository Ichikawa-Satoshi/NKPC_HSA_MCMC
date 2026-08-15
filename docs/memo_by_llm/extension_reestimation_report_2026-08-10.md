# NKPC/HSA QCEW and SEC inverse-HHI extension audit

Date: 2026-08-10

## Executive findings

The requested joint firm/establishment model and SEC inverse-HHI data specification were implemented without replacing or overwriting the annual-Q4 baseline. The production QCEW grid contains 30 HSA steady/const-theta cells; the SEC grid contains all 77 cells in the current report grid (CES, HSA steady, dynamic, const-theta, and full).

For the primary HSA-steady comparison, adding QCEW establishments leaves `delta` and the average `kappa_t` essentially unchanged. The posterior innovation correlation is weak: `corr(u_N,u_E)` has mean 0.073 and 95% interval [-0.312, 0.443]. The firm-cycle path is less, not more, precisely estimated and mixes badly. Quarterly establishments therefore do not solve quarterly firm-cycle identification in this run.

SEC inverse HHI changes the primary `delta` posterior from positive to centered almost exactly at zero and changes average `kappa_t` from positive to negative. This is substantial sensitivity to the competition proxy. It is not a clean same-sample comparison: the SEC series supports 2012Q1--2025Q1 in the primary macro cell, whereas A/B cover 1982Q1--2012Q4.

## Resolved storage and inputs

- Repository: `/Users/satoshi/GitHub/NKPC_HSA_MCMC`
- Dropbox root: `/Users/satoshi/Library/CloudStorage/Dropbox`
- Dropbox project: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC`
- QCEW processed input: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/data/processed/qcew_establishments.csv`
- QCEW model-ready input: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/data/processed/model_ready_qcew_joint.csv`
- QCEW raw directory: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/data/raw/competition/qcew`
- SEC HHI input: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/data/processed/sec_hhi_quarterly.csv`
- SEC model-ready input: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/data/processed/model_ready_sec_inverse_hhi.csv`
- Annual firm source: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/data/raw/competition/BN_N_Gustavo_26.csv`, column `original_series`

The central path resolver now prefers `NKPC_HSA_PROJECT_DIR` and `NKPC_HSA_DROPBOX_DIR` when set, otherwise discovers the checkout and the same-named Dropbox project folder. No machine-specific Dropbox root is embedded in data/model code or configuration.

### QCEW verification

The exact archive field is `qtrly_estabs_count`, defined as “Count of establishments for a given quarter.” The loader canonicalizes it to `qcew_establishments` only after filtering:

- `area_fips == US000` (United States)
- `own_code == 5` (private ownership)
- total/all-industries record (`industry_code == 10` or equivalent title/code in older layouts)
- quarters 1--4

The resulting series has 124 observations from 1982Q1 through 2012Q4, no duplicate quarter, no missing quarter, no missing or nonpositive level, and ranges from 4,654,640 to 8,883,186 establishments. It is a quarterly published level, already aggregated nationally across private total industry, and is not seasonally adjusted. It is not a firm count and is never inserted in the NKPC as the competition state.

The raw directory contains one source per year (31 sources): official annual QCEW ZIP totals for 1982--1989 and selected official total-industry CSV records for 1990--2012. No QCEW file was found during the initial Dropbox search, so the official BLS archive fallback was used; 2011--2012, which remained missing after the interrupted first extraction, were completed from the official archive/archived official files. This is the only internet data fallback used.

### SEC HHI verification

The exact model input column is `hhi`. It is a fraction, not a 0--10,000 HHI:

- `hhi` range: 0.205396--0.227526
- `hhi_10000 == 10000 * hhi` (range 2053.96--2275.26)
- `effective_firms == 1 / hhi` (range 4.3951--4.8686)

The file has 56 quarterly observations from 2012Q1 through 2025Q4, no duplicates, no missing values, and no zero, negative, or above-one HHI. The estimator uses `N_SEC_inverse_HHI = 1 / hhi`, not `10000 / hhi` and not the pre-existing column without validation.

Repository construction code forms firm revenue shares within three-digit SIC markets, sums squared shares to market HHI, then takes a firm-count-weighted mean across SIC3 markets for each quarter. Eligible facts are deduplicated SEC 10-Q/10-K revenue observations. The file covers 248--270 markets and 3,527--4,897 firms per quarter. `hhi_market_mean` and `hhi_revenue_weighted` are retained diagnostics but are not the selected competition measure.

## Transformations and observation timing

- A/B firm series: raw annual `original_series` is mapped to Q4 only. No Q1--Q3 firm observations are interpolated in estimation. The preserved transform is `(100 log N - 206.2059507304) / 10`; the center is the mean of the existing 124-quarter model-ready reference path over 1982Q1--2012Q4, matching current baseline behavior. Annual raw levels are transformed with that center.
- B establishment series: `(100 log E - 1574.2889114690) / 10`, centered on its own 124 quarterly observations over 1982Q1--2012Q4. It is never centered with the firm mean.
- C SEC effective N: first `N_eff = 1 / hhi`, then `(100 log N_eff - 153.7171157517) / 10`, centered once on the 53 observations used in the primary 2012Q1--2025Q1 sample.
- Stored `kappa0`, `delta`, and `kappa_t` are physical coefficients. `KAPPA_SCALE=100` is applied once internally and divided out once at storage.

## Statistical implementation

The joint state order is exactly:

`[Nhat_t, Nhat_{t-1}, Nbar_t, Ehat_t, Ehat_{t-1}, Ebar_t]`.

The transition covariance places firm and establishment cycle shocks at indices 0 and 3 and uses `Cov(u_N,u_E) = rho_NE * sigma_uN * sigma_uE`; separate trend innovations are at indices 2 and 5. A positive-definite 2x2 cycle covariance is sampled with an inverse-Wishart update and stored as `rho_NE` plus the two standard deviations.

One six-dimensional FFBS smoother draws the complete state path. Its observation row order is inflation, finite firm level, finite establishment level, with `R` constructed in that same order. At Q1--Q3 only the missing firm row is omitted. The inflation row remains linked to `Nbar` (and to `Nhat` in const-theta), never to `Ebar` or `Ehat`.

The linear joint kernel is supported for HSA steady and HSA const-theta. HSA full is bilinear in the competition states, and dynamic/full would require a separately validated nonlinear joint Particle-Gibbs kernel. The QCEW grid therefore contains the 30 applicable steady/const-theta production cells rather than inserting an invalid FFBS into nonlinear models. SEC inverse HHI is an ordinary replacement competition series and therefore runs across all model families.

## Primary A/B/C posterior comparison

All intervals below are equal-tail 95% intervals. R-hat and bulk ESS are rank-normalized ArviZ diagnostics.

| Specification | Parameter | Mean | Median | 95% interval | R-hat | Bulk ESS |
|---|---:|---:|---:|---:|---:|---:|
| A annual firms | alpha | 0.8241 | 0.8247 | [0.7308, 0.9146] | 0.9996 | 3140 |
| A annual firms | kappa0 | 0.0582 | 0.0583 | [0.0233, 0.0921] | 1.0013 | 973 |
| A annual firms | delta | 0.0256 | 0.0253 | [0.0116, 0.0412] | 0.9998 | 2986 |
| A annual firms | n_N | -0.0425 | -0.0433 | [-0.0683, -0.0105] | 1.0062 | 685 |
| A annual firms | rho_N1 | 0.1883 | 0.1679 | [0.0687, 0.3497] | 1.0193 | 111 |
| A annual firms | rho_N2 | -0.8862 | -0.8970 | [-0.9715, -0.7517] | 1.0069 | 161 |
| A annual firms | sigma_uN | 0.1458 | 0.1489 | [0.0615, 0.2448] | 1.0163 | 112 |
| A annual firms | sigma_epsN | 0.1156 | 0.1157 | [0.0471, 0.2002] | 1.0156 | 92 |
| A annual firms | sigma_N | 0.0687 | 0.0643 | [0.0391, 0.1237] | 0.9996 | 1360 |
| B QCEW joint | alpha | 0.8255 | 0.8261 | [0.7302, 0.9166] | 1.0007 | 3152 |
| B QCEW joint | kappa0 | 0.0518 | 0.0519 | [0.0121, 0.0917] | 1.0185 | 243 |
| B QCEW joint | delta | 0.0238 | 0.0233 | [0.0101, 0.0397] | 1.0050 | 2177 |
| B QCEW joint | n_N | -0.0475 | -0.0474 | [-0.0807, -0.0152] | 1.0249 | 81 |
| B QCEW joint | rho_N1 | 0.9624 | 1.0181 | [0.1793, 1.3919] | 1.1238 | 11 |
| B QCEW joint | rho_N2 | -0.1695 | -0.1084 | [-0.9055, 0.2021] | 1.0930 | 22 |
| B QCEW joint | sigma_uN | 0.1254 | 0.1206 | [0.0622, 0.2157] | 1.0114 | 181 |
| B QCEW joint | sigma_epsN | 0.1205 | 0.1177 | [0.0491, 0.2152] | 1.0066 | 128 |
| B QCEW joint | sigma_N | 0.0705 | 0.0666 | [0.0391, 0.1230] | 1.0003 | 1376 |
| B QCEW joint | n_E | 0.0322 | 0.0324 | [0.0129, 0.0504] | 1.0003 | 872 |
| B QCEW joint | rho_E1 | 0.9875 | 0.9864 | [0.8364, 1.1450] | 1.0005 | 2641 |
| B QCEW joint | rho_E2 | -0.0075 | -0.0044 | [-0.1665, 0.1357] | 1.0004 | 2646 |
| B QCEW joint | sigma_uE | 0.0458 | 0.0454 | [0.0366, 0.0566] | 1.0008 | 2384 |
| B QCEW joint | sigma_epsE | 0.0385 | 0.0382 | [0.0294, 0.0492] | 1.0000 | 1610 |
| B QCEW joint | sigma_E | 0.0322 | 0.0319 | [0.0259, 0.0401] | 1.0003 | 2352 |
| B QCEW joint | corr(u_N,u_E) | 0.0731 | 0.0742 | [-0.3117, 0.4427] | 1.0037 | 1413 |
| C SEC inverse HHI | alpha | 0.9124 | 0.9136 | [0.8034, 1.0236] | 1.0009 | 2765 |
| C SEC inverse HHI | kappa0 | -0.0517 | -0.0494 | [-0.1604, 0.0441] | 0.9999 | 3055 |
| C SEC inverse HHI | delta | 0.0001 | 0.0004 | [-0.0382, 0.0379] | 1.0003 | 3164 |
| C SEC inverse HHI | n_N | -0.0038 | -0.0036 | [-0.0326, 0.0248] | 0.9996 | 3161 |
| C SEC inverse HHI | rho_N1 | 0.4315 | 0.4266 | [0.0820, 0.8028] | 1.0015 | 1242 |
| C SEC inverse HHI | rho_N2 | -0.3745 | -0.3752 | [-0.7224, -0.0160] | 1.0019 | 1010 |
| C SEC inverse HHI | sigma_uN | 0.1188 | 0.1168 | [0.0682, 0.1820] | 1.0002 | 728 |
| C SEC inverse HHI | sigma_epsN | 0.0964 | 0.0943 | [0.0493, 0.1565] | 1.0008 | 629 |
| C SEC inverse HHI | sigma_N | 0.0797 | 0.0756 | [0.0439, 0.1348] | 1.0009 | 731 |

### State paths and kappa

| Specification/path | Max R-hat | Min bulk ESS | Mean posterior path SD | Average-path mean (95% interval) |
|---|---:|---:|---:|---:|
| A Nbar | 1.013 | 108 | 0.479 | 0.375 [-0.002, 0.840] |
| A Nhat | 1.013 | 112 | 0.515 | 0.024 [-0.002, 0.051] |
| A kappa_t | 1.007 | 195 | 0.023 | 0.0674 [0.0321, 0.1012] |
| B Nbar | 1.071 | 19 | 0.727 | 0.666 [-0.105, 1.991] |
| B Nhat | 1.135 | 10 | 0.789 | -0.640 [-1.964, 0.094] |
| B Ebar | 1.003 | 861 | 0.906 | 1.239 [0.055, 3.343] |
| B Ehat | 1.003 | 863 | 0.906 | -1.239 [-3.346, -0.055] |
| B kappa_t | 1.017 | 197 | 0.023 | 0.0664 [0.0315, 0.1018] |
| C Nbar | 1.002 | 882 | 0.108 | -0.002 [-0.060, 0.053] |
| C Nhat | 1.002 | 992 | 0.113 | 0.002 [-0.049, 0.055] |
| C kappa_t | 1.000 | 3038 | 0.052 | -0.0517 [-0.1601, 0.0440] |

The A/B change in `delta` is small (0.0256 to 0.0238), and average `kappa_t` is almost identical (0.0674 to 0.0664). The apparent change in B's AR coefficients is not reliable: both coefficients and the N paths have severe convergence failures. Establishment parameters themselves mix well, but `Ebar` and `Ehat` remain a highly offset decomposition and their innovation correlation with N is weak.

The C result changes both magnitude and uncertainty: `delta` moves from a positive, relatively tight posterior to zero with a much wider interval; average `kappa_t` changes sign. This is competition-measure sensitivity jointly with sample-period sensitivity.

## Grid execution and convergence

Current production keys are obtained from `report_run_keys()`: 77 cells across 10 activity/inflation specifications, with 47 baseline-prior, 15 weak-prior, and 15 tight-prior cells. Model counts are CES 16, dynamic 16, and 15 each for steady, const-theta, and full.

- QCEW joint: 30/30 jobs completed and wrote `posterior.nc`; 0 process failures. Under the strict criterion that all requested scalar/path diagnostics satisfy R-hat <= 1.01 and bulk ESS >= 400, 0/30 pass. Worst R-hat is 1.630 and worst bulk ESS is 3.45. These are convergence failures, not missing outputs.
- SEC inverse HHI: 77/77 jobs completed; 0 process failures. 61/77 pass the strict diagnostic criterion. CES 16/16, dynamic 16/16, steady 15/15, and const-theta 14/15 pass; HSA full is 0/15. HSA-full failures reach R-hat 1.042 and bulk ESS 84.4.

Complete per-run diagnostics are in `results/extensions/comparison/extension_grid_convergence.csv`.

## Commands, configs, and output locations

Build:

`PYTHONPATH=src python scripts/15_build_extension_data.py`

Production estimation:

`PYTHONPATH=src python scripts/16_estimate_extensions.py qcew_joint --jobs 4`

`PYTHONPATH=src python scripts/16_estimate_extensions.py sec_inverse_hhi --jobs 4`

Comparison:

`PYTHONPATH=src python scripts/17_compare_extensions.py`

Every production run used `configs/models.yaml`, the selected `configs/priors_{baseline,weak,tight}.yaml`, 12,000 iterations, 4,000 burn-in, thinning 5, and two chains. Run-specific seeds and complete resolved data specifications are stored in each run's `metadata.json` and `data_spec.json`.

- QCEW outputs: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/results/extensions/qcew_joint/runs`
- SEC outputs: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/results/extensions/sec_inverse_hhi/runs`
- Logs: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/results/extensions/logs`
- Comparisons: `/Users/satoshi/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC/results/extensions/comparison`

## Bugs found, fixes, and historical impact

1. The prior establishment experiment directly loaded an HP-filtered establishment cycle onto the firm cycle. Root cause: it represented establishments as a noisy deterministic measurement of `Nhat`. It is now explicitly deprecated and excluded from production; joint N/E states replace it. Historical legacy-establishment results are conceptually affected and should not be described as the requested correlated-innovation model. The annual firm baseline is unaffected.
2. YAML date values in cell-specific sample windows remained `datetime.date` after overriding already-normalized defaults. Root cause: normalization occurred before the per-cell merge. The merged window is now normalized, preventing JSON serialization failures in saved run specifications. Numerical posterior content is unaffected; affected runs would fail while writing artifacts.
3. Production scripts and tests required path environment variables despite the resolvable same-name Dropbox layout, and the report result link could remain absent. Root cause: central path resolution had no discovery fallback. The central resolver and script bootstrap now discover both roots, and the intended `results` link is ignored as generated output. This affects portability/report loading, not posterior mathematics.
4. The prior converter did not pass the new E drift/AR/variance and N/E inverse-Wishart settings. Root cause: the internal prior map covered only the original N system. The mappings and all three prior files now include the joint parameters. No old baseline result uses these parameters.
5. SEC normalization was implicit. Root cause: code could accept a plausibly named HHI without checking its scale or precomputed inverse. A validator now enforces fraction-scale HHI and exact consistency of `hhi_10000` and `effective_firms`. The local file passes; no numerical correction to its values was needed.

No stale state, state/covariance index, variance/standard-deviation, missing-as-zero, Q4 mapping, double transform, or double `KAPPA_SCALE` bug was found in the active baseline call chain. Synthetic tests exercise the new paths explicitly.

## Files changed

- `src/nkpc_hsa/gibbs/common/joint_ffbs.py`: six-state system construction and exact joint FFBS.
- `src/nkpc_hsa/gibbs/joint_ne.py`: joint steady/const-theta Gibbs kernel and correlated cycle covariance updates.
- `src/nkpc_hsa/gibbs/hsa_steady/model.py`, `hsa_const_theta/model.py`: dispatch to the joint kernel when quarterly E levels are supplied.
- `src/nkpc_hsa/inference/wrappers.py`: joint-E preparation/dispatch, SEC quarterly-observed support, metadata, and reporting aliases.
- `src/nkpc_hsa/dataprep/qcew.py`: QCEW schema/filter/frequency/duplicate validation and merge.
- `src/nkpc_hsa/dataprep/sec_hhi.py`: HHI scale validation and inverse-HHI merge.
- `src/nkpc_hsa/dataprep/competition.py`: explicit quarterly-observed competition measurement.
- `src/nkpc_hsa/models/common.py`, `configs/priors_*.yaml`: joint E and covariance priors.
- `configs/models.yaml`: new QCEW joint specification and deprecation of legacy direct loading.
- `src/nkpc_hsa/config.py`: merged sample-date normalization.
- `src/nkpc_hsa/paths.py`, `scripts/_bootstrap.py`, `.gitignore`: centralized path discovery and generated-result-link handling.
- `src/nkpc_hsa/inference/diagnostics.py`: N/E scalar and path diagnostics.
- `scripts/15_build_extension_data.py`: distinct model-ready data builds.
- `scripts/16_estimate_extensions.py`: current production-grid extension runner.
- `scripts/17_compare_extensions.py`: reproducible A/B/C and grid convergence outputs.
- `scripts/download_qcew_establishments.py`: official-archive fallback utility used only because no local QCEW source was discovered.
- Tests: joint state ordering/covariance, missing annual N with quarterly E/inflation, joint FFBS draw/storage, independent E centering, QCEW schema, SEC normalization, prior wiring, JSON dates, and path discovery.

## Validation

- Full test suite without either path environment variable: **139 passed**, 0 failed, 7 warnings, 87.60 seconds.
- `python -m compileall -q src scripts`: passed.
- `git diff --check`: passed.

The warnings are environment/numerical warnings already surfaced by pandas/NumPy/ArviZ tests; none is a failed assertion or nonfinite posterior result.

## Final assessment

Quarterly QCEW establishments are observed quarterly; firm counts are still observed only at annual Q4. The only cross-series channel is correlated cycle innovations. That correlation is near zero with a broad interval, and the firm-cycle posterior becomes less precise and substantially harder to mix. The evidence therefore does not support a claim that QCEW genuinely improves inference about quarterly `Nhat` in this specification.

SEC inverse HHI materially changes `delta` and `kappa_t`, but the available SEC sample is later and much shorter. The result should be reported as strong sensitivity to the competition measure/sample combination, not as an apples-to-apples causal comparison with the annual firm-count baseline.
