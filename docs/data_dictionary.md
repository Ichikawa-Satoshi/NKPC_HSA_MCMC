# Data dictionary

> **Inflation-observation migration.** The processed data retains the legacy
> four-quarter columns and now also creates non-annualized Q/Q percentage-point
> columns (`pi_*_qoq` plus one-quarter lags). `Epi_qoq=Epi/4` remains a legacy
> quarterly-rate equivalent of the Cleveland Fed one-year source, not a one-quarter
> forecast. The SPF loaders now separately identify genuine one-quarter-ahead and
> forward four-quarter/YoY forecasts.
> F0/U compare Q/Q and direct 4Q YoY in separate likelihoods; R1--R3 use Q/Q.
> See `docs/theory_model_migration.md`.

Source of truth: `src/nkpc_hsa/dataprep/func_data_build.py`, `src/nkpc_hsa/dataprep/build.py`,
`src/nkpc_hsa/dataprep/transforms.py`, `src/nkpc_hsa/dataprep/competition.py`, `configs/models.yaml`.
Generated dataset: `data/processed/model_ready.csv` (454 rows; the production-report sample is
the complete-case subset, **T = 124**, 1982Q1–2012Q4). The separate establishment experiment
uses **T = 79**, 1993Q2–2012Q4.

Anything not recoverable from code/config is marked **UNVERIFIED**.

---

## How a data spec becomes sampler input

`configs/models.yaml → data_specs.<name>` names six columns. `_coerce_model_data`
(`src/nkpc_hsa/inference/wrappers.py:100-125`) selects them and applies `.dropna()` **jointly**,
so the estimation sample is the complete-case intersection of all six series:

```python
sample = _apply_sample_window(data[[cols[k] for k in required]], spec).dropna()
```

That is why every production-report specification has T = 124 even though
`model_ready.csv` has 454 rows. A data spec may override the window; the establishment
experiment does so explicitly.

---

## Inflation measures

### π_t — headline CPI inflation

| | |
|---|---|
| **Symbol** | `π_t` (headline) |
| **Code name** | `pi_cpi`; lag `pi_cpi_prev` |
| **Source** | `data/raw/inflation/CPIAUCSL.csv` (FRED CPIAUCSL, monthly) |
| **Construction** | monthly → quarterly mean (`resample_quarterly_mean`), then `yoy_pct`: `100*(x_t/x_{t-4} − 1)` — `func_data_build.py:22-25, 258` |
| **Frequency** | quarterly, four-quarter change |
| **Units** | annual percent |
| **Lag** | `pi_cpi_prev = pi_cpi.shift(1)` — `func_data_build.py:252` |
| **Missing** | dropped jointly with the rest of the spec |
| **Used in** | `unemployment_gap`, `output_gap_bn`, `output_gap_hp`, `inv_markup`, `labor_share_gap_hp` |
| **Economic meaning** | headline consumer-price inflation, the left-hand side of the NKPC |

⚠️ **Overlapping observations.** `π_t` is a *four-quarter* change sampled *quarterly*, so `π_t`
and `π_{t-1}` share three of four quarters by construction. Every model's inflation-equation
likelihood nevertheless treats the residual `η_t` as i.i.d. Gaussian. This is a property of the
data construction, not of the samplers; it is documented here because it is invisible from the
sampler code alone.

### π_t — core CPI inflation

Identical pipeline **except the transform**: `log_yoy`, i.e. `100*(log x_t − log x_{t-4})`
(`func_data_build.py:27-29, 261`). Code name `pi_cpi_core`, lag `pi_cpi_core_prev`. Source
`CPILFESL.csv`. Used in `unemployment_gap_core`, `output_gap_bn_core`, `output_gap_hp_core`.

⚠️ **Discrepancy vs. headline/PPI**: headline CPI and PPI use `pct_yoy` (simple percent change)
while core CPI uses `log_yoy` (log difference). Second-order at these rates, but the three price
indices are therefore *not* constructed identically, in a comparison whose purpose is the price
index. Neither the report nor the config remarks on this.

### π_t — PPI inflation

`pi_ppi`, lag `pi_ppi_prev`, from `PPIACO.csv`, transform `pct_yoy` (`func_data_build.py:264`).
Used in `unemployment_gap_ppi`, `output_gap_bn_ppi`, `output_gap_hp_ppi`.

---

## Expected inflation

### SPF: genuine one-quarter-ahead GDP-price inflation

| | |
|---|---|
| **Source file** | `data/raw/inflation/Median_PGDP_Growth.xlsx`, Philadelphia Fed SPF annualized percent change of median responses |
| **Source field** | `DPGDP3`; a row dated at survey quarter `t` forecasts GDP-price inflation in `t+1` (`DPGDP2` is the current-quarter forecast) |
| **Published units** | discrete annualized percentage points |
| **Processed columns** | `Epi_spf_gdp_1q_ahead_ann_pct` (published units); `Epi_spf_gdp_1q_ahead_ann_log` (annualized log); `Epi_spf_gdp_1q_ahead_qoq_pct` (exact nonannualized Q/Q) |
| **Conversion** | annualized log: `100*log(1+r/100)`; Q/Q: `100*expm1(log1p(r/100)/4)` |
| **Loader** | `load_spf_quarter_ahead_expectations` |

The exact compounding conversion is deliberate: dividing the published annualized
forecast by four is only an approximation and does not change the forecast horizon.
The median statistic matches the construction of the companion SPF one-year-ahead
series.

### SPF: forward four-quarter/YoY inflation

| | |
|---|---|
| **Source file** | `data/raw/inflation/SPF_Inflation_Expectation.xlsx` |
| **Source fields** | `INFPGDP1YR`, `INFCPI1YR` |
| **Horizon** | average inflation over the four quarters following the survey quarter; not interchangeable with `DPGDP3` |
| **Processed columns** | `Epi_spf_gdp_yoy_1y_ahead`, `Epi_spf_cpi_yoy_1y_ahead`, and corresponding `_log` columns |
| **Loader** | `load_spf_yoy_expectations` |

`Epi_spf_gdp` and `Epi_spf_cpi` remain backward-compatible aliases for the two
forward-four-quarter level-rate columns. New specifications should use the explicit
horizon names.

### Cleveland Fed legacy expectation

| | |
|---|---|
| **Symbol** | `E_t π_{t+1}` |
| **Code name** | `Epi` |
| **Source** | `data/raw/inflation/Clev_Fed_Inflation_Expectation.csv` — Cleveland Fed inflation expectations |
| **Construction** | `epi["Epi"] = epi[" Epi"] * 100`, monthly → `resample("QE").mean()` — `func_data_build.py:155-160` |
| **Units** | annual percent |
| **Horizon** | **UNVERIFIED** which maturity is in the ` Epi` column. The model equation calls it `E_t π_{t+1}`; with four-quarter inflation on the LHS, a one-year-ahead series is the coherent match, but the column is not labelled in code and the raw file is not self-documenting. |
| **Used in** | every model and every data spec — the same series in all of them |
| **Note** | The current configured legacy data specs continue to use this series. The SPF columns are present in `model_ready.csv` for the reorganized estimation cells. |

⚠️ **Deliberate mismatch, disclosed in the report**: the PPI and core-CPI specs pair their price
index with this *same* headline-oriented expectation series, because no PPI expectation is
available. Report §2 states this.

---

## Activity / slack measures (`x_t`)

### Negative unemployment gap

| | |
|---|---|
| **Symbol** | `x_t = u*_t − u_t` |
| **Code name** | `unemp_gap`, lag `unemp_gap_prev` |
| **Source** | `data/raw/unemp_gap/NROU.csv` (CBO natural rate) and `UNRATE.csv` (unemployment, **seasonally adjusted**) |
| **Construction** | `tt_gap["unemp_gap"] = tt_gap["NROU"] - tt_gap["UNRATE"]` — `func_data_build.py:231`, after `resample("QE").mean()` |
| **Seasonal adjustment** | Must be `UNRATE`, not `UNRATENSA`. `NROU` is a smooth trend with no seasonal of its own, so differencing it against the unadjusted rate leaves the unemployment seasonal in the gap: over 1982–2012 the NSA version has a 0.76-point peak-to-trough quarterly swing (F = 31.5, p = 4.5e-15), correlating −0.9998 with the raw `UNRATENSA` seasonal — nothing cancels. Every inflation series here is a four-quarter change, so the left-hand side carries no seasonal to match it and that component is pure measurement error in `x_t`, attenuating `κ`. The SA series leaves a swing of 0.02 (F = 0.03). `UNRATENSA.csv` is retained in `data/raw/` but is no longer read. |
| **Units** | percentage points of unemployment |
| **Sign convention** | **`u* − u`, i.e. the NEGATIVE unemployment gap.** Positive in booms, negative in slumps (≈ −4 in 2009). Co-moves positively with output gaps. A **positive** `κ` is therefore a conventionally-signed downward-sloping Phillips curve. |
| **Used in** | `unemployment_gap`, `unemployment_gap_core`, `unemployment_gap_ppi` (+ TNIC variants) |

### BN output gap

`output_gap_BN`, lag `output_gap_BN_prev`. Source `data/raw/output_gap/BN_filter_GDPC1_quaterly.csv`,
column `cycle` — a pre-computed Beveridge–Nelson decomposition of real GDP
(`func_data_build.py:226-229`). Units: 100 log points. The BN filtering itself is **UNVERIFIED**
(done outside this repository; only its output is read).

### HP output gap

`output_gap_HP`, lag `output_gap_HP_prev`. Constructed **inside** this repo:
`add_hp_output_gap` (`src/nkpc_hsa/dataprep/build.py:49-70`) applies a λ=1600 HP filter to
`100 * output`, where `output = log(GDPC1_original_series * 0.01)`
(`func_data_build.py:229`). The ×100 is explicit so HP and BN share 100-log-point units. The
filter runs separately on each contiguous finite block (`hp_filter_series`, `build.py:26-47`), so
raw gaps are not silently interpolated.

### HP labor-share gap

`labor_share_gap_HP`. From `data/raw/laborshare/PRS85006173.csv`; the cycle is taken from
`100 * log(index)` then HP-filtered at λ=1600 — `build.py:73-107`. Used in `labor_share_gap_hp`.

### Inverse markup (real marginal cost proxy)

`markup_BN_inv`, lag `markup_BN_inv_prev`. From `data/raw/markup/BN_markup_inv.csv`, column
`cycle` (`func_data_build.py:200-204`). Used **only** in the `inv_markup` spec, which the report
uses for the CES–HSA slope-bias diagnostic because it is the closest empirical counterpart to the
theory's real-marginal-cost regressor. The upstream BN decomposition is **UNVERIFIED**.
A related series `markup` (`mu_bus` from `nekarda_ramey_markups.xlsx`) is loaded but not used by
any configured spec.

---

## Firm count / competition

| | |
|---|---|
| **Symbol** | `N^obs_t` (transformed); raw level `N` |
| **Code name** | `N_Gustavo` |
| **Source** | `data/raw/competition/BN_N_Gustavo_26.csv`, column `original_series` |
| **Meaning** | inverse HHI of U.S. listed firms — an effective firm count |
| **Native frequency** | **annual** |
| **Raw range in sample** | 5.96 – 9.68 (after PCHIP; declining over the sample) |
| **Used in** | every HSA model. **Not** used by CES. |

Two observation schemes, selected by `configs/models.yaml → defaults.competition_measurement.frequency`.

**`configs/models.yaml` declares `frequency: annual_q4`**, and the library default
(`DEFAULT_COMPETITION_MEASUREMENT`) and `scripts/13_estimate_cpi_ppi_report.py` resolve to
the same value, so a caller that omits the argument gets the main design.
`tests/test_observation_design_default.py` pins that agreement.

⚠️ Code that *reads* a saved run's metadata still falls back to `quarterly_interpolated`
when the field is absent, because runs written before the field existed were interpolated.
That asymmetry is deliberate and is also pinned by the test.

Both designs are estimated. The report presents the mixed-frequency one as primary
(paper §4) and the interpolated one as a comparison (paper §7), because the interpolation
is not innocuous: it drives σ_N to a third of its mixed-frequency value, forces the AR(2)
cycle to a near-unit root, and thereby opens a near-exact ridge between `Nbar` and `Nhat`
(posterior correlation −0.9996 versus +0.13). See `docs/estimation_specification.md` §N.

**1. `quarterly_interpolated` (PCHIP) — reported as the comparison.**
`annual_to_quarterly_pchip` (`func_data_build.py:41-70`) fits a `scipy` `PchipInterpolator` to the
annual levels and evaluates it at quarter-ends. The result is treated as **observed every
quarter**: `N^obs_t` is finite for all 124 quarters.

**2. `annual_q4` — the mixed-frequency scheme, and the report's main design.**
`build_competition_observation` (`competition.py:131-141`) places the annual value in that year's
Q4 and leaves Q1–Q3 as `np.nan`:

```python
out = np.full(len(q_index), np.nan, dtype=float)
for i, period in enumerate(q_index):
    if int(period.quarter) == 4 and int(period.year) in by_year:
        out[i] = float(by_year[int(period.year)])
```

31 finite observations out of 124. The samplers treat `nan` as genuinely missing: the firm-count
observation row is dropped for that quarter and `σ_N²` uses only the finite residuals
(`finite_N_residuals`, `src/nkpc_hsa/gibbs/common/competition.py:12-21`).

**Centering consistency.** Under `annual_q4` the annual values are centered on the **PCHIP
quarterly** mean, not their own, so coefficient units stay comparable across the two schemes —
`_transform_annual_competition_like_quarterly` (`wrappers.py:167-179`):

```python
center = float(np.mean(100.0 * np.log(reference)))
return (100.0 * np.log(annual_values) - center) / 10.0
```

**Initialisation only.** `initial_competition_path` (`common/competition.py:23-36`) linearly
interpolates missing `N^obs` — used *solely* to seed the state path at iteration 0, never in a
likelihood.

### Alternative competition series (configured but not in `run_data_specs`)

`N_TNIC` (from `HHI_TNIC`) feeds `unemployment_gap_tnic` and `unemployment_gap_core_tnic`. These
specs exist in `configs/models.yaml` but are **absent from `run_data_specs`**, so they are not
estimated by the production pipeline.

### Quarterly establishment series (experimental)

| Column | Construction | Units / range |
|---|---|---|
| `establishment_births` | BED quarterly establishment births, series `...120007LQ5` | establishments; source values multiplied by 1,000 |
| `establishment_deaths` | BED quarterly establishment deaths, series `...120008LQ5` | establishments; source values multiplied by 1,000 |
| `establishment_net_entry` | births minus deaths | establishments per quarter |
| `establishment_stock` | 1993 BDS `ESTAB` anchor plus cumulative quarterly net entry | establishments |

The annual anchor is 5,682,098 establishments. Treating it as the 1993Q1 level,
1993Q2 births of 181,000 minus deaths of 160,000 produce a first quarter-end stock
of 5,703,098. Both flows are finite from 1993Q2 through 2023Q3; the configured
estimation window stops at 2012Q4.

Inside `_coerce_model_data`, the 79-quarter stock is transformed as
`(100*log(E) - sample mean)/10` and HP-filtered with `lambda=1600`. The resulting
`Ehat` is the observation in `Ehat_obs_t = lambda_E*Nhat_t + omega_t` for the
HSA const-theta experiment. It is not a production-report input and is not added
to `run_data_specs`.

---

## Auxiliary series loaded but unused by production specs

| Series | Loaded at | Status |
|---|---|---|
| `Epi_spf_gdp_1q_ahead_*`, `Epi_spf_*_yoy_1y_ahead*` | SPF loaders in `func_data_build.py` | available for reorganized estimation cells; legacy aliases retained |
| `pi_pce`, `pi_pce_core` | `func_data_build.py:262-263` | no spec references them |
| `oil` | `func_data_build.py:244` | no spec references them |
| `markup` (`mu_bus`) | `func_data_build.py:197` | only `markup_BN_inv` is used |
| `N_TNIC`, `HHI_TNIC` | competition loader | specs exist, not in `run_data_specs` |

No shock series is constructed from an auxiliary series: `ζ_t` is built inside the samplers from
`x_t` and `φ_1` (see `docs/estimation_specification.md` §C).
