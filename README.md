# NKPC HSA MCMC

Bayesian state-space MCMC workflows for CES and Matsuyama-Fujiwara /
Fujiwara-Matsuyama HSA New Keynesian Phillips Curve specifications.

The canonical workflow is script-based. Notebooks are kept only for exploration.

---

## Quick Start

```bash
cd /path/to/NKPC_HSA_MCMC   # ← プロジェクトルートに移動（以降のコマンドはここから実行）
python -m pip install -e .   # 初回のみ：パッケージをローカルにインストール
```

動作確認（スモークテスト、数秒で完了）:

```bash
python scripts/01_build_data.py
python scripts/02_estimate_models.py --quick
```

---

## Full Pipeline（推定からレポートまで）

プロジェクトルートから下記8ステップを順に実行する。
4モデル（CES / hsa_steady / hsa_dynamic / hsa_full）× 5データ系列 = 計20本のギブスサンプリング
（`n_iter=12000`、バーンイン`4000`、チェーン数`2`）。

```bash
cd /Users/satoshi/GitHub/NKPC_HSA_MCMC   # ← 最初に必ずここへ移動

# 1. データ構築（data/processed/model_ready.csv を生成）
python scripts/01_build_data.py

# 2. フル推定（最も時間がかかる：20本のGibbs sampling）
python scripts/02_estimate_models.py \
    --config configs/models.yaml \
    --priors configs/priors_baseline.yaml

# 2b. 制約付き推定（kappa >= 0 制約あり版：reportで制約なし版と比較するため）
python scripts/02_estimate_models.py \
    --config configs/models.yaml \
    --priors configs/priors_baseline.yaml \
    --positive kappa

# 時変kappaモデルで全時点の kappa_t >= 0 を課す版
python scripts/02_estimate_models.py \
    --config configs/models.yaml \
    --priors configs/priors_baseline.yaml \
    --positive kappa_t

# 3. MCMC収束診断（R-hat, ESS を results/diagnostics/ に出力）
python scripts/03_run_diagnostics.py

# 4. 事前分布ロバストネス（baseline / weak / tight の3種で hsa_steady を再推定）
python scripts/04_prior_robustness.py

# 5. サブサンプル期間ロバストネス（configs/periods.yaml の各期間で再推定）
python scripts/05_period_robustness.py

# 6. モデル比較（SDDR・予測スコアを results/model_comparison/ に出力）
python scripts/06_model_comparison.py

# 7. 表・図の生成（results/tables/, results/figures/ に出力）
python scripts/07_make_tables_figures.py

# 8. レポートコンパイル（results/report/main.pdf を生成）
python scripts/08_compile_report.py
```

推定結果は `results/runs/<model>_<data_spec>_<prior>_<timestamp>/posterior.nc` に保存される。

特定データ系列のみ推定する場合:

```bash
python scripts/02_estimate_models.py --data-spec inv_markup        # 逆マークアップのみ
python scripts/02_estimate_models.py --data-spec output_gap_bn     # 産出ギャップのみ
python scripts/02_estimate_models.py --data-spec output_gap_hp     # HPフィルター産出ギャップのみ
python scripts/02_estimate_models.py --data-spec labor_share_gap_hp # HPフィルター労働分配率ギャップのみ
python scripts/02_estimate_models.py --data-spec unemployment_gap  # 失業率ギャップのみ
```

---

## Reproducible Pipeline（フルパイプライン）

```bash
python scripts/01_build_data.py
python scripts/02_estimate_models.py --config configs/models.yaml
python scripts/02_estimate_models.py --config configs/models.yaml --positive kappa
python scripts/03_run_diagnostics.py
python scripts/04_prior_robustness.py
python scripts/05_period_robustness.py
python scripts/06_model_comparison.py
python scripts/07_make_tables_figures.py
python scripts/08_compile_report.py
```

スモークテスト版:

```bash
python scripts/01_build_data.py
python scripts/02_estimate_models.py --quick
python scripts/03_run_diagnostics.py
python scripts/04_prior_robustness.py --quick
python scripts/05_period_robustness.py --quick
python scripts/06_model_comparison.py
python scripts/07_make_tables_figures.py
python scripts/08_compile_report.py
```

## Structure

- `src/nkpc_hsa/`: reusable package code.
- `scripts/`: canonical pipeline entry points.
- `configs/`: model, path, and prior YAML files.
- `data/raw/`: raw input files, grouped by source/topic.
- `data/processed/`: generated model-ready datasets.
- `results/runs/`: one folder per estimation run with posterior draws, configs, and metadata.
- `results/tables/`, `results/figures/`, `results/diagnostics/`, `results/prior_robustness/`, `results/period_robustness/`, `results/model_comparison/`, `results/report/`: generated outputs.
- `reports/`: LaTeX report sources.
- `notebooks/`: exploration only.
- `analysis/gibbs/func_gibbs/`: legacy Gibbs sampler backend used by the current adapters.
- `archive/`: old notebooks, old scripts, and old generated outputs retained for reference.

Raw data should never be overwritten. Processed data and estimation outputs are
regenerated under `data/processed/` and `results/`. Legacy outputs are retained
under `archive/legacy_results/` and are not used by the pipeline.

## Unit Conventions

Kappa-related priors in YAML files are specified in economic/physical units.
Some Gibbs samplers use internal parameters multiplied by `KAPPA_SCALE = 100`
because the regression column is divided by 100. Posterior draws saved to
`InferenceData`, exported tables, figures, SDDR outputs, and Chib marginal
likelihood inputs are in physical units.

The default HSA competition aggregator transform is:

```text
N_model = (100 * log(N_level) - sample_mean(100 * log(N_level))) / 10
```

If a supplied `N` series is already transformed, call the wrapper with
`n_transform="identity"` and verify that the run metadata records this.
Under the default transform, one unit of `N_model`, `Nhat`, or `Nbar` is a
ten-log-point movement around the sample mean.

The default HSA dynamic covariance convention is `e_zeta_only`: the sampler
allows correlation between the NKPC shock `e_t` and output-gap shock `zeta_t`
only. Set `covariance_structure: diagonal` or `covariance_structure: full` in
`configs/models.yaml` only when that alternative is intentional and documented.

Reported `delta`, `theta`, `theta_0`, and `gamma` are already in the same
ten-log-point units used during estimation. Do not multiply them by 10 again.

Because the transformed `N` series is measured in ten-log-point units, the
inverse-gamma hyperparameters for the N-state shock variances (`a_u`/`b_u`,
`a_eps`/`b_eps`) and the N measurement-error variance (`a_N`/`b_N`) must be
scaled to that unit: the observed quarterly variance of the transformed series
is roughly 0.01, so the prior scales sit in that decade (see comments in
`configs/priors_baseline.yaml`). The same applies to the `u` and `eps` entries
of `S_Sigma` for the `hsa_dynamic` inverse-Wishart prior. Earlier IG(2, 2)
settings forced these variances two orders of magnitude too large, which made
the `Nbar`/`Nhat` decomposition spuriously volatile and attenuated `delta`.

`hsa_full` now includes an explicit N measurement error
(`N_obs_t = Nhat_t + Nbar_t + nu_t`, `nu_t ~ N(0, sigma_N^2)`) like the other
HSA models. Conditional on this measurement equation, its two-block state
sampler (`Nhat | Nbar`, then `Nbar | Nhat`) is an exact Gibbs step; the old
`target_scale`/`rw_scale` pseudo measurement variances have been removed.

Real-activity specifications are configured in `configs/models.yaml`.
`output_gap_BN` is the Baxter-King/BN-filtered cycle supplied in the raw output
file. `output_gap_HP` is generated by `scripts/01_build_data.py`: the HP filter
is applied to `100 * output`, where `output` is the log real-output level from
the legacy raw-data builder. The resulting HP cycle is therefore in the same
100-log-point unit as `output_gap_BN`.
`labor_share_gap_HP` is generated from `data/raw/laborshare/PRS85006173.csv` by
applying the HP filter to `100 * log(labor_share_index)`. It is treated as an
additional real-activity proxy in the same script pipeline.
`kappa_0` and `theta_0` are intercepts at average competition because the
default `N` transform is centered.

## Coefficient Sign Constraints

By default, regression coefficients are sampled from their unconstrained
Gaussian conditional posterior, except for the AR(2) stationarity restriction
controlled by `enforce_stationary` inside the Gibbs samplers. To impose hard
coefficient restrictions, set `defaults.coefficient_constraints` in
`configs/models.yaml` or pass `--positive` to the estimation scripts.

Example smoke run with nonnegative kappa and theta:

```bash
python scripts/02_estimate_models.py --quick --positive kappa,theta
python scripts/02_estimate_models.py --quick --positive kappa_t
```

Example YAML:

```yaml
defaults:
  coefficient_constraints:
    enabled: true
    max_tries: 1000
    positive: [kappa, theta, kappa_t]
    bounds:
      alpha: [0.0, 1.0]
```

Bounds for `kappa`, `kappa_0`, `kappa_t`, and `delta` are specified in physical
units and converted internally by the wrapper. Bounds for `theta`, `theta_0`,
`gamma`, `alpha`, `rho_1`, and `rho_2` are used as written. In HSA steady/full
models, `kappa_t` is a path restriction: each candidate draw must satisfy the
bound for every sampled period of \(\kappa_t=\kappa_0+\delta\bar N_t\). A
generic `--positive kappa` also applies to the time-varying kappa path in
models that do not have a scalar `kappa`. These constraints define a different
posterior with hard prior support; they should be reported as a restricted
robustness specification rather than silently mixed with the unrestricted
baseline.

## Robustness And Comparison

Prior robustness is controlled by `configs/priors_baseline.yaml`,
`configs/priors_weak.yaml`, and `configs/priors_tight.yaml`. Data-period
robustness is controlled by `configs/periods.yaml`; add a new named period
there and rerun `scripts/05_period_robustness.py`.

Model comparison reports SDDR for nested restrictions when posterior draws are
available, posterior predictive scores when data can be matched, and Chib
marginal likelihood when the legacy ordinate calculation is compatible with the
model family and supplied data. For `hsa_full`, the Chib output is a
conditional Chib calculation for the inflation equation, conditioning on
posterior mean `Nhat` and `Nbar` paths, because the full observation equation
contains the nonlinear term `gamma * Nbar_t * Nhat_t`. WAIC/LOO are not used as
primary criteria for these latent-state models.

## Adding A Model Variant

1. Add the model implementation or adapter under `src/nkpc_hsa/models/`.
2. Expose it in `src/nkpc_hsa/inference/wrappers.py`.
3. Specify the model name in `configs/models.yaml`.
4. Add focused tests for any new unit conversions or transforms.

## Adding A Prior Specification

Create a new YAML file under `configs/`. Keep kappa, kappa_0, and delta in
physical units; the wrapper converts them to sampler-internal units.

## Report

`scripts/08_compile_report.py` writes `reports/main.tex` and compiles
`results/report/main.pdf`. It also writes one PDF per configured data
specification, such as `results/report/inv_markup.pdf`,
`results/report/output_gap_bn.pdf`, `results/report/output_gap_hp.pdf`, and
`results/report/labor_share_gap_hp.pdf`, and `results/report/unemployment_gap.pdf`.
Table fragments are read from `results/tables/` and figures from
`results/figures/`. Baseline model results, prior-set robustness, and
sample-period robustness are displayed in separate tables and figures.
