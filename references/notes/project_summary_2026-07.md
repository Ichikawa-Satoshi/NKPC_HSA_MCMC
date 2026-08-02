# プロジェクト現状まとめノート（2026年7月30日時点）

## 1. このプロジェクトは何をやっているのか

**市場の競争度（企業数 N）がニューケインジアン・フィリップス曲線（NKPC）の傾き
を時変させるか** を、ベイズ状態空間 MCMC（Gibbs / FFBS）で推定するプロジェクト。

理論的背景は Matsuyama–Fujiwara / Fujiwara–Matsuyama の HSA（Herfindahl-Style
Aggregator）NKPC。全モデル共通の観測方程式：

```
pi_t = alpha*pi_{t-1} + (1-alpha)*E_t pi_{t+1} + kappa_t*x_t - theta_t*Nhat_t + e_t
```

競争度は `N_obs_t = Nbar_t + Nhat_t + nu_t` に分解（`Nbar`=RWトレンド、
`Nhat`=定常AR(2)ギャップ、`nu`=測定誤差）。中心的な問い：

- **傾き `kappa_t = kappa_0 + delta * Nbar_t`** が競争度トレンドで動くか（`delta`の符号・有意性）。
- 競争度が上がる（N が増える）と NKPC はフラット化するのか。

### 5つのモデル

| モデル | kappa_t | theta_t | 位置づけ |
|---|---|---|---|
| `ces` | 定数 | 0 | ベースライン（従来型NKPC） |
| `hsa_steady` | `kappa_0 + delta*Nbar_t` | 0 | **主力**。傾きがトレンド競争度で時変 |
| `hsa_dynamic` | 定数 | 定数 | Nhatチャネルのみ |
| `hsa_const_theta` | `kappa_0 + delta*Nbar_t` | 定数 | hsa_fullでgamma=0固定（2026-07追加） |
| `hsa_full` | `kappa_0 + delta*Nbar_t` | `theta_0 + gamma*Nbar_t` | フルモデル |

## 2. コード / パイプライン構成

- 正準パッケージ：`src/nkpc_hsa/`。Gibbsエンジンは `src/nkpc_hsa/gibbs/`
  （2026-07に `analysis/gibbs/func_gibbs/` から移設。旧履歴はタグ `pre-restructure`）。
- 入口：`src/nkpc_hsa/inference/wrappers.py::run_model`。
- パイプライン `scripts/01_…` → `10_build_html_report.py` を root から順に実行。
  ベースラインは **5モデル × 8データ仕様 = 40 run**（n_iter=12000, burn-in=4000, 2 chains）。
- 生成物は `results/` 配下（**全ツリーgit-ignore、スクリプトから再現可能**）。
- 成果物：PDF（`results/report/*.pdf`）、ブラウズ用 `results/report.html`、
  テーブル `results/tables/`、レビュー報告 `results/final_review_report.md`。

### 単位規約（重要・間違えやすい）
- N変換デフォルト：`(100*log(N) - 標本平均)/10`。1単位 = 10 log-point。
- `delta, theta, theta_0, gamma` は既に10-log-point単位。**テーブル/図で×10しない**。
- kappa系プライヤはYAMLでは物理単位、wrapperが内部で`KAPPA_SCALE=100`変換。
- N状態ショック/測定分散のプライヤは二乗10-log-point単位（≈0.01 decade付近を維持）。

## 3. 主要な実証結果（ベースライン, 1982Q1–2012Q4, T=124, quarterly_interpolated）

### ★ 中心的な結果：`delta > 0` は失業ギャップ仕様で頑健

`results/tables/coefficient_means.csv` からの現行 delta 事後値：

| 活動指標 (spec) | model | delta 事後平均 | sd | P(delta>0) |
|---|---|---|---|---|
| **unemployment_gap_core** | hsa_steady | **+0.0227** | 0.0075 | **1.000** |
| unemployment_gap_core | hsa_full | +0.0253 | 0.0090 | 0.998 |
| **unemployment_gap** (headline) | hsa_steady | **+0.0302** | 0.0109 | 0.997 |
| unemployment_gap | hsa_full | +0.0328 | 0.0139 | 0.989 |
| output_gap_bn_core | hsa_steady | +0.0106 | 0.0101 | 0.875（限界的） |
| output_gap_bn (headline) | hsa_steady | −0.0128 | 0.0144 | 0.177（**負**） |
| output_gap_hp | hsa_steady | −0.0024 | 0.0148 | 0.43（≈0） |
| labor_share_gap_hp | hsa_steady | +0.0088 | 0.0169 | 0.70 |
| inv_markup | hsa_steady | −0.0006 | 0.0188 | 0.48（≈0） |

**要点：**
1. **失業ギャップ仕様の delta>0 が論文の主張。** 弱/強プライヤ、seed、
   kappa_t>=0制約、steady/full、周波数（annual_q4含む）を通じて頑健（P≈1.0）。
   含意 `kappa_t` は標本期間で **0.16 → 0.00 にフラット化**（競争↑でNKPCフラット化）。
2. **BN出力ギャップ（headline CPI）だけ delta<0**。これはバグではなく、
   **2008Q3–2009Q4 の石油ショック起因のheadline CPI変動が原因**（下記§4）。
3. core CPI へ切り替えると **BN も符号反転して正**（+0.011, P=0.88）＝限界的だが方向一致。
   失業のdeltaは core でむしろシャープ化（sd 0.011→0.008、オイルノイズ除去）。
4. inv_markup / labor_share / HP は **インフレシグナルがほぼ無く**、delta≈0でプライヤ支配。

### kappa_0（平均競争度での切片＝曲線が生きているか）
- unemployment_gap_core: hsa_steady +0.051 (P=0.997) → 曲線は健在。
- output_gap_hp: +0.14 前後（P≈1.0）で最も大きい。
- inv_markup / labor_share: ≈0でプライヤ支配。

## 4. delta符号の食い違いの根本原因（2026-07 診断、確立済み）

FWL/Okun分解 + ベイズ両方で確認：

- BN delta = −0.046 の **100% が 2008Q3–2009Q4 の6四半期**由来。この6四半期を落とすと
  BN は **+0.088 (t≈2.1)** に反転し失業と符号一致。1983–86 は副次的（−0.003）。
- メカニズム：2008–09にGDPが労働市場より早く深く崩落（Okun残差−2.4）した時期に、
  石油がheadline CPIを±2以上振らせた → BNは「巨大なスラック×巨大なディスインフレ＝
  急な傾き」と最低Nbar期に読み、delta<0に。
- **core CPI へのLHS切替が最もクリーンな修正**（サンプラー変更不要）。
  pre-2008に限ってもBN deltaの負符号は消滅（−0.013→+0.002, P=0.55）。

## 5. 拡張・ロバストネスの現状

- **スコープ条件（標本期間）**：post-1988標本では失業フィリップス曲線が既に平均的にフラット（κ₀≈0）で
  傾きの変動余地が乏しく、delta≈0に弱まる。識別を担うのは1982–87のVolcker期。
  → **主結果は1980年代（Volcker期）を含む標本に固有**。論文では隠さずスコープ条件として明記する方針。
- annual_q4（N を Q4のみ観測、Q1–Q3欠測をKalman/FFBSで補完）：サンプラー側は
  検証済みで正しい。delta方向は不変。quarterly での theta/gamma の「有意性」は
  PCHIP補間アーティファクトで、annual_q4ではプライヤ支配に戻る（実質未識別）。
- 制約付き（kappa>=0, kappa_t>=0）は別事後・ハード台の頑健性仕様として別掲。

## 6. 既知の未解決事項 / 注意点

- **Chib周辺尤度が annual_q4 ブロックで無効**（欠測N非対応で log-ML が+1388等の異常値、
  BF=inf）。PDFの annual_q4 セクションのML欄に流入。**SDDRと予測スコアは無影響**。
  → 修正案：Chib Kalmanループに masked N_obs + isfinite分岐を入れるか、当該欄をブランク化。
- `hsa_const_theta` は Chib ML 未実装（意図的）。theta は hsa_full 比でシャープ化せず
  （gammaはもともとプライヤ支配、外しても情報増えない）。BN_core の theta で R̂=1.28 の
  収束不良セルあり（長チェーンなしで信用しない）。
- hsa_full の weak/tight/annual_q4 は R-hat 1.1–1.3 の収束不良。**hsa_steady が clean workhorse。**
- gamma の回帰子 `Nhat*Nbar` は `Nhat` と ~98% 共線 → gamma は基本未識別。

## 7. 現在の作業（未コミット差分）

- 直近コミット：HTMLレポートのブロック別図タブ選択、core-CPI仕様の昇格、
  hsa_const_theta追加、リポジトリ再構成（単一コードツリー化）。

## 8. 「使える結果」ショートリスト（論文に載せられるもの）

1. **失業ギャップ×core CPIの delta>0（P≈1.0, +0.023）＝主結果**。競争度↑でNKPCフラット化。
   実装含意 kappa_t: 0.12→−0.005（P(decline)=1.0）。
2. headline→core の切替でBN出力ギャップの符号が反転（オイル汚染の直接証拠）＝
   識別ロジックの説得的なストーリー。
3. delta>0 の頑健性表（プライヤ×周波数×制約×サブ標本）。
4. スコープ条件：post-1988（1980年代を除く標本）では消える（正直に報告する材料も揃っている）。
5. N分解 `Nbar/Nhat` はどの仕様でもよく識別（RMSE≤0.01）— ただし
   「状態がきれいに見える」ことは kappa/delta の識別を意味しない、という注意も込みで。
