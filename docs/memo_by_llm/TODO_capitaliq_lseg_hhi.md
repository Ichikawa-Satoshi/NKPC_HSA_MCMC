# TODO: Capital IQ / LSEG Datastream による四半期 HHI の作成

目的は、現在の年次 `N_Gustavo`（米国上場企業の逆 HHI）と経済概念を揃えた、長期の四半期 competition measure を構築することである。QCEW 事業所数とは別の「上場企業・売上集中度」ルートとして扱い、最終的に既存の `sec_hhi_quarterly.csv` と同じ検証・推定パイプラインへ接続する。

## 方針決定（2026-08-12）

- [x] Capital IQ / LSEGの検証済み四半期HHIを追加するまで、`N_Gustavo`単独の混合頻度状態空間モデルを暫定主仕様とする。
- [x] `N_Gustavo`は年次Q4だけで観測し、Q1--Q3は欠測として四半期状態を推定する。年次回帰と決定論的な四半期補間は主仕様に用いない。
- [x] 暫定主仕様ではQCEWとSEC HHIを競争状態の推定に入れず、inflationを状態推定から切り離したmodular cutを用いる。
- [x] `N_Gustavo`単独の状態空間本番推定（1982Q1--2013Q4、年次観測32点、四半期状態128点）と78本のinflation式を20,000反復・warmup 5,000・4チェーンで実行した。
- [x] Cell 1の`theta`は基準E1・persistent AR(1)でほぼゼロとなり、固定`phi=0.5/0.9`およびno-drift仕様でも95%区間はゼロを含んだ。
- [x] 1期前インフレを外し、persistent AR(1) errorを残した仕様を、価格5系列、activity/slack 4系列、E0/SLOW/E1/E2、fast-state timing 6通りの計200仕様で本番検証した。
- [x] no-lag E1では20/20の`theta_0`が収束基準を通過し、16/20で事後SD／事前SDが0.75以下となったが、95%区間がゼロを除外した組合せは0/20だった。`kappa_1`も0/20、E2の`gamma`も0/20だった。
- [x] core PCEでは全activity指標で`theta_0`が正方向（事後符号確率0.87--0.91）かつE1のconditional WAICが最良となったが、全区間がゼロを含むため、示唆的な価格系列感度としてのみ保存し、主結果へ昇格させない。
- [x] no-lag E1のresidual AR(1)は平均0.25、組合せ間範囲0.12--0.37であり、persistent errorは持続性を吸収するが単位根付近には張り付かなかった。
- [x] HSA理論から直接は出ないslow-level nuisance `psi*(qbar-q0)`をゼロ固定し、同じ200仕様を4チェーン・本番state posteriorで再推定した。
- [x] `psi=0`でもE1の`theta_0`、`kappa_1`、E2の`gamma`はすべて0/20で95%区間がゼロを含んだ。`theta_0`は`psi`ありからの平均値変化が絶対値中央値0.011、最大0.042で、20/20の符号が維持されたため、`psi`による`theta`吸収仮説は支持されなかった。
- [x] `psi=0`でもcore PCEの`theta_0`は全activityで正方向（正の事後確率0.84--0.91）かつE1が4/4でconditional WAIC最良だったが、全区間がゼロを含むため示唆的結果に留める。
- [x] 基準状態モデルでは`phi_q`とslow-state分散の収束ゲートが失敗したため、年次`N_Gustavo`だけでは四半期slow/fast分解を自由には識別できないことを暫定主仕様の明示的限界とする。
- [x] 現在の `QCEW establishments + N_Gustavo inverse HHI` 共通因子を主仕様から外す。
- [x] 現行のQCEW共同モデルとmeasurement-only推定は既存診断として保存するが、今後の必須工程にはしない。
- [x] `SEC/Compustat競争ルート` と `BDS/QCEW企業動態ルート` の二本立ては採用しない。
- [x] 独立したmeasurement-only推定と `R_q` ゲートは今後の必須工程から外す。
- [x] 既存SEC四半期HHIを代替入力に、観測HHI直接モデル297仕様と回復シミュレーションを実行した。
- [x] 観測HHI直接モデルではQCEW共通因子のMCMC問題が消える一方、Cell 1の短標本・共線性・タイミング感度が残ることを確認した。
- [x] 同じ観測HHIをone-sided innovationと予測可能な水準に分解する仕様を比較した。Cell 1のfast regressor直交成分は3.9%から65.3%へ改善したが、current/lag 1で符号が変わり、95%区間はゼロを含んだ。
- [ ] 新しい四半期HHIを用いたCell 1の `theta` 再推定を行うか決定する（保留）。
- [ ] `theta` の事後SD／事前SD基準を打切り条件にするか決定する（保留）。
- [x] 産業別PPI×産業別competitionパネルへの移行案は採用しない。
- [ ] 外生変動と `b_x` を追加したHSA等式の再評価は保留する。

## 完了条件

- [ ] Capital IQ と LSEG Datastream のそれぞれから、米国上場企業の point-in-time に近い四半期企業パネルを取得する。
- [ ] 同一の市場定義・企業選択・売上定義で、ベンダー別の市場 HHI と `1/HHI` を作る。
- [ ] 年次 `N_Gustavo`、SEC inverse HHI、Capital IQ、Datastream の重複期間比較を完了する。
- [ ] 一社抜き・上位社寄与・産業分類変更・生存者バイアスの診断を通す。
- [ ] Capital IQ版とDatastream版のHHIを比較し、推定に用いる主系列と感度分析系列を固定する。
- [ ] HHIの直接利用方法を固定し、保留中の `theta` 再推定について判断できる状態にする。

## Capital IQ / LSEG追加までの暫定主仕様（必須）

1. `N_Gustavo`だけをcompetition観測として使用する。
2. 年次値は各年Q4にのみ置き、Q1--Q3は欠測とする。
3. 四半期のslow stateとstationary stateは混合頻度状態空間モデルで推定し、PCHIP等による決定論的補間はしない。
4. 状態推定にはinflationを入れず、状態事後分布をinflation式へ渡すmodular cutとする。
5. QCEWとSEC HHIは暫定主仕様に含めない。
6. 自由な状態モデルの収束失敗と、固定持続性に対する`theta`感度を必ず併記する。
7. Capital IQ / LSEGの四半期HHIがデータ監査を通過した時点でのみ、暫定主仕様の置換を検討する。

## 0. 三田メディアセンターでの事前確認

- [ ] Capital IQ Pro と LSEG Workspace / Datastream の利用可能端末、場所、利用時間、予約要否を確認する。
- [ ] Excel add-in、Datastream for Office、CSV/Excel export の利用可否を確認する。
- [ ] 1回・1日当たりの行数、企業数、時系列期間、バッチ数の制限を確認する。
- [ ] 契約上、raw export を研究用PC・Dropbox・Gitへ保存できるか確認する。
- [ ] 再配布不可の場合、rawデータはGitへ入れず、加工コード、変数表、取得日、検索条件、集計済みHHIだけを保存できるか確認する。
- [ ] 長時間抽出に備え、USB/暗号化ストレージ、空き容量、Excelの最大行数を確認する。
- [ ] 不明点は三田メディアセンターのレファレンス担当へ問い合わせる。

## 1. 推定対象を取得前に固定する

- [ ] 国・上場市場：米国で取引される事業会社を基本とする。
- [ ] 証券ではなく発行企業単位で集計し、複数クラス株を二重計上しない。
- [ ] ADR、外国企業、OTC、SPAC、shell、fund、ETF、closed-end fund の採否を固定する。
- [ ] 基本ケースでは金融業を含む。ただし SIC 6000--6999 除外版を必ず作る。
- [ ] market definition の主仕様を SIC3 とする。NAICS3/4、GICS industry は感度分析とする。
- [ ] 売上は正の連結売上を基本とし、単位・通貨を USD に統一する。
- [ ] 主仕様は calendar-quarter の単独四半期売上とする。LTM売上版を感度分析にする。
- [ ] 対象期間は利用可能な最長期間とし、最低でも既存推定期間 `1982Q1--2012Q4` の被覆可能性を確認する。
- [ ] aggregate の主仕様を revenue-weighted geometric mean of effective firms と固定する。

市場内の企業売上シェアを

```text
s_i,m,t = revenue_i,t / sum_i(revenue_i,t)
HHI_m,t = sum_i(s_i,m,t^2)
N_eff,m,t = 1 / HHI_m,t
```

とする。全国集約の主仕様は

```text
N_t = exp(sum_m(w_m,t * log(N_eff,m,t)))
w_m,t = market_revenue_m,t / total_revenue_t
```

とする。既存実装では `inv_hhi_logrevw` に対応する。

## 2. Capital IQ から取得する

- [ ] Screen/Companies で、固定した米国上場企業universeを作る。
- [ ] active firmsだけでなく、各時点に存在した inactive / delisted firms を含める。
- [ ] 現在の構成企業を過去へ遡る検索になっていないことを確認する。
- [ ] 企業ID、company name、security ID、上場・廃止日、domicile、exchangeを取得する。
- [ ] 各時点の SIC、NAICS、GICS と、その分類の有効日または取得可能な履歴を取得する。
- [ ] standardized revenue と as-reported revenue の双方の利用可能性を確認する。
- [ ] fiscal period end、filing/publication date、通貨、単位、会計期間月数を取得する。
- [ ] 四半期売上、FY売上、YTD売上、LTM売上を識別できるフィールドを保存する。
- [ ] merger、spin-off、identifier changeを追跡できるparent/company mappingを取得する。
- [ ] 検索条件、field名、画面またはtemplate名、取得日時をログに残す。
- [ ] まず10社×10四半期を試験抽出し、単独四半期値とYTD値を混同していないか手計算する。
- [ ] 全期間を年または5年単位に分割してexportする。

想定raw schema：

```text
vendor, vendor_company_id, company_name, security_id,
calendar_quarter, fiscal_period_end, filing_date,
sic, naics, gics, revenue, currency, unit,
revenue_basis, active_status, listing_start, listing_end,
extract_timestamp, query_version
```

## 3. LSEG Datastream から独立に取得する

- [ ] Datastreamの米国equity universeを固定し、dead/delisted seriesを含める。
- [ ] constituent listではなく、全企業universeまたはpoint-in-time listを使う。
- [ ] Datastream code、Worldscope/PermID等の企業ID、企業名、上場・廃止日を取得する。
- [ ] 売上、通貨、単位、fiscal period end、報告日を取得する。
- [ ] SIC/NAICS/ICB/TRBC等、利用可能な産業分類と分類履歴を取得する。
- [ ] Capital IQと同じcalendar-quarter・USD・企業単位へ変換できるか確認する。
- [ ] 10社×10四半期の試験抽出をCapital IQと突合する。
- [ ] 全期間を分割exportし、検索式・mnemonic・list ID・取得日時を保存する。
- [ ] Capital IQと同じraw schemaに正規化する。ただしvendor固有IDは保持する。

## 4. データ整形・企業IDの管理

- [ ] rawファイルは不変とし、`data/raw/competition/vendor_hhi/` 相当の非Git領域へ保存する。
- [ ] rawファイルごとに SHA-256、取得日時、利用端末、query versionをmanifestへ記録する。
- [ ] Capital IQ ID、Datastream code、PermID、CUSIP、ISIN、tickerのcrosswalkを作る。
- [ ] tickerだけで企業を結合しない。
- [ ] 複数証券をcompany-levelへ集約し、売上を重複計上しない。
- [ ] fiscal quarterをcalendar quarterへ割り当てる規則を固定する。
- [ ] YTDしかない場合は差分で単独四半期売上を作り、FY Q4の計算規則を記録する。
- [ ] restatementは原則として最新値版とpoint-in-time版を分ける。
- [ ] 非USD売上は同一四半期の為替規則でUSD化し、名目換算の方法を記録する。
- [ ] 売上ゼロ・負値・欠損・異常な会計期間の処理規則を固定する。
- [ ] 産業分類変更を現在分類で過去へ遡及しない主仕様を作る。

## 5. HHI生成コード

- [ ] vendor共通schemaを入力する新しいloaderを `src/nkpc_hsa/dataprep/` に追加する。
- [ ] 既存の `calculate_quarterly_hhi()` と同じ出力schemaへ変換する。
- [ ] 少なくとも以下を四半期ごとに出力する。

```text
quarter
hhi
hhi_10000
effective_firms
hhi_revenue_weighted
inv_hhi_revw
inv_hhi_logrevw
inv_hhi_firmw_exfin
inv_hhi_revw_exfin
inv_hhi_logrevw_exfin
n_firms
n_markets
total_revenue_usd
vendor
universe_version
```

- [ ] market-level panelも保存し、`quarter × industry` のHHI、企業数、売上、最大企業シェアを残す。
- [ ] company-level寄与表も保存し、特定企業がaggregateを支配していないか追跡可能にする。
- [ ] `validate_hhi_fraction()` を通し、`HHI in (0,1]`、`effective_firms=1/HHI`を検証する。
- [ ] Capital IQ版とDatastream版を別ファイルで保存し、平均して一系列にしない。

## 6. 品質管理

- [ ] 四半期ごとの企業数、産業数、総売上、欠損率を図示する。
- [ ] coverageが急変する四半期とvendorの制度変更を特定する。
- [ ] 上位1、5、10社を除外したleave-largest-out HHIを作る。
- [ ] 各企業を一社ずつ除いた近似jackknife influenceを計算する。
- [ ] 金融込み／除外、SIC3／NAICS3、quarterly／LTM、最新値／point-in-timeを比較する。
- [ ] M&A、spin-off、大型IPO・delisting前後に機械的な断層がないか確認する。
- [ ] Capital IQとDatastreamの重複企業について売上差を比較する。
- [ ] ベンダー間のHHI相関、平均差、トレンド差、転換点を比較する。
- [ ] 既存SEC inverse HHIとの重複期間相関と水準差を比較する。
- [ ] 年次 `N_Gustavo` のQ4値とのPearson/Spearman相関を比較する。
- [ ] ベンダー差が大きい四半期について、上位寄与企業とcoverage差を手作業で監査する。

## 7. HHI系列の選定と推定への接続

別個の潜在共通因子やmeasurement-onlyゲートは設けない。Capital IQとDatastreamから同じ定義でHHIを作り、coverageと構成差を確認したうえで、四半期HHIを観測されたcompetition coordinateとして直接利用する。

- [ ] 主系列の選択規則を、期間、欠損率、inactive/delisted coverage、分類履歴、企業重複率で固定する。
- [ ] 他方のvendorを同一定義による感度分析系列として固定する。
- [ ] 既存SEC inverse HHIとの重複期間で水準、変化率、転換点を比較する。
- [ ] `N_Gustavo`との水準接続が必要か、四半期HHIだけで標本を構成するか決定する。
- [ ] HHIを直接使うか、事前に固定した一変量trend/cycle分解を使うか決定する。
- [ ] trend/cycle分解を採用する場合も、QCEWとの共通因子には戻さない。
- [ ] 主系列とvendor感度系列で主要係数の符号・大きさを比較する。

## 8. `theta` と HSA の再推定へ進む条件

この節の採用自体を保留する。独立したmeasurement-onlyゲートは前提にしない。採用する場合の候補条件を以下に記録する。

- [ ] Capital IQ／Datastream／SECの少なくとも二つでHHIの方向と主要転換点が整合する。
- [ ] `theta` に対応する短期変動の定義を、推定結果を見る前に固定する。
- [ ] `theta`の事後SD／事前SDが `<=0.75`（打切り基準化は保留）。
- [ ] `theta`の符号と大きさがpersistent-error、low-frequency nuisanceで安定する。
- [ ] Cell 1で `kappa_1 - b_x*zeta_ref*theta_ref` を再計算する。

産業別PPI × 産業別HHIのパネル推定には移行しない。候補状態が測定上支持されなければ、fast stateと `theta` の実質的解釈を行わず、slow-only仕様を維持する。

## 9. 成果物

- [ ] `capitaliq_company_quarter_manifest.json`
- [ ] `datastream_company_quarter_manifest.json`
- [ ] `capitaliq_hhi_quarterly.csv`
- [ ] `datastream_hhi_quarterly.csv`
- [ ] `capitaliq_hhi_market_quarter.csv`
- [ ] `datastream_hhi_market_quarter.csv`
- [ ] `vendor_hhi_crosswalk.csv`
- [ ] `vendor_hhi_quality_audit.csv`
- [ ] `vendor_hhi_comparison.csv`
- [ ] coverage図、ベンダー比較図、年次座標との比較図
- [ ] rawから集計済みHHIまでの再現手順。ただし契約上の再配布制限を明記する。

## 三田で最初に持参する取得メモ

```text
Research objective:
Quarterly U.S. listed-company revenue HHI by industry, including inactive/delisted firms.

Required frequency:
Quarterly; longest history available, target 1982Q1 onward.

Required identifiers:
Stable company ID, security ID, ISIN/CUSIP if allowed, listing and delisting dates.

Required financial fields:
Standalone quarterly revenue, fiscal-period end, filing/publication date,
currency, unit, annual/YTD/quarterly flag, restatement or vintage information.

Required classifications:
Historical SIC and NAICS; GICS/ICB/TRBC where available.

Universe requirement:
U.S. listed operating companies, including inactive and delisted observations;
exclude funds/ETFs and prevent duplicate share classes.
```
