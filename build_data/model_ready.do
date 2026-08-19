*==============================================================================
* 03_model_ready.do
*
* Build the merged quarterly model-ready dataset in STATA.
* The ONLY non-STATA step is the PCHIP interpolation of the annual competition
* series, produced beforehand by build_data/pchip_competition.py into
* processed/interim/competition_bn_quarterly.csv.
*
* Inflation source files are read from ~/Dropbox/infalation/.
* Everything is keyed on the STATA quarterly date tq.
* Output: $CLEAN/model_ready.csv
*==============================================================================
version 17
clear all
set more off

do "`c(pwd)'/build_data/paths.do"
global INTERIM "$PROC/interim"
local home : environment HOME
global INFL "`home'/Dropbox/infalation"
capture confirm file "$INFL/CPIAUCSL.csv"
if _rc global INFL "`home'/Library/CloudStorage/Dropbox/infalation"
confirm file "$INFL/CPIAUCSL.csv"

*------------------------------------------------------------------------------
* Inflation: quarterly mean of source index, then log changes.
* YoY      = 100*[ln(P_t)-ln(P_{t-4})]
* QoQ      = 100*[ln(P_t)-ln(P_{t-1})]
* QoQ ann. = 400*[ln(P_t)-ln(P_{t-1})]
*------------------------------------------------------------------------------
* --- Headline CPI ---
import delimited using "$INFL/CPIAUCSL.csv", varnames(1) clear case(preserve) stringcols(_all)
capture confirm variable DATE
if _rc rename observation_date DATE
gen int tq = qofd(date(DATE, "YMD"))
gen double _x = real(CPIAUCSL)
collapse (mean) x = _x, by(tq)
tsset tq
gen double pi_cpi = 100*(ln(x) - ln(L4.x))
gen double pi_cpi_qoq = 100*(ln(x) - ln(L1.x))
gen double pi_cpi_qoq_ann = 400*(ln(x) - ln(L1.x))
keep tq pi_cpi pi_cpi_qoq pi_cpi_qoq_ann
tempfile b_cpi
save "`b_cpi'"

* --- Core CPI ---
import delimited using "$INFL/CPILFESL.csv", varnames(1) clear case(preserve) stringcols(_all)
capture confirm variable DATE
if _rc rename observation_date DATE
gen int tq = qofd(date(DATE, "YMD"))
gen double _x = real(CPILFESL)
collapse (mean) x = _x, by(tq)
tsset tq
gen double pi_cpi_core = 100*(ln(x) - ln(L4.x))
gen double pi_cpi_core_qoq = 100*(ln(x) - ln(L1.x))
gen double pi_cpi_core_qoq_ann = 400*(ln(x) - ln(L1.x))
keep tq pi_cpi_core pi_cpi_core_qoq pi_cpi_core_qoq_ann
tempfile b_cpicore
save "`b_cpicore'"

* --- Core PCE ---
import delimited using "$INFL/PCEPILFE.csv", varnames(1) clear case(preserve) stringcols(_all)
capture confirm variable DATE
if _rc rename observation_date DATE
gen int tq = qofd(date(DATE, "YMD"))
gen double _x = real(PCEPILFE)
collapse (mean) x = _x, by(tq)
tsset tq
gen double pi_pce_core = 100*(ln(x) - ln(L4.x))
gen double pi_pce_core_qoq = 100*(ln(x) - ln(L1.x))
gen double pi_pce_core_qoq_ann = 400*(ln(x) - ln(L1.x))
keep tq pi_pce_core pi_pce_core_qoq pi_pce_core_qoq_ann
tempfile b_pcecore
save "`b_pcecore'"

* --- Headline PCE ---
import delimited using "$INFL/PCEPI.csv", varnames(1) clear case(preserve) stringcols(_all)
capture confirm variable DATE
if _rc rename observation_date DATE
gen int tq = qofd(date(DATE, "YMD"))
gen double _x = real(PCEPI)
collapse (mean) x = _x, by(tq)
tsset tq
gen double pi_pce = 100*(ln(x) - ln(L4.x))
gen double pi_pce_qoq = 100*(ln(x) - ln(L1.x))
gen double pi_pce_qoq_ann = 400*(ln(x) - ln(L1.x))
keep tq pi_pce pi_pce_qoq pi_pce_qoq_ann
tempfile b_pce
save "`b_pce'"

* --- PPI ---
import delimited using "$INFL/PPIACO.csv", varnames(1) clear case(preserve) stringcols(_all)
capture confirm variable DATE
if _rc rename observation_date DATE
gen int tq = qofd(date(DATE, "YMD"))
gen double _x = real(PPIACO)
collapse (mean) x = _x, by(tq)
tsset tq
gen double pi_ppi = 100*(ln(x) - ln(L4.x))
gen double pi_ppi_qoq = 100*(ln(x) - ln(L1.x))
gen double pi_ppi_qoq_ann = 400*(ln(x) - ln(L1.x))
keep tq pi_ppi pi_ppi_qoq pi_ppi_qoq_ann
tempfile b_ppi
save "`b_ppi'"

*------------------------------------------------------------------------------
* Inflation expectations
*------------------------------------------------------------------------------
* --- Cleveland Fed: monthly one-year-ahead expectation -> quarterly mean ---
import delimited using "$INFL/Clev_Fed_Inflation_Expectation.csv", ///
    varnames(1) clear case(preserve) stringcols(_all)
ds
local dcol : word 1 of `r(varlist)'
local vcol : word 2 of `r(varlist)'
gen int tq = qofd(date(`dcol', "YMD"))
gen double Epi = real(`vcol') * 100
collapse (mean) Epi, by(tq)
tempfile b_clev
save "`b_clev'"

* --- SPF one-year-ahead expectations ---
* INFPGDP1YR and INFCPI1YR are the SPF official one-year-ahead series.
* They are based on median forecasts and summarize average inflation over the
* four quarters following the survey quarter; keep them at that horizon.
import excel using "$INFL/SPF_Inflation_Expectation.xlsx", ///
    sheet("INFLATION") firstrow clear
foreach v in YEAR QUARTER INFPGDP1YR INFCPI1YR {
    capture confirm string variable `v'
    if !_rc destring `v', replace force
}
gen int tq = yq(YEAR, QUARTER)
gen double Epi_spf_gdp = INFPGDP1YR
gen double Epi_spf_cpi = INFCPI1YR
keep tq Epi_spf_gdp Epi_spf_cpi
tempfile b_spf_1y
save "`b_spf_1y'"

* --- SPF one-quarter-ahead GDP price-index inflation forecast ---
* SPF documentation:
*   PGDP2 = mean forecast level for the survey quarter (current quarter).
*   PGDP3 = mean forecast level for the following quarter.
* For PGDP, the official mean growth rate is the growth rate of the mean
* forecast level (not the mean of individual growth-rate forecasts), using
* annualized discrete Q/Q compounding.  The log-equivalent series is retained
* separately for specifications whose realized inflation is 400*Delta log(P).
import excel using "$INFL/Mean_PGDP_Level.xlsx", ///
    sheet("Mean_Level") firstrow clear
foreach v in YEAR QUARTER PGDP2 PGDP3 {
    capture confirm string variable `v'
    if !_rc destring `v', replace force
}
gen int tq = yq(YEAR, QUARTER)
gen double Epi_spf_gdp_1q = 100*((PGDP3/PGDP2)^4 - 1) ///
    if PGDP2 > 0 & PGDP3 > 0
gen double Epi_spf_gdp_1q_log = 400*(ln(PGDP3) - ln(PGDP2)) ///
    if PGDP2 > 0 & PGDP3 > 0
keep tq Epi_spf_gdp_1q Epi_spf_gdp_1q_log
tempfile b_spf_1q
save "`b_spf_1q'"

*------------------------------------------------------------------------------
* Competition measures
*------------------------------------------------------------------------------
* --- Competition (PCHIP output from Python) ---
import delimited using "$INTERIM/competition_bn_quarterly.csv", ///
    varnames(1) clear case(preserve) stringcols(_all)
gen int tq = yq(real(year), real(quarter))
foreach v of varlist N_* {
    gen double _`v' = real(`v')
    drop `v'
    rename _`v' `v'
}
keep tq N_Gustavo N_Gustavo_BN_cycle N_Gustavo_BN_trend ///
     N_TNIC N_TNIC_BN_cycle N_TNIC_BN_trend ///
     N_Gustavo_annual_q4 N_Gustavo_BN_cycle_annual_q4 N_Gustavo_BN_trend_annual_q4 ///
     N_TNIC_annual_q4 N_TNIC_BN_cycle_annual_q4 N_TNIC_BN_trend_annual_q4
tempfile b_comp
save "`b_comp'"

* --- Capital IQ economy-wide effective firms ---
import delimited using "$PROC/capital_iq_N_quarterly.csv", ///
    varnames(1) clear case(preserve) stringcols(_all)
gen int _tq = quarterly(tq, "YQ")
gen double _fw = real(N_capitaliq_firmw)
gen double _rw = real(N_capitaliq_revw)
drop tq N_capitaliq_firmw N_capitaliq_revw
rename (_tq _fw _rw) (tq N_capitaliq_firmw N_capitaliq_revw)
keep tq N_capitaliq_firmw N_capitaliq_revw
tempfile b_ciqN
save "`b_ciqN'"

*------------------------------------------------------------------------------
* Phillips-curve forcing-variable candidates
*------------------------------------------------------------------------------
* --- Output gap: Beveridge-Nelson ---
import delimited using "$RAW/output_gap/BN_filter_GDPC1_quaterly.csv", ///
    varnames(1) clear case(preserve) stringcols(_all)
gen int tq = qofd(date(date, "YMD"))
gen double output_BN = real(GDPC1_transformed_series)
gen double output_gap_BN = real(cycle)
gen double output = ln(real(GDPC1_original_series) * 0.01)
gen double output_trend_BN = output_BN - output_gap_BN
keep tq output_BN output_gap_BN output_trend_BN output
drop if missing(output_BN, output_gap_BN, output_trend_BN, output)
tempfile b_out
save "`b_out'"

* --- Negative unemployment gap: NROU - UNRATE ---
import delimited using "$RAW/unemp_gap/NROU.csv", varnames(1) clear case(preserve)
capture confirm variable DATE
if _rc rename observation_date DATE
gen int tq = qofd(date(DATE, "YMD"))
collapse (mean) NROU, by(tq)
tempfile _nrou
save "`_nrou'"

import delimited using "$RAW/unemp_gap/UNRATE.csv", varnames(1) clear case(preserve)
capture confirm variable DATE
if _rc rename observation_date DATE
gen int tq = qofd(date(DATE, "YMD"))
collapse (mean) UNRATE, by(tq)
merge 1:1 tq using "`_nrou'", nogenerate
gen double unemp_gap = NROU - UNRATE
keep tq unemp_gap
drop if missing(unemp_gap)
tempfile b_unemp
save "`b_unemp'"

* --- Markup: Nekarda-Ramey level and BN inverse-markup measures ---
import excel using "$RAW/markup/nekarda_ramey_markups.xlsx", firstrow clear
gen int tq = qofd(qdate)
gen double markup = mu_bus
keep tq markup
drop if missing(markup)
tempfile b_mk
save "`b_mk'"

import delimited using "$RAW/markup/BN_markup_inv.csv", ///
    varnames(1) clear case(preserve)
gen int tq = qofd(date(date, "YMD"))
gen double markup_BN_inv = cycle
gen double markup_inv = original_series
keep tq markup_BN_inv markup_inv
drop if missing(markup_BN_inv, markup_inv)
tempfile b_mkbn
save "`b_mkbn'"

*------------------------------------------------------------------------------
* Merge all blocks onto a contiguous quarterly spine
*------------------------------------------------------------------------------
use "`b_cpi'", clear
foreach b in b_cpicore b_pcecore b_pce b_ppi b_clev b_spf_1y b_spf_1q ///
             b_comp b_ciqN b_out b_unemp b_mk b_mkbn {
    merge 1:1 tq using "``b''", nogenerate
}
drop if missing(tq)
tsset tq
tsfill
sort tq

* --- HP output gap: 100-log-point units ---
gen double output_100log = 100 * output
tsfilter hp output_gap_HP = output_100log if !missing(output_100log), ///
    smooth(1600) trend(output_trend_HP)
drop output_100log

*------------------------------------------------------------------------------
* Labor-share gap: quarterly FRED index -> 100*log -> HP
*------------------------------------------------------------------------------
preserve
    import delimited using "$RAW/laborshare/PRS85006173.csv", ///
        varnames(1) clear case(preserve) stringcols(_all)
    capture confirm variable DATE
    if _rc rename observation_date DATE
    gen int tq = qofd(date(DATE, "YMD"))
    gen double labor_share = real(PRS85006173)
    keep if labor_share > 0 & !missing(labor_share)
    collapse (mean) labor_share, by(tq)
    gen double labor_share_100log = 100 * ln(labor_share)
    tsset tq
    tsfilter hp labor_share_gap_HP = labor_share_100log, ///
        smooth(1600) trend(labor_share_trend_HP)
    keep tq labor_share labor_share_100log labor_share_trend_HP labor_share_gap_HP
    tempfile b_ls
    save "`b_ls'"
restore
merge 1:1 tq using "`b_ls'", nogenerate
sort tq

*------------------------------------------------------------------------------
* Establishment stock from BED births/deaths anchored on BDS 1993 ESTAB
*------------------------------------------------------------------------------
preserve
    import delimited using "$RAW/competition/bds/BDSTIMESERIES_BDSGEO.csv", ///
        varnames(1) clear case(preserve) stringcols(_all)
    keep Yeartime NumberofestablishmentsESTAB
    rename Yeartime year
    rename NumberofestablishmentsESTAB estab
    destring year, replace force
    replace estab = subinstr(estab, ",", "", .)
    destring estab, replace force
    keep if year == 1993
    keep if !missing(estab)
    local anchor = estab[1]
restore
display as text "BDS 1993 ESTAB anchor = `anchor'"

preserve
    import delimited using ///
        "$RAW/competition/bed/BLS-bd-BDS0000000000000000120007LQ5.csv", ///
        varnames(1) clear case(preserve) stringcols(_all)
    ds
    local pcol : word 1 of `r(varlist)'
    local vb : word 2 of `r(varlist)'
    gen double establishment_births = real(`vb') * 1000
    gen int tq = quarterly(subinstr(`pcol', "-", "", .), "YQ")
    keep tq establishment_births
    tempfile _births
    save "`_births'"

    import delimited using ///
        "$RAW/competition/bed/BLS-bd-BDS0000000000000000120008LQ5.csv", ///
        varnames(1) clear case(preserve) stringcols(_all)
    ds
    local pcol : word 1 of `r(varlist)'
    local vd : word 2 of `r(varlist)'
    gen double establishment_deaths = real(`vd') * 1000
    gen int tq = quarterly(subinstr(`pcol', "-", "", .), "YQ")
    keep tq establishment_deaths
    merge 1:1 tq using "`_births'", nogenerate
    drop if missing(establishment_births, establishment_deaths)
    sort tq
    gen double establishment_net_entry = establishment_births - establishment_deaths
    gen double establishment_stock = `anchor' + sum(establishment_net_entry)
    keep tq establishment_births establishment_deaths establishment_net_entry establishment_stock
    tempfile b_estab
    save "`b_estab'"
restore
merge 1:1 tq using "`b_estab'", nogenerate
sort tq

*------------------------------------------------------------------------------
* One-quarter lags
*------------------------------------------------------------------------------
tsset tq
foreach v in pi_cpi pi_ppi pi_cpi_core pi_pce pi_pce_core ///
             pi_cpi_qoq pi_ppi_qoq pi_cpi_core_qoq pi_pce_qoq pi_pce_core_qoq ///
             pi_cpi_qoq_ann pi_ppi_qoq_ann pi_cpi_core_qoq_ann ///
             pi_pce_qoq_ann pi_pce_core_qoq_ann ///
             unemp_gap markup_BN_inv markup_inv output_gap_BN output_gap_HP ///
             labor_share_gap_HP {
    capture confirm variable `v'
    if !_rc gen double `v'_prev = L1.`v'
}

*------------------------------------------------------------------------------
* Quarter-end dates and column order
*------------------------------------------------------------------------------
gen double _qend = dofq(tq + 1) - 1
gen str10 time = string(_qend, "%tdCCYY-NN-DD")
gen str26 DATE = time + " 23:59:59.999999"
drop _qend

order pi_cpi pi_cpi_qoq pi_cpi_qoq_ann ///
      Epi Epi_spf_gdp Epi_spf_cpi Epi_spf_gdp_1q Epi_spf_gdp_1q_log ///
      pi_cpi_core pi_cpi_core_qoq pi_cpi_core_qoq_ann ///
      pi_pce_core pi_pce_core_qoq pi_pce_core_qoq_ann ///
      pi_pce pi_pce_qoq pi_pce_qoq_ann ///
      pi_ppi pi_ppi_qoq pi_ppi_qoq_ann ///
      N_Gustavo N_Gustavo_BN_cycle N_Gustavo_BN_trend ///
      N_TNIC N_TNIC_BN_cycle N_TNIC_BN_trend ///
      N_Gustavo_annual_q4 N_Gustavo_BN_cycle_annual_q4 N_Gustavo_BN_trend_annual_q4 ///
      N_TNIC_annual_q4 N_TNIC_BN_cycle_annual_q4 N_TNIC_BN_trend_annual_q4 ///
      N_capitaliq_firmw N_capitaliq_revw ///
      output_BN output_gap_BN output_trend_BN output ///
      unemp_gap markup markup_BN_inv markup_inv ///
      pi_cpi_prev pi_ppi_prev pi_cpi_core_prev pi_pce_prev pi_pce_core_prev ///
      pi_cpi_qoq_prev pi_ppi_qoq_prev pi_cpi_core_qoq_prev ///
      pi_pce_qoq_prev pi_pce_core_qoq_prev ///
      pi_cpi_qoq_ann_prev pi_ppi_qoq_ann_prev pi_cpi_core_qoq_ann_prev ///
      pi_pce_qoq_ann_prev pi_pce_core_qoq_ann_prev ///
      unemp_gap_prev markup_BN_inv_prev markup_inv_prev ///
      output_gap_BN_prev DATE output_trend_HP output_gap_HP output_gap_HP_prev ///
      labor_share labor_share_100log labor_share_trend_HP labor_share_gap_HP ///
      labor_share_gap_HP_prev establishment_births establishment_deaths ///
      establishment_net_entry establishment_stock
order time, first
drop tq

quietly count
display as text "model_ready rows : " %10.0fc r(N)
quietly ds, has(type numeric)
format `r(varlist)' %20.15g

capture mkdir "$CLEAN"
export delimited using "$CLEAN/model_ready.csv", replace datafmt
shell ln -sf "$CLEAN/model_ready.csv" "$PROC/model_ready.csv"
display as text "wrote $CLEAN/model_ready.csv (symlinked from $PROC/model_ready.csv)"
