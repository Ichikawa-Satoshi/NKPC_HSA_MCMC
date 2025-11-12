clear all
cd "/Users/satoshan/GitHub/NKPC_HSA_MCMC/code_matlab/"
* ---------------- SPF (xlsx) ----------------
import excel "./data/inflation/Inflation.xlsx", firstrow clear
gen double qdate = yq(YEAR, QUARTER)
format qdate %tq
rename INFPGDP1YR Epi_spf_gdp
rename INFCPI1YR  Epi_spf_cpi
keep if Epi_spf_gdp != "#N/A"
keep if Epi_spf_cpi != "#N/A"
destring Epi_spf_cpi, replace
destring Epi_spf_gdp, replace
keep qdate Epi_spf_gdp Epi_spf_cpi
tempfile spf
save `spf'

* ---------------- CPI headline (CSV→ log diff YoY) ----------------
import delimited "./data/inflation/CPIAUCSL.csv", varnames(1) clear
gen double qdate = qofd(daily(date,"YMD"))
format qdate %tq
collapse (mean) cpiaucsl, by(qdate)
tsset qdate, quarterly
gen double pi_cpi = 100*(cpiaucsl/L4.cpiaucsl - 1)
keep qdate pi_cpi
tempfile cpi
save `cpi'

* ---------------- Core CPI (CSV, log diff YoY) ----------------
import delimited "./data/inflation/CPILFESL.csv", varnames(1) clear
capture confirm variable observation_date
if !_rc gen double qdate = qofd(daily(observation_date,"YMD"))
else     gen double qdate = qofd(daily(DATE,"YMD"))
format qdate %tq
collapse (mean) cpilfesl, by(qdate)
tsset qdate, quarterly
gen double pi_cpi_core = 100*(ln(cpilfesl) - ln(L4.cpilfesl))
keep qdate pi_cpi_core
tempfile cpicore
save `cpicore'

* ---------------- PCE headline (CSV, log diff YoY) ----------------
import delimited "./data/inflation/PCEPI.csv", varnames(1) clear
gen double qdate = qofd(daily(observation_date,"YMD"))
format qdate %tq
collapse (mean) pcepi, by(qdate)
tsset qdate, quarterly
gen double pi_pce = 100*(ln(pcepi) - ln(L4.pcepi))
keep qdate pi_pce
tempfile pce
save `pce'

* ---------------- PCE core (CSV, log diff YoY) ----------------
import delimited "./data/inflation/PCEPILFE.csv", varnames(1) clear
gen double qdate = qofd(daily(observation_date,"YMD"))
format qdate %tq
collapse (mean) pcepilfe, by(qdate)
tsset qdate, quarterly
gen double pi_pce_core = 100*(ln(pcepilfe) - ln(L4.pcepilfe))
keep qdate pi_pce_core
tempfile pcecore
save `pcecore'

* ---------------- HHI / number of firms（interporate） ----------------
import delimited "./data/competition/BN_N_26.csv", varnames(1) clear
gen double yearnum = real(substr(date,1,4))
keep if !missing(original_series) & !missing(yearnum)
keep yearnum original_series
rename original_series N
gen double qdate = yq(yearnum,1)
format qdate %tq
tsset qdate, quarterly
tsfill, full
pchipolate N qdate, gen(N_q)
drop N
rename N_q N
keep qdate N
tempfile hhiq
save `hhiq'

* ---------------- N_BN ----------------
import delimited "./data/competition/BN_N_26.csv", varnames(1) clear
gen double yearnum = real(substr(date,1,4))
keep if !missing(cycle) & !missing(yearnum)
keep yearnum cycle
rename cycle N_BN
gen double qdate = yq(yearnum,1)
format qdate %tq
tsset qdate, quarterly
tsfill, full
pchipolate N_BN qdate, gen(N_BN_q)
drop N_BN
rename N_BN_q N_BN
keep qdate N_BN
tempfile nbq
save `nbq'
* ---------------- Detrended Markup (BN inverse) ----------------
import delimited "./data/markup/BN_markup_inv.csv", varnames(1) clear
gen double qdate = qofd(daily(date,"YMD"))
format qdate %tq
rename cycle markup_BN_inv
keep qdate markup_BN_inv
tempfile mkbn
save `mkbn'
* ---------------- Employment gap ----------------
import delimited "./data/unemp_gap/NROU.csv", varnames(1) clear
gen double qdate = qofd(daily(observation_date,"YMD"))
format qdate %tq
collapse (mean) nrou, by(qdate)
tempfile a
save `a'

import delimited "./data/unemp_gap/UNRATENSA.csv", varnames(1) clear
gen double qdate = qofd(daily(observation_date,"YMD"))
format qdate %tq
collapse (mean) unrate, by(qdate)
tempfile b
save `b'

use `a', clear
merge 1:1 qdate using `b', nogen
gen double unemp_gap = nrou - unrate
keep qdate unemp_gap
tempfile gap
save `gap'

* ---------------- Markup（level） ----------------
import excel "./data/markup/nekarda_ramey_markups.xlsx", firstrow clear
gen double qdate_q = qofd(qdate)
replace qdate = qofd(qdate)
format qdate %tq
rename mu_bus markup
keep qdate markup
tempfile mk
save `mk'

* ---------------- Output gap (BN filter) ----------------
import delimited "./data/output_gap/BN_filter_GDPC1_quaterly.csv", varnames(1) clear
gen double qdate = qofd(daily(date,"YMD"))
format qdate %tq
rename gdpc1_transformed_series output_BN
rename cycle output_gap_BN
gen double output = log(gdpc1_original_series * 0.01)
gen double output_trend_BN = output_BN - output_gap_BN
keep qdate output_BN output_gap_BN output_trend_BN output
tempfile out
save `out'

* ----------------One year inflation expectation ----------------
import delimited "./data/inflation/one_year_inflation_expectation.csv", varnames(1) clear
capture confirm numeric variable epi
if _rc {
    tostring epi, replace
    split epi, parse(",") gen(e_)
    gen double epi_num = real(e_2)
    replace epi_num = real(epi) if missing(epi_num)
}
else {
    gen double epi_num = epi
}
gen double qdate = qofd(daily(date,"YMD"))
format qdate %tq
collapse (mean) epi_num, by(qdate)
gen double Epi = epi_num*100
keep qdate Epi
tempfile epi
save `epi'

* ---------------- Oil (WTI/CPI,) ----------------
import delimited "./data/others/WTISPLC_CPIAUCSL.csv", varnames(1) clear
gen double qdate = qofd(daily(observation_date,"YMD"))
format qdate %tq
collapse (mean) wtisplc_cpiaucsl, by(qdate)
tsset qdate, quarterly
gen double log_oil = ln(wtisplc_cpiaucsl)
gen double oil     = log_oil - L4.log_oil
keep qdate oil
tempfile oil
save `oil'

* ---------------- merge ----------------
use `cpi', clear
merge 1:1 qdate using `epi',     nogen
merge 1:1 qdate using `spf',     nogen
merge 1:1 qdate using `cpicore', nogen
merge 1:1 qdate using `out',     nogen
merge 1:1 qdate using `hhiq',    nogen
merge 1:1 qdate using `nbq',     nogen
merge 1:1 qdate using `mk',      nogen
merge 1:1 qdate using `pcecore', nogen
merge 1:1 qdate using `pce',     nogen
merge 1:1 qdate using `mkbn',    nogen
merge 1:1 qdate using `gap',     nogen
merge 1:1 qdate using `oil',     nogen
tsset qdate, quarterly
gen double pi_cpi_prev      = L1.pi_cpi
gen double pi_cpi_core_prev = L1.pi_cpi_core
gen double pi_pce_prev      = L1.pi_pce
gen double pi_pce_core_prev = L1.pi_pce_core

ds qdate, not
local vlist `r(varlist)'
egen __nmiss = rowmiss(`vlist')
drop if __nmiss>0
drop __nmiss

twoway ///
    (line markup qdate, lwidth(med)) ///
    (line N qdate, yaxis(2) lpattern(dash)), ///
    legend(order(1 "Markup" 2 "N (≈ inv HHI)") col(2) position(6) ring(1)) ///
    ytitle("Markup") ytitle("N", axis(2)) ///
    xtitle("Quarter") ///
    name(fig_bncycles, replace)
