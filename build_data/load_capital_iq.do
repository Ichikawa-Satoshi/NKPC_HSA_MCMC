*==============================================================================
* 01_capital_iq.do
*
* Build ALL Capital IQ / S&P Global competition data entirely in STATA, straight
* from the raw Excel exports -- in three parts, no Python:
*   (A) the company-quarter revenue panel,
*   (B) the revenue-share HHI per broad market and quarter,
*   (C) the economy-wide effective-firm-count series (firm-count and revenue
*       weighted) that feeds model_ready.
*
* Raw layout (Sheet1 of every SPGlobal_Export_<years>.xlsx):
*   row 4  machine field codes (SP_ENTITY_ID, NAICS_CODE, IQ_TOTAL_REV ...)
*   row 5  fiscal-quarter labels FQ<q><yyyy> above each revenue column
*   row 6  "Current" marker
*   row 7+ one row per company; revenue in THOUSANDS of USD, "NA" when unreported
* The revenue columns are not in date order and the order differs across files,
* so the FQ label -- read from row 5 -- defines each column's quarter.
*
* Outputs (data/processed/):
*   capital_iq_company_quarter_revenues.csv   (panel)
*   capital_iq_hhi_by_<market>.csv            (per-market HHI)
*   capital_iq_N_quarterly.csv                (economy-wide N, firmw & revw)
*==============================================================================
version 17
clear all
set more off

do "`c(pwd)'/build_data/paths.do"

* Broad market for HHI: SIC divisions with manufacturing split durable/nondurable.
if "$MARKET"   == "" global MARKET   "sic_group"
if "$MINFIRMS" == "" global MINFIRMS "10"

*------------------------------------------------------------------------------
* Program: read one revenue workbook -> long (entity_id, tq, rev string)
*------------------------------------------------------------------------------
capture program drop _load_ciq_rev
program define _load_ciq_rev
    args file
    import excel using "`file'", sheet("Sheet1") cellrange(A4) allstring clear
    * obs 1 = field codes, obs 2 = FQ labels, obs 3 = "Current".
    foreach v of varlist * {
        local code = `v'[1]
        local lab  = `v'[2]
        if "`code'" == "SP_ENTITY_ID"        rename `v' entity_id
        else if "`code'" == "SP_ENTITY_NAME" rename `v' entity_name
        else if "`code'" == "NAICS_CODE"     rename `v' naics_code
        else if "`code'" == "IQ_TOTAL_REV" {
            local q = substr("`lab'", 3, 1)
            local y = substr("`lab'", 4, 4)
            capture confirm number `y'
            if _rc == 0  rename `v' rev`=yq(`y', `q')'
            else         rename `v' _drop_`v'
        }
        else rename `v' _drop_`v'
    }
    capture drop _drop_*
    drop in 1/3
    destring entity_id, replace force
    drop if missing(entity_id)
    * long: one row per (entity, quarter); tq is the STATA quarterly date.
    reshape long rev, i(entity_id entity_name naics_code) j(tq)
end

*------------------------------------------------------------------------------
* Load and stack the four (disjoint-decade) revenue workbooks
*------------------------------------------------------------------------------
tempfile stack
local first = 1
foreach yy in 1980_1990 1991_2000 2001_2010 2011_2023 {
    _load_ciq_rev "$CIQ/SPGlobal_Export_`yy'.xlsx"
    if `first' {
        save "`stack'", replace
        local first = 0
    }
    else {
        append using "`stack'"
        save "`stack'", replace
    }
}
use "`stack'", clear

*------------------------------------------------------------------------------
* Clean revenue
*------------------------------------------------------------------------------
* "NA" strings -> missing.
destring rev, generate(total_revenue_thousand_usd_raw) force
drop rev
drop if missing(total_revenue_thousand_usd_raw)      // keep reported firm-quarters
* collapse any stray duplicate (entity, quarter) to the last, as a safeguard.
bysort entity_id tq (total_revenue_thousand_usd_raw): keep if _n == _N

* Negative "total revenue" (net-of-loss financial filers) -> 0 cleaned revenue.
gen double total_revenue_thousand_usd = total_revenue_thousand_usd_raw
replace     total_revenue_thousand_usd = 0 if total_revenue_thousand_usd < 0
gen double total_revenue_usd = total_revenue_thousand_usd * 1000

* Calendar bookkeeping from the quarterly date.
format tq %tq
gen int fiscal_year = year(dofq(tq))
gen byte fiscal_q   = quarter(dofq(tq))
gen str  fiscal_quarter = string(fiscal_year) + "Q" + string(fiscal_q)

* Leading NAICS code from "525990 - Other Financial ...".
gen str naics = regexs(1) if regexm(naics_code, "^([0-9]+)")

*------------------------------------------------------------------------------
* Attach SIC + S&P industry from the company-list workbook, and derive the
* broad classifications (SIC divisions, manufacturing split durable/nondurable)
*------------------------------------------------------------------------------
preserve
    import excel using "$CIQ/SPGlobal_Export_list.xlsx", ///
        sheet("Sheet1") cellrange(A4) allstring clear
    foreach v of varlist * {
        local code = `v'[1]
        if "`code'" == "SP_ENTITY_ID"                 rename `v' entity_id
        else if "`code'" == "SP_SIC_CODE"             rename `v' sic
        else if "`code'" == "IQ_INDUSTRY_CLASSIFICATION" rename `v' sp_industry
        else rename `v' _drop_`v'
    }
    capture drop _drop_*
    drop in 1/3
    destring entity_id, replace force
    drop if missing(entity_id)
    * keep clean 4-digit SIC only.
    replace sic = "" if !regexm(sic, "^[0-9][0-9][0-9][0-9]$")
    * S&P sector = top level of the ";"-delimited industry text (the readable 大分類).
    gen str sp_sector = sp_industry
    replace sp_sector = substr(sp_industry, 1, strpos(sp_industry, ";") - 1) ///
        if strpos(sp_industry, ";") > 0
    replace sp_sector = strtrim(sp_sector)
    duplicates drop entity_id, force
    tempfile meta
    save "`meta'"
restore
merge m:1 entity_id using "`meta'", keep(master match) nogenerate

* SIC major group and division (the coarse 大分類).
gen str sic2 = substr(sic, 1, 2)
destring sic2, generate(mg) force
gen str sic_division = ""
replace sic_division = "A_agriculture"         if inrange(mg,  1,  9)
replace sic_division = "B_mining"              if inrange(mg, 10, 14)
replace sic_division = "C_construction"        if inrange(mg, 15, 17)
replace sic_division = "D_manufacturing"       if inrange(mg, 20, 39)
replace sic_division = "E_transport_utilities" if inrange(mg, 40, 49)
replace sic_division = "F_wholesale"           if inrange(mg, 50, 51)
replace sic_division = "G_retail"              if inrange(mg, 52, 59)
replace sic_division = "H_finance_realestate"  if inrange(mg, 60, 67)
replace sic_division = "I_services"            if inrange(mg, 70, 89)
replace sic_division = "J_public_admin"        if inrange(mg, 91, 97)

* Broad market key: division, but manufacturing split durable / nondurable.
gen str sic_group = sic_division
replace sic_group = "D1_manuf_nondurable" if inlist(mg,20,21,22,23,26,27,28,29,30,31)
replace sic_group = "D2_manuf_durable"    if inlist(mg,24,25,32,33,34,35,36,37,38,39)

drop mg

*------------------------------------------------------------------------------
* Order, sort, export
*------------------------------------------------------------------------------
order entity_id entity_name naics_code naics sic sic2 sic_division sic_group ///
      sp_sector sp_industry fiscal_quarter fiscal_year fiscal_q tq ///
      total_revenue_thousand_usd total_revenue_thousand_usd_raw total_revenue_usd
sort entity_id tq

quietly count
display as text "panel rows      : " %12.0fc r(N)
quietly levelsof entity_id, local(ids)
display as text "entities        : " `: word count `ids''
quietly summarize fiscal_year
display as text "fiscal years    : " r(min) " .. " r(max)

export delimited using "$PROC/capital_iq_company_quarter_revenues.csv", replace
display as text "wrote $PROC/capital_iq_company_quarter_revenues.csv"

*==============================================================================
* (B) Revenue-share HHI per broad market and quarter
*==============================================================================
* Work on the in-memory panel (one row per firm-quarter).
keep entity_id tq fiscal_year $MARKET total_revenue_usd
drop if missing($MARKET) | $MARKET == ""
drop if missing(total_revenue_usd) | total_revenue_usd <= 0

egen double market_rev = total(total_revenue_usd), by(tq $MARKET)
gen  double share_sq   = (total_revenue_usd / market_rev)^2
collapse (sum)   hhi           = share_sq          ///
         (sum)   total_rev_usd = total_revenue_usd ///
         (count) n_firms       = total_revenue_usd ///
         (first) fiscal_year, by(tq $MARKET)
drop if n_firms < $MINFIRMS

gen double hhi_10000       = hhi * 10000
gen double effective_firms = 1 / hhi
gen str    market_level    = "$MARKET"
gen str    fiscal_quarter  = string(year(dofq(tq))) + "Q" + string(quarter(dofq(tq)))
rename $MARKET market
format tq %tq

* Seasonally adjust each industry's HHI before aggregation.  Quarterly firm
* revenue is seasonal (e.g. retail Q4), so revenue-share HHIs carry an
* industry-specific seasonal that must go before the series are pooled.
*
* MULTIPLICATIVE seasonal adjustment per industry via a log two-way fit.  HHI is
* a ratio whose seasonal is proportional to the level (finance Q4 HHI is ~2x the
* other quarters) and the seasonal evolves as HHI trends, so a within-year effect
* is used to soak up the level/trend before the quarter effect is measured:
*   1. work in logs; per industry, remove the WITHIN-YEAR mean (absorbs the level
*      of each year, i.e. the full trend/cycle)
*   2. seasonal factor per calendar quarter = mean of that deviation, demeaned so
*      the four log factors sum to zero
*   3. hhi_sa = exp(log HHI - seasonal factor)  (>0 by construction)
gen int    _q    = quarter(dofq(tq))
gen double _lhhi = ln(hhi)
bysort market fiscal_year: egen double _ymean = mean(_lhhi)
gen double _dev = _lhhi - _ymean
bysort market _q: egen double _qeff = mean(_dev)
by market _q: gen byte _firstq = (_n == 1)
bysort market: egen double _qmean = mean(cond(_firstq, _qeff, .))
gen double hhi_sa = exp(_lhhi - (_qeff - _qmean))
drop _q _lhhi _ymean _dev _qeff _firstq _qmean

preserve
    order fiscal_quarter fiscal_year market_level market n_firms ///
          total_rev_usd hhi hhi_sa hhi_10000 effective_firms tq
    sort tq market
    quietly count
    display as text "HHI market-quarters : " %10.0fc r(N)
    export delimited using "$PROC/capital_iq_hhi_by_$MARKET.csv", replace
    display as text "wrote $PROC/capital_iq_hhi_by_$MARKET.csv"
restore

*==============================================================================
* (C) Economy-wide effective firm count = 1 / (weighted mean SEASONALLY-ADJUSTED
*     market HHI)
*     N_capitaliq_firmw : markets weighted by firm count (attaches each market
*                         HHI to every firm; matches SEC inv_hhi_firmw)
*     N_capitaliq_revw  : markets weighted by revenue
*==============================================================================
egen double _tot_firms = total(n_firms),       by(tq)
egen double _tot_rev   = total(total_rev_usd),  by(tq)
gen  double _hhi_fw    = hhi_sa * n_firms       / _tot_firms
gen  double _hhi_rw    = hhi_sa * total_rev_usd / _tot_rev
collapse (sum) mean_hhi_firmw = _hhi_fw ///
         (sum) mean_hhi_revw  = _hhi_rw, by(tq)
gen double N_capitaliq_firmw = 1 / mean_hhi_firmw
gen double N_capitaliq_revw  = 1 / mean_hhi_revw
keep tq N_capitaliq_firmw N_capitaliq_revw
sort tq
format tq %tq

quietly count
display as text "N_capitaliq quarters: " %10.0fc r(N)
export delimited using "$PROC/capital_iq_N_quarterly.csv", replace
display as text "wrote $PROC/capital_iq_N_quarterly.csv"
