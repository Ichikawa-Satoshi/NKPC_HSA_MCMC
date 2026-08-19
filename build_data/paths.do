*==============================================================================
* paths.do -- shared path globals for the STATA data-build pipeline
*
* All raw inputs and processed outputs live in the Dropbox data folder (the
* repository's `data/` is that folder).  Every build .do file runs this first.
* Override $DROPBOX before calling if the folder lives elsewhere.
*==============================================================================
version 17
set more off

if "$DROPBOX" == "" {
    * Default Dropbox project folder (CloudStorage is the canonical mount).
    local home : environment HOME
    local probe "data/raw/competition/CapitalIQ_USmajorlistedRevenue/SPGlobal_Export_list.xlsx"
    local c1 "`home'/Library/CloudStorage/Dropbox/NKPC_HSA_MCMC"
    local c2 "`home'/Dropbox/NKPC_HSA_MCMC"
    if      fileexists("`c1'/`probe'") global DROPBOX "`c1'"
    else if fileexists("`c2'/`probe'") global DROPBOX "`c2'"
    else {
        display as error "Could not find Dropbox/NKPC_HSA_MCMC; set global DROPBOX."
        exit 601
    }
}

global RAW   "$DROPBOX/data/raw"
global PROC  "$DROPBOX/data/processed"
global CLEAN "$DROPBOX/data/clean"
global CIQ   "$RAW/competition/CapitalIQ_USmajorlistedRevenue"

display as text "DROPBOX = $DROPBOX"
