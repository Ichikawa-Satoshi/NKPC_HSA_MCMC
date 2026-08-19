*==============================================================================
* run.do -- master script for the STATA data-build pipeline
*
* Run from the repository root:
*   /Applications/StataNow/StataSE.app/Contents/MacOS/stata-se -b do build_data/run.do
*
* Each step is self-contained (clears, re-reads paths.do) and writes its output
* CSV under data/processed/.  Downstream Python only READS those clean files.
*==============================================================================
version 17
set more off

* Capital IQ competition panel + HHI + economy-wide effective-firm series
do "`c(pwd)'/build_data/capital_iq.do"

* Model-ready dataset.  The single non-STATA step (PCHIP interpolation of the
* annual competition series) runs first and writes an interim CSV that 03 reads.
shell python3 "`c(pwd)'/build_data/pchip_competition.py"
do "`c(pwd)'/build_data/model_ready.do"

display as text "build_data pipeline complete."
