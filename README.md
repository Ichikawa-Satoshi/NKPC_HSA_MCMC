# FMI_NKPC_HSA_MCMC

Research workspace for estimating New Keynesian Phillips Curve specifications with endogenous cost-push channels. The active codebase is organized around reusable source code, analysis workflows, inputs, and generated results, while `_scrap/` remains as an archive of exploratory material.

## Directory Layout

- `src/`
  Shared Python modules used by the active Gibbs and HMC workflows.

- `analysis/`
  Active estimation and reporting workflows.
  - `gibbs/`: Gibbs sampling notebooks and helper scripts.
  - `hmc/`: HMC notebooks and export scripts.
  - `ols/`: OLS notebooks and table-export helpers.

- `data/`
  Source datasets grouped by topic.
  - `competition/`
  - `inflation/`
  - `markup/`
  - `others/`
  - `output_gap/`
  - `unemp_gap/`
  - `_scrap/`: archived raw inputs retained for reference

- `results/`
  Generated artifacts from active workflows.
  - `fig/`: figures
  - `idata/`: posterior draws and inference datasets
  - `tex/`: LaTeX tables and summary outputs

- `docs/`
  Supporting project material.
  - `notes/`: LyX notes and derivations
  - `literature/`: reference papers

- `_scrap/`
  Archived experiments, older MATLAB code, draft outputs, and exploratory notebooks. This directory is intentionally left outside the active workflow layout.

## Working Conventions

- Run active estimation notebooks from `analysis/`.
- Reusable code should live in `src/`, not inside notebook folders.
- Generated outputs should go to `results/`.
- `_scrap/` is archival; do not treat it as the source of truth for the active pipeline.
