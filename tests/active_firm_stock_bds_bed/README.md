# Active-firm stock from BDS/BED

This isolated bundle implements the external-`N` step that follows the frozen
effective-competition slope test. BDS annual firm levels anchor the state; BED
establishment births/deaths provide timing through an estimated transition
loading. Inflation is cut out of the state block.

The first empirical equation contains a constant slope and free `theta_N` only.
`delta`, `lambda`, dynamic HSA terms, and marginal-likelihood comparisons are
deliberately absent until the free coefficient passes recovery.

Run from the repository root:

```bash
PYTHONPATH=src:. python tests/active_firm_stock_bds_bed/run.py --profile mock --workers 4
PYTHONPATH=src:. python tests/active_firm_stock_bds_bed/run.py --profile smoke --workers 4
```

The full recovery design is intentionally expensive and should be run only after
the smoke report is reviewed:

```bash
PYTHONPATH=src:. python tests/active_firm_stock_bds_bed/run.py --profile full --workers 4
```

Regenerate a report from saved results:

```bash
PYTHONPATH=src:. python tests/active_firm_stock_bds_bed/build_report.py --profile smoke
```

Outputs are written only to `tests/active_firm_stock_bds_bed/results/<profile>/`.

## Recorded status (2026-08-25)

The mock and smoke profiles have been run. Both are **not for inference**. The
smoke state sampler reached maximum R-hat `1.0226`, but minimum bulk ESS was only
`115.1` for `omega`, below the predeclared smoke requirement of `300`. The two
observed NKPC fits themselves mixed adequately (maximum R-hat `1.0074`, minimum
bulk ESS `845.1`), but their free `theta_N` intervals were extremely wide and
included zero.

The recovery diagnostic is the binding economic stop. With the injected cycle
treated as known (`oracle_state`), 10-replicate smoke detection reached 100% only
at `theta_N=50`; it was 70% at `30` and 30% at `20`. After propagating the
external-state posterior (`propagated_state`), detection was zero at every grid
value through `50`. Smoke rates are not power estimates, but this result is too
weak to justify the expensive full profile or any HSA restriction. See
`RESULTS.md` and the seven-page report under `results/smoke/report/`.

## Output inventory

- `SPECIFICATION.md`: frozen equations, ordering, gates, and interpretation rules.
- `RESULTS.md`: machine-generated numerical record for the latest smoke run.
- `results/<profile>/manifest.json`: data hashes, sample, sampler settings, and gates.
- `results/<profile>/state/`: state draws, summaries, and BDS/BED input series.
- `results/<profile>/nkpc/`: free-`theta_N` observed-data diagnostics.
- `results/<profile>/recovery/`: oracle and propagated-state recovery results.
- `results/smoke/report/active_firm_stock_bds_bed_report.pdf`: equation-first report.

## Changelog

- `v1`, 2026-08-25: created the isolated external-`N` bundle and added mock/smoke
  runs, unit tests, recovery modes, report generation, and audit documentation.
- During development, a free BDS measurement-error scale accumulated at its zero
  boundary. It was replaced by the predeclared fixed numerical scale `0.005`,
  treating the published annual BDS firm count as the level anchor. This change
  was made before the recorded smoke result and did not use inflation or the sign
  of `theta_N`.
