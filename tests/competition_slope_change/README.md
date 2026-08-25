# Competition-linked structural slope change test

This bundle implements the semi-structural baseline in `SPECIFICATION.md`:

1. estimate the exact competition slow/cycle decomposition using competition
   data only;
2. propagate state draws into a slope-only MA(3) NKPC;
3. calculate historical `kappa_t`, competition-induced slope changes, and a
   fixed-competition counterfactual;
4. estimate `theta_C` only as a current/lag/lead diagnostic; and
5. generate an English equation-first report directly from saved results.

The empirical competition variable is an effective-firm/concentration coordinate
`C`, not automatically the theoretical active-firm mass `N`.

## Commands

Run from the repository root.

```bash
PYTHONPATH=src:. python tests/competition_slope_change/run.py --profile smoke --workers 4
PYTHONPATH=src:. python tests/competition_slope_change/build_report.py --profile smoke
```

After the smoke gates pass:

```bash
PYTHONPATH=src:. python tests/competition_slope_change/run.py --profile full --workers 4
PYTHONPATH=src:. python tests/competition_slope_change/build_report.py --profile full
```

Results are written to `tests/competition_slope_change/results/<profile>/`. The
full PDF is also copied to
`output/pdf/competition_slope_change_report.pdf`.

## Status

Implemented and run at both smoke and full profiles on 2026-08-25. The full
computational gate passed (maximum primary R-hat 1.0012; minimum bulk ESS 4187.6;
minimum tail ESS 4984.2; exact-accounting error `2.22e-16`). This does not imply
structural identification: every `delta` 95% interval includes zero, `theta_C`
is essentially prior-wide, and the slow/cycle variance allocation is sensitive
to the omega prior. See `RESULTS.md` for the generated numerical record and
`results/full/report/competition_slope_change_report.pdf` for the report.

`RESULTS.md` is generated from saved CSV/JSON output after estimation and is not
filled by hand.
