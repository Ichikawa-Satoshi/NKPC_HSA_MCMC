# HSA deep identification audit

This bundle is the latest identification-first search over economically admissible
competition-state and inflation-error specifications. The protocol and pass/fail
gates were frozen in [`SPECIFICATION.md`](SPECIFICATION.md) before candidate
results were inspected. The complete numerical interpretation is recorded in
[`RESULTS.md`](RESULTS.md).

Current status: **screen, mock, and selected quick diagnostics only; not a promoted
full estimate and not for structural inference**. No candidate passed the complete
sequence, so formal marginal likelihood was not run.

## What is estimated

The total competition measure is observed without an extra measurement-error term:

```math
N_t=\bar N_t+\hat N_t,
\qquad
\sigma_{\bar N}^2=\omega\tau^2,
\qquad
\sigma_{\hat N}^2=(1-\omega)\tau^2.
```

The leading S1 state law assigns the slow innovation across quarters using the
measurement-only average Capital IQ allocation profile, while Gustavo fixes each
annual Q4 total. The cycle is a stationary AR(2):

```math
\hat N_t=2r\cos(2\pi/P)\hat N_{t-1}-r^2\hat N_{t-2}+\eta^h_t.
```

The free static NKPC is

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
+(\kappa_0+\delta\bar N_t)x_t-\theta\hat N_{t-j}+\varepsilon_t.
```

The restricted static model replaces `delta` with `lambda*theta`. Confirmatory
fixed values are `lambda in {3,6,9}`; free lambda is only an identification
diagnostic because, near `theta=0`, the likelihood mostly sees the product.

Overlapping YoY inflation uses an estimated MA(3) disturbance. The QoQ diagnostic
uses genuine one-quarter-ahead SPF expectations and a non-overlapping quarterly
likelihood. PPI and core CPI are never pooled.

## Tests performed

1. A 1,296-row deterministic candidate screen over state, price, activity,
   timing, frequency, error law, and sample split.
2. A 192-row free-versus-HSA dynamic path screen.
3. A Q4-only non-overlapping YoY discovery/validation diagnostic.
4. Exact-N joint MA(3) sampling under the quarterly-local-level AR(2) and
   annual-allocation AR(2) state laws.
5. Exact-N joint QoQ sampling for four price/activity cells.
6. Simulation recovery for the free static sampler and state parameters.
7. No-intercept, `alpha_b+alpha_f=1`, fixed-lambda, and free-lambda diagnostics.
8. Dense-Gaussian FFBS equivalence tests for the sampler implementation.

## Exact commands

Run from the repository root. The screen scripts overwrite their corresponding
CSV files; the estimation scripts write to the selected profile directory.

```bash
PYTHONPATH=src:. python tests/hsa_deep_identification/screen.py
PYTHONPATH=src:. python tests/hsa_deep_identification/dynamic_screen.py
PYTHONPATH=src:. python tests/hsa_deep_identification/nonoverlap_screen.py
```

Reproduce the saved simulation-recovery mock and the four-cell QoQ mock:

```bash
PYTHONPATH=src:. python tests/hsa_deep_identification/simulation_recovery.py --profile mock
PYTHONPATH=src:. python tests/hsa_deep_identification/run_qoq.py \
  --profile mock \
  --architectures annual_allocation_ar2 \
  --models ces free
```

Reproduce the selected quick MA(3) results used in the report:

```bash
PYTHONPATH=src:. python tests/hsa_deep_identification/run_joint.py \
  --profile quick \
  --architectures annual_allocation_ar2 \
  --cells ppi_inverse_markup \
  --models free hsa6

PYTHONPATH=src:. python tests/hsa_deep_identification/run_joint.py \
  --profile quick \
  --architectures annual_allocation_ar2 \
  --cells ppi_negative_unemployment_gap \
  --models ces
```

Validate the FFBS implementation and rebuild the English report:

```bash
pytest -q tests/hsa_deep_identification/test_joint_ma3.py
PYTHONPATH=src:. python tests/hsa_deep_identification/build_report.py
```

The report is written to
`output/pdf/hsa_deep_identification_report.pdf`.

The scripts also accept `--profile full`, but that option is intentionally not
the current recommended command. Under the frozen protocol a full run is launched
only after a candidate passes screening, recovery, short-run convergence,
identification, and validation. No candidate has done so.

## Saved-output inventory

- `results/screen/`: candidate, dynamic, non-overlap, free-channel, and theory
  restriction screens plus the screen manifest.
- `results/mock/simulation_recovery.json`: mock recovery summary.
- `results/mock/joint_qoq_iid/`: four-cell QoQ mock output.
- `results/mock/joint_ma3/`: broad MA(3) mock grid and restriction sensitivities.
- `results/quick/joint_ma3/`: selected annual-allocation AR(2) quick fits.
- `output/pdf/hsa_deep_identification_report.pdf`: equation-first English report.

## Decision

The annual-allocation AR(2) law fixes the earlier excessive slow-innovation share,
and the MA(3) sampler passes its algorithmic unit tests. It does not identify the
free direct coefficient. Fixed lambda only transfers partial slope information
into theta, while free lambda remains prior-wide. The current result is therefore
“state stabilization succeeded; HSA structural identification failed,” not a
preferred HSA specification.
