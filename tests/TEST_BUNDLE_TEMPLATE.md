# Required record for every new test bundle

Copy this file to `tests/<new_test>/README.md` when a test folder is created and
complete it as the work progresses. A test is not complete when only code and
figures exist; the folder must explain what was tested, what was held fixed, and
what the saved result permits us to conclude.

## 1. Question and status

- Research question:
- Null/nested comparison:
- Why this test is needed:
- Current status: `planned`, `mock only`, `quick diagnostic`, `full empirical`,
  `failed`, or `superseded`.
- Replaces/supersedes:

State prominently whether the saved output is **not for inference**.

## 2. Model and equations

Write the actual likelihood and state equations implemented by this folder. Do
not refer only to a model nickname. Define every state, coefficient, timing index,
and sign convention. For an HSA model, state whether lambda is fixed or estimated
and show the mapping between free and restricted coefficients.

```math
N_t=\bar N_t+\hat N_t,
```

```math
\pi_t=a+\alpha_b\pi_{t-1}+\alpha_fE_t\pi_{t+1}
+\kappa_t x_t-\theta_t\hat N_{t-j}+\varepsilon_t.
```

Also record the inflation disturbance law. Overlapping YoY inflation must not be
described as iid if an MA(3) likelihood is actually required or implemented.

## 3. Data and frozen transformations

- Price series and frequency:
- Activity/slack series:
- Expectations series and forecast horizon:
- Competition sources:
- Sample start/end and number of observations:
- Missing-data treatment:
- Centering/scaling:
- Timing (`current`, `lag1`, distributed, or other):
- Data file hashes or immutable input identifiers:

Explain which transformations were chosen before results were inspected. Label
post-result sensitivity choices as such.

## 4. What changes and what is held fixed

List the exact experimental cells. A reader must be able to reconstruct the
comparison without reading source code.

| Dimension | Values tested | Held fixed across comparison? |
|---|---|---|
| Price | | |
| Activity | | |
| State law | | |
| Error law | | |
| HSA restriction | | |
| Timing | | |

## 5. Estimands, priors, and expected signs

| Parameter/function | Meaning | Prior | Theory sign | Identification criterion |
|---|---|---|---|---|
| `theta` | Direct competition channel | | positive | |
| `delta` | Slow-state slope channel | | | |
| `lambda` | HSA proportionality coefficient | | unrestricted or positive | |
| `kappa_t` | Time-varying NKPC slope | derived | positive path | |
| `omega` | Slow innovation variance share | | none | |

Support restrictions such as positive variance priors are not empirical evidence
for a theoretical sign. If lambda is fixed, explicitly say that theta can borrow
identification from the slope channel.

## 6. Sampling profiles

| Profile | Iterations | Warmup | Thin | Chains | Purpose |
|---|---:|---:|---:|---:|---|
| mock | | | | | Code-path check only |
| quick | | | | | Diagnostic screening |
| full | | | | | Empirical inference, if all gates pass |

Record seeds, particle counts, parallelism, and any adaptive settings. Never call
a mock/quick result a full estimate.

## 7. Gates declared before estimation

At minimum report:

- maximum rank-normalized R-hat;
- minimum bulk and tail ESS across coefficients and state parameters;
- exact-identity error when `N=Nbar+Nhat` is imposed;
- posterior mean, 95% interval, theory-sign probability, and posterior/prior SD
  ratio for every claimed structural parameter;
- pathwise sign probability for time-varying `kappa_t` and `theta_t`;
- simulation-recovery coverage and boundary behavior for a new sampler/state law;
- predictive comparison and a valid integrated likelihood method only after
  convergence and identification pass.

Write the numeric thresholds here before inspecting the final run.

## 8. Exact commands

Commands must work from the repository root and use the real `tests/` path.

```bash
PYTHONPATH=src:. python tests/<new_test>/run.py --profile mock
PYTHONPATH=src:. python tests/<new_test>/run.py --profile quick
PYTHONPATH=src:. python tests/<new_test>/run.py --profile full
```

Include separate validation and report commands if they exist. Do not document a
script that is absent from the folder.

## 9. Output inventory

Record the exact locations of:

- manifest/config snapshot;
- coefficient table;
- prior/posterior table;
- convergence table;
- state draws and `Nbar`/`Nhat` figures;
- model-comparison output;
- simulation recovery;
- report PDF.

The manifest should contain, where applicable: revision, profile, `is_test_run`,
creation time, git revision, dirty-tree flag, data/config hashes, sample, sampling
settings, seeds, convergence gates, identification gates, pass/fail status, and
report path.

## 10. Results

Populate this section after every completed mock, quick, or full run.

### Run identity

- Profile and timestamp:
- Manifest:
- Number of fitted models:
- Whether all planned cells completed:

### Numerical result

Report posterior mean and interval together. Report prior/posterior shrinkage and
sign probability separately; do not hide them in plots.

| Cell/model | Parameter | Mean | 95% interval | P(theory sign) | Post/prior SD | R-hat | Bulk ESS | Tail ESS |
|---|---|---:|---|---:|---:|---:|---:|---:|
| | | | | | | | | |

### Gate decision

| Gate | Threshold | Observed | Pass? |
|---|---|---|---|
| Convergence | | | |
| Structural identification | | | |
| State identification | | | |
| Economic sign/path | | | |
| Validation/recovery | | | |
| Model evidence | | | |

### Interpretation

Write three separate statements:

1. What the test establishes.
2. What it does not establish.
3. Whether it changes the current preferred specification.

If the run failed, retain the failure and explain the cause. Do not overwrite it
with a later successful configuration; add a dated run entry or a new results
subdirectory.

## 11. Limitations and next admissible step

Describe weak identification, collinearity, short samples, prior sensitivity,
state uncertainty, frequency mismatch, and any unavailable formal comparison.
The next step must follow from the failure mode; it must not be chosen because it
is likely to produce a desired sign.

## 12. Changelog

| Date | Change | Reason | Affects interpretation? |
|---|---|---|---|
| | | | |
