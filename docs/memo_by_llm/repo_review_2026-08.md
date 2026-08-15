# Repository review — 2026-08

Scope: the uncommitted / recently added work (`src/nkpc_hsa/nine_cell/`, scripts
`20`–`28`, `dataprep/func_data_build.py`, the new `configs/*.yaml`, and the
`report/` + `output/` layout). Findings are grouped by the four requests:
file layout, function clutter, code mistakes, and econometrics.

Severity: **[H]** should fix before the next production run · **[M]** fix when
convenient · **[L]** cosmetic / naming.

---

## 1. File layout (ファイルの散らばり)

### 1.1 `output/` is a second, non-canonical results tree — **[H]**
`output/pdf/` holds compiled experiment PDFs including `*_TEST.pdf` smoke
builds. CLAUDE.md is explicit: *"New outputs should go under `results/`, not
`references/` or `report/`."* This directory is now git-ignored (`.gitignore`),
but the files should be **moved to `results/nine_cell_design/` (or deleted for
the `_TEST` ones)** and the producing scripts pointed at `results/`.

### 1.2 `report/` violates the "two deliverables only" rule — **[H]**
CLAUDE.md and the README say `report/` holds exactly two deliverables
(`nkpc_hsa_report`, `nkpc_hsa_restriction_report`). It now also contains
`design.tex`, `nine_cell_design_report.tex`, and
`nine_cell_design_final_report.*`. Either (a) promote the nine-cell design to a
declared third deliverable in CLAUDE.md, or (b) treat these `.tex` as build
products and move the compiled PDFs to `results/`.

### 1.3 The restriction report was deleted in the working tree — **[H, verify intent]**
`git status` shows `D report/nkpc_hsa_restriction_report.tex`,
`D report/nkpc_hsa_restriction_report.pdf`, and the two
`report/data_model_estimation.*`. Per CLAUDE.md the restriction report is a
headline F0/U/R1/R2/R3 deliverable. If the deletion is intentional, record why;
if it is fallout from the `dab3bbb update` commit, restore with
`git checkout -- report/nkpc_hsa_restriction_report.tex …`.

### 1.4 Experiment configs / scripts are flat — **[M]**
`configs/` now mixes core specs (`models.yaml`, `priors_*.yaml`, `periods.yaml`)
with seven experiment configs. Proposed:

```
configs/
  models.yaml  priors_*.yaml  periods.yaml        # core, unchanged
  experiments/
    nine_cell_design.yaml  observed_hhi_experiments.yaml
    nolag_price_gap_tests.yaml  markup_*.yaml  error_robustness.yaml
scripts/
  01_..19_...                                     # core pipeline, unchanged
  experiments/
    20_run_nine_cell_design.py … 28_run_markup_feedback_path.py
```

Keep the `NN_` prefixes for ordering; only the folder changes. Update the path
constants in `scripts/_bootstrap.py` and the `configs/experiments/` references.

---

## 2. Function clutter (関数のごちゃつき)

### 2.1 Scripts 22–28 duplicate a ~150-line runner harness — **[M]**
`22_test_observed_hhi_models.py` (799 lines) and
`24_test_nolag_price_gap_models.py` (752) — and 25–28 — share a near-identical
block: `argparse` + `hashlib`/`json` provenance + `shutil` + `subprocess`
LaTeX compile + `ProcessPoolExecutor` fan-out + matplotlib figure builders.
Extract one harness, e.g. `src/nkpc_hsa/nine_cell/experiment_runner.py`:

```python
def run_experiment(*, config, cells, estimate_one, build_report, results_dir, jobs): ...
def stamp_provenance(results_dir, config_path, code_rev) -> dict: ...
def compile_pdf(tex_path) -> Path: ...
```

Each script then becomes a thin `main()` that supplies `estimate_one` and the
report template. This is the single biggest reduction in "散らばり".

### 2.2 `nine_cell/followup.py` (758 lines) is a grab-bag — **[M]**
The name says nothing about contents. Split by responsibility (equivalence
tests / secondary joint / smoke sensitivities) into named modules, or fold into
`inflation.py` + `reporting.py` where each piece belongs.

### 2.3 Scripts named `NN_test_*` are runners, not tests — **[L]**
`22_test_observed_hhi_models.py`, `24_test_nolag_price_gap_models.py` collide
conceptually with `tests/`. Rename to `NN_run_*` / `NN_experiment_*`.

### 2.4 `dataprep/func_data_build.py` — legacy `func_` prefix — **[L]**
Naming smell; rename to `dataprep/build.py`-style when touched (low priority,
many importers).

---

## 3. Code mistakes

### 3.1 `nine_cell/inflation.py` — loop-invariant work inside the draw loop — **[L]**
- L378–380: the `model_seed`/`transform_seed`/`extra_seed` lookup dicts are
  rebuilt on every `chain` iteration. Hoist above the chain loop.
- L376/411: `n_endpoints` is reassigned on every `draw`; only the last survives.
  It is constant across draws given a fixed `endpoint_mask`, so compute it once
  before the loop. Harmless today, fragile if the mask ever becomes draw-varying.

### 3.2 `load_spf_cpi_quarter_ahead_expectations` is added but never wired in — **[M]**
`func_data_build.py` gained the CPI-matched loader (`CPI3`), and a test, but no
caller uses it. `nine_cell/data.py:145` still loads only the GDP-deflator SPF
series. See §4.1 — this is the intended fix, left half-done.

---

## 4. Econometrics

### 4.1 GDP-deflator expectations used as the forward term for CPI **and** PPI — **[H]**
`nine_cell/data.py:159`:

```python
q["expectation"] = pd.to_numeric(q["Epi_spf_gdp_1q_ahead_ann_log"], errors="coerce")
```

Every cell — CPI, core-CPI, PPI — uses the one-quarter-ahead **GDP-deflator**
SPF forecast as `E_t π_{t+1}`. The Phillips-curve outcome is CPI/PPI inflation,
so the forward regressor is a proxy measured with error that is correlated with
the outcome. That biases `beta_f` and, because `x·(q̄−q0)` shares variance with
the expectation term, contaminates the slope `kappa`/`delta` the design is built
to identify. The benchmark already lists this as a limitation
(`inflation.py:513`, "GDP-deflator expectation proxy used for PPI"), so the fix
is intended, not a surprise.

**Recommended fix** (results-changing — do not apply silently):
- CPI / core-CPI cells → use the just-added `CPI3` loader.
- PPI → the SPF has no PPI series; keep the GDP-deflator proxy but flag it as a
  declared limitation, do not present it as identified.

Concrete patch for `nine_cell/data.py`:

```python
from nkpc_hsa.dataprep.func_data_build import (
    load_spf_quarter_ahead_expectations,
    load_spf_cpi_quarter_ahead_expectations,   # add
    resample_quarterly_mean,
)
...
spf_cpi = load_spf_cpi_quarter_ahead_expectations(raw_dir)
spf_cpi.index = spf_cpi.index.to_period("Q")
q = q.join(spf_cpi, how="left")
...
# per-price expectation, matched where a series exists:
q["expectation_gdp"] = pd.to_numeric(q["Epi_spf_gdp_1q_ahead_ann_log"], errors="coerce")
q["expectation_cpi"] = pd.to_numeric(q["Epi_spf_cpi_1q_ahead_ann_log"], errors="coerce")
```

…then have `fit_cut_model` select `expectation_{cpi|gdp}` from the cell's price.
This needs a design decision (do we re-estimate all nine cells?), so I have left
it as a proposal.

### 4.2 No convexity / adding-up discipline on (β_b, β_f) — **[M, modeling choice]**
`inflation.py:_prior_sds` gives lagged inflation `beta_b ~ N(0, 1)` and
expectations `beta_f ~ N(0, 1)`, independent and centered at zero. The hybrid
NKPC normally imposes `beta_b + beta_f ≈ 1` (or at least both in `[0, 1]`).
With these priors the backward/forward split is economically unrestricted and
can wander to signs that are hard to interpret. If that freedom is deliberate
(a reduced-form check) it is fine — but it should be stated, because it weakens
the structural reading of `kappa`.

### 4.3 Q4 anchor uses a near-degenerate observation variance — **[L, acknowledged]**
`markup_measurement.py:173` sets the annual-Q4 anchor observation variance to
`(1e-6 * q_scale)**2` to emulate a hard equality (comment says so). With the
`1e-10` innovation on the carried lag states this is numerically delicate; if
FFBS ever throws a non-PD Cholesky, this is the first place to widen. Not a bug
today.

### 4.4 `CPI3` annualized→log transform is correct — **no action**
`100 * log1p(a/100)` maps the SPF geometric-annualized percent `a` to the
`400·Δlog P` convention used elsewhere (`400·log(1+q) = 100·log((1+q)^4)`).
Verified consistent; noted here so it is not re-flagged.

---

## 5. Proposed `tests/` split (test_src / test_run / test_output)

Current `tests/` (27 files) mixes pure unit tests, pipeline smoke tests, and
report/artifact checks. Proposed layout matching your `test_src / test_run /
test_output` idea:

```
tests/
  conftest.py                      # shared fixtures + PYTHONPATH guard
  unit/         (= test_src)        # pure functions, no disk, fast
    test_transforms.py  test_common.py  test_statistical_updates.py
    test_spf_loaders.py  test_markup_measurement.py  test_feedback.py
    test_identification.py  test_pointwise_loglik.py  test_joint_ffbs.py
    test_conditional_ml.py  test_conditional_ml_comparison.py
    test_error_robustness_ma3.py  test_particle_gibbs_missing_n.py
    test_competition_measurement.py  test_sec_hhi.py
    test_sec_competition_variants.py
  run/          (= test_run)        # pipeline / script smoke, touches data
    test_nine_cell_design.py  test_paths.py  test_prior_wiring.py
    test_observation_design_default.py  test_progress.py
    test_theory_model_migration.py  test_establishment_identification.py
    test_markup_interpolation_sensitivity.py
  output/       (= test_output)     # report inputs / generated artifacts
    test_report_and_robustness.py  test_report_inputs.py
```

Two practical notes:
1. Add a `tests/conftest.py` that inserts `src/` on `sys.path` (or rely on the
   editable install) — the memory note *worktree_import_shadowing* warns that a
   bare `import nkpc_hsa` can hit the wrong checkout; a conftest makes the intent
   explicit.
2. A **lighter alternative** that gets you the same fast/slow separation without
   moving 27 files: keep the flat layout but add pytest markers
   (`@pytest.mark.unit / run / output`) and register them in `pyproject.toml`,
   so `pytest -m unit` runs the fast set. Directories are cleaner for humans;
   markers are cheaper to adopt. Recommend markers first, directories if you
   want the on-disk separation to be visible.

---

## Suggested order of operations
1. Confirm the §1.3 restriction-report deletion (restore if accidental).
2. Decide §4.1 (re-estimate cells with matched CPI expectations?) — biggest
   econometric item.
3. Extract the §2.1 experiment-runner harness (biggest de-clutter).
4. Move `output/` → `results/`, tidy `report/` (§1.1–1.2).
5. Adopt the `tests/` split or markers (§5).
