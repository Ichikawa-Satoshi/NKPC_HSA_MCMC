# Estimation specification — NKPC / HSA project

**What this is.** A complete, code-verified description of what the production estimation code
actually computes. The repository is the source of truth. Where the code and the report
(`report/nkpc_hsa_report.tex`) disagree, both are shown and the discrepancy is labelled; the
documentation follows the **code**.

**Companion documents**
- `docs/data_dictionary.md` — every observed series
- `docs/estimation_flow.md` — one full MCMC iteration, per model
- `docs/code_equation_crosswalk.md` — compact equation ↔ code table

**How to read this** — see §P at the end.

---

## A. Executive map of the estimation system

```
data/raw/*                     raw CSV/XLSX
    │  func_data_build.build_dataset  +  build.add_hp_output_gap / add_labor_share_gap
    ▼
data/processed/model_ready.csv        451 quarters, 40 columns
    │  configs/models.yaml : data_specs.<name> picks 6 columns
    │  wrappers._coerce_model_data    joint dropna  →  T = 124 (1982Q1–2012Q4)
    ▼
model_data = {pi, pi_prev, pi_expect, x, x_prev, N}
    │  transforms.transform_competition_series   N → (100 log N − mean)/10
    │  competition.build_competition_observation PCHIP (all 124) or annual-Q4 (31 + 93 nan)
    ▼
wrappers.run_model(model, …)
    │  models.common.prior_specs_to_internal    configs/priors_*.yaml → sampler units
    │  dispatch table  wrappers._run_sampler:491-500
    ├── ces             → gibbs/ces/model.func_nkpc_ces                    conjugate Gibbs
    ├── hsa_steady      → gibbs/hsa_steady/model.func_nkpc_hsa_decomp_tv_kappa_kalman
    ├── hsa_dynamic     → gibbs/hsa_dynamic/model.func_nkpc_hsa_decomp
    ├── hsa_const_theta → gibbs/hsa_const_theta/model.func_nkpc_hsa_const_theta
    └── hsa_full        → gibbs/hsa_full_pg/model.func_nkpc_hsa_full_pg   (Particle Gibbs)
    ▼
results/runs/<model>_<spec>_<prior>_<run_id>/{posterior.nc, metadata.json, priors.json}
    │  scripts/12_build_cpi_ppi_report.load_report_runs   (one call per observation design)
    ▼
report/generated/tables/             interpolated (PCHIP) tables
report/generated/tables/annual_q4/   mixed-frequency tables  ← MAIN results
    ▼
report/nkpc_hsa_report.tex
    §4-§6, §9-§10 and Appendix A/C  read  annual_q4/   (main)
    §7 and Appendix B               read  the base dir (comparison)

The report is English-only and the tables are written in English at source; there is no
translation pass. ``_write_latex`` refuses to emit a table containing CJK text.
```

### The five models at a glance

| Model | κ_t | θ_t | Latent states | State sampler | Production entry point |
|---|---|---|---|---|---|
| CES | κ (constant) | — (no N term) | none | conjugate Gibbs | `gibbs/ces/model.py:58` |
| HSA steady | κ₀ + δ·N̄_t | 0 | (N̂_t, N̂_{t−1}, N̄_t) | **exact joint FFBS** | `gibbs/hsa_steady/model.py:401` |
| HSA dynamic | κ (constant) | θ (constant) | (N̂_t, N̂_{t−1}, N̄_t) | **exact joint FFBS**, correlated-shock form | `gibbs/hsa_dynamic/model.py:1027` |
| HSA const-theta | κ₀ + δ·N̄_t | θ (constant) | (N̂_t, N̂_{t−1}, N̄_t) | **exact joint FFBS** | `gibbs/hsa_const_theta/model.py:67` |
| HSA full | κ₀ + δ·N̄_t | θ₀ + γ·N̄_t | (N̂_t, N̂_{t−1}, N̄_t) | **Particle Gibbs** (conditional SMC, P=512) | `gibbs/hsa_full_pg/model.py:341` |

### Code-status classification

| Status | Code |
|---|---|
| **Production** | `gibbs/ces`, `gibbs/hsa_steady`, `gibbs/hsa_dynamic`, `gibbs/hsa_const_theta`, `gibbs/hsa_full`, `gibbs/common/joint_ffbs.py`, `gibbs/common/competition.py`, `gibbs/common/constraints.py`, `inference/wrappers.py`, `models/common.py` |
| **Production (`hsa_full`)** | `gibbs/hsa_full_pg/model.py` — `run_model("hsa_full")` dispatches here for **both** observation designs, via the facade `models/hsa_full.py`. Particle count comes from `configs/models.yaml → defaults.n_particles` (512) and is recorded in run metadata. |
| **Validation only** | `sample_states_joint_ffbs_gamma0` (`gibbs/hsa_full_pg/model.py:240`) — the γ=0 benchmark used to check Particle Gibbs and, in `tests/test_joint_ffbs.py`, the shared joint FFBS |
| **Legacy / deprecated** | `gibbs/hsa_full/model.py:func_nkpc_hsa_full` — the superseded **alternating-FFBS** state sampler, still importable as `models.hsa_full.func_nkpc_hsa_full_alternating_ffbs` for validation but no longer reachable from `run_model`; its helper functions are still imported by `hsa_full_pg`. `scripts/appendix_pg_full_runs.py` (retired: the monkeypatch it applied is obsolete). `gibbs/gibbs_wrappers.py` (emits `DeprecationWarning`; uses `100·log N`, **not** the production `log100_centered10`), `func_nkpc_hsa_full_static_theta` (old alternating-FFBS const-theta, kept as an alias), `func_nkpc_hsa_decomp_tv_theta_kappa` (raises `NotImplementedError`), **`gibbs/gibbs_ces.py`** — a second, older CES sampler that returns a `"deprecated"` key in its own output (`:206-209`) and is imported by nothing |
| **Unused** | `_sample_ar2_states_ffbs`, `_sample_rw_states_ffbs` (`gibbs/hsa_full/model.py:86, 265`) — superseded by their `_tv_theta` variants; re-exported by `gibbs/common/state_space.py` but called by nothing. `gibbs/gibbs_utils.py` and `gibbs/gibbs_notebook_utils.py` survive only through the thin re-export shims `gibbs/common/math.py` and `gibbs/common/notebook.py` |
| **Retired** | `scripts/appendix_pg_full_tables.py`, `scripts/appendix_pg_full_runs.py`, `scripts/build_english_tables.py` — all no-op stubs. The first two are obsolete now that Particle Gibbs is the dispatched sampler; the third is obsolete now that the tables are English at source and the Japanese edition of the report has been removed. |

---

## B. Data dictionary

See `docs/data_dictionary.md`.

---

## C. Common transformations and notation

All four transformations below are computed **inside every sampler**, not in the data pipeline.

### C.1 Inflation net of expectations

**Mathematical form**  `y_t = π_t − E_t π_{t+1}`

**Actual code** (`gibbs/hsa_steady/model.py:562`, identically in every model)
```python
y = pi_t - pi_expect
```

**Why** Moves the forward-looking term to the left-hand side so the inflation equation becomes a
linear regression. Starting from `π_t = α π_{t−1} + (1−α) E_tπ_{t+1} + κ_t x_t + e_t` and
subtracting `E_tπ_{t+1}` from both sides gives `y_t = α a_t + κ_t x_t + e_t`.

**Units** annual percentage points.

**Restriction this embeds** The coefficients on `π_{t−1}` and `E_tπ_{t+1}` are forced to sum to
one (α and 1−α). This is a vertical-long-run-Phillips-curve restriction; it is **not** implied by
the theoretical equation in report §1.1, which has `β(1−δ_exit) < 1` on the expectation term and
no lagged term at all. Report §1.4 lists "we add backward-looking inertia (α)" but does not flag
the sum-to-one restriction.

### C.2 Inertia regressor

**Mathematical form**  `a_t = π_{t−1} − E_t π_{t+1}`

**Actual code** (`gibbs/hsa_steady/model.py:561`)
```python
a_t = pi_tm1 - pi_expect
```

**Why** The regressor whose coefficient is α, in the same net-of-expectations space as `y_t`.

### C.3 Activity innovation

**Mathematical form**  `ζ_t = x_t − φ_1 x_{t−1}`

**Actual code** (`gibbs/hsa_steady/model.py:630`, recomputed at `:735`)
```python
zeta = x_t - phi_1 * x_tm1
```

**Why** `x_t` follows an AR(1), `x_t = φ_1 x_{t−1} + ζ_t`. The NKPC shock is decomposed as
`e_t = λ_eζ ζ_t + η_t`, so `ζ_t` is a regressor in the inflation equation. This lets the
contemporaneous activity shock correlate with the inflation shock instead of treating `x_t` as
exogenous. Note `ζ_t` is recomputed after `φ_1` is redrawn, so the value used for `σ_ζ²`, `σ_η²`
and the state block reflects the **current** iteration's `φ_1`.

### C.4 Firm-count transform

**Mathematical form**  `N^obs_t = (100 log N_t − mean_t[100 log N_t]) / 10`

**Actual code** (`dataprep/transforms.py:44-46`)
```python
if transform == "log100_centered10":
    raw = 100.0 * np.log(arr)
    return (raw - float(np.mean(raw))) / 10.0
```

**Why / units** One unit of `N̄` or `N̂` is a **ten-log-point** deviation from the sample mean.
Coefficients on `N̄`/`N̂` (`δ`, `θ`, `θ₀`, `γ`) are therefore *per ten log points*. Report §2
states this; the values in tables are already on this scale and must not be rescaled again.

### C.5 KAPPA_SCALE — the one genuinely confusing convention

`KAPPA_SCALE = 100.0` (`gibbs/hsa_steady/model.py:15`, and identically in every model module).

The samplers regress on `x_t / 100` rather than `x_t`:

```python
X = np.column_stack(
    [
        a_t,
        x_t / KAPPA_SCALE,
        (x_t * Nbar) / KAPPA_SCALE,
    ]
)                                                     # hsa_steady/model.py:632-638
```

so the *internal* coefficients are 100× the physical ones. Three places must agree, and do:

| | Conversion | Code |
|---|---|---|
| **Prior in** | physical → internal, ×100 | `models/common.py:183-190` `kappa_prior_physical_to_internal` |
| **Regressor** | `x → x/100` | each model's `X` block |
| **Draw out** | internal → physical, ÷100 | e.g. `kappa0_draws[i] = kappa0 / KAPPA_SCALE` (`hsa_steady:861-862`) |

Applies to `kappa`, `kappa_0`, `delta` **only**. `theta`, `theta_0`, `gamma` multiply *unscaled*
regressors (`-Nhat`, `-(Nhat*Nbar)`) and are therefore **not** rescaled — see
`models/common.py:167-181` where they go through the plain `pair()` path. `tests/test_prior_wiring.py::test_gamma_prior_is_not_kappa_scaled` pins this.

**Everything stored in `posterior.nc` is in physical units.** Where a sampler passes coefficients
into a state routine it converts explicitly at the call site, e.g. `hsa_const_theta/model.py:314`:
```python
h_nbar=(delta / KAPPA_SCALE) * x_t,
```

### C.6 Sign conventions

- `x_t = u* − u` (negative unemployment gap): positive in booms. **Positive `κ` = conventional
  downward-sloping Phillips curve.**
- The entry channel enters as `−θ_t N̂_t`. The regressor is coded `-Nhat`, so the stored `θ`
  is the coefficient in `−θ N̂`. Report §1.1's theory implies `θ > 0`; the report never states
  this prediction (see §N).

---

## D. State-space system

Shared by HSA steady, dynamic, const-theta and full.

### State vector
```
s_t = (N̂_t, N̂_{t−1}, N̄_t)'
```
The middle element carries the AR(2) lag so the transition is first-order. `states[0,1]` is the
sampled `N̂_{−1}`.

### Transition  `s_t = c + F s_{t−1} + ω_t`

```
      ⎡ρ₁  ρ₂  0⎤        ⎡0⎤        ⎡σ_u²  0  0   ⎤
F  =  ⎢ 1   0  0⎥ ,  c = ⎢0⎥ ,  Q = ⎢ 0    0  0   ⎥
      ⎣ 0   0  1⎦        ⎣n⎦        ⎣ 0    0  σ_ε²⎦
```

**Actual code** (`gibbs/common/joint_ffbs.py:111-120`)
```python
F = np.array([[rho1, rho2, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]], dtype=float)
c = np.array([0.0, 0.0, n_drift], dtype=float)
Q = np.diag([sigma_u2, 0.0, sigma_eps2])
```
Row 2 of `Q` is exactly zero: `N̂_{t−1}` is a deterministic copy, not a shock.

### Observation system

Two rows, the second of which is model-specific:

```
⎡N^obs_t ⎤   ⎡  1        0    1      ⎤          ⎡σ_N²   0   ⎤
⎢        ⎥ = ⎢                       ⎥ s_t + v, ⎢           ⎥
⎣ ỹ_t    ⎦   ⎣h_nhat_t   0  h_nbar_t ⎦          ⎣ 0    σ_η² ⎦
```

**Actual code** (`joint_ffbs.py:137-146`)
```python
pi_row = np.array([h_nhat[t], 0.0, h_nbar[t]], dtype=float)
if np.isfinite(N_obs[t]):
    y_obs = np.array([N_obs[t], y_tilde[t]], dtype=float)
    H = np.vstack([[1.0, 0.0, 1.0], pi_row])
    R = np.diag([sigma_N2, sigma_eta2])
else:
    y_obs = np.array([y_tilde[t]], dtype=float)
    H = pi_row.reshape(1, 3)
    R = np.array([[sigma_eta2]], dtype=float)
```

**Element by element**

| Element | Value | Equation term |
|---|---|---|
| `H[0,0] = 1` | 1 | `N̂_t` in `N^obs_t = N̄_t + N̂_t + ν_t` |
| `H[0,2] = 1` | 1 | `N̄_t` in the same equation |
| `H[1,0] = h_nhat_t` | 0 (steady) / `−θ₀` (const-theta) | `−θ₀ N̂_t`, the entry channel |
| `H[1,2] = h_nbar_t` | `δ x_t / 100` | `δ N̄_t x_t`, the slope's dependence on the trend |
| `R[0,0] = σ_N²` | | firm-count measurement error `ν_t` |
| `R[1,1] = σ_η²` | | inflation-equation residual `η_t` |

`ỹ_t` is the inflation observation net of everything not loading on the state:

```python
        y_tilde_state = y - alpha * a_t - (kappa0 / KAPPA_SCALE) * x_t - lambda_ez * zeta
                                              # hsa_const_theta/model.py:310
```

### Initial state
```
s_0 ~ N(m₀, P₀),   m₀ = (0,0,0)',   P₀ = 10·I₃
```
From `_common_priors` defaults (`hsa_steady/model.py:262-268`); overridable via `opts` but never
overridden in production. **`N^obs_0` is deliberately not substituted into `m₀`** — the period-0
firm count enters through the ordinary observation update instead, so it is not used twice. Report
§3.3 states the same.

### Missing observations
Only the firm-count row is dropped (`np.isfinite(N_obs[t])` above). Prediction and the inflation
row run at **every** quarter. Setting all 124 quarters finite recovers the PCHIP filter exactly;
annual-Q4 is that filter with 93 firm-count rows removed.

---

## E. Model-by-model specifications

### E.1 CES

| | |
|---|---|
| **Production code** | `src/nkpc_hsa/gibbs/ces/model.py`, `func_nkpc_ces` (line 58) |
| **Estimates** | α, κ, φ_1, λ_eζ, σ_ζ², σ_η² (stored: `alpha, kappa, phi_1, lambda_ez, sigma_e2, sigma_zeta2, rho`) |
| **Latent states** | none |
| **State sampler** | n/a |
| **Parameter sampler** | conjugate Gibbs, 4 blocks |

**Inflation equation**
```
y_t = α a_t + κ x_t + λ_eζ ζ_t + η_t ,      η_t ~ N(0, σ_η²)
x_t = φ_1 x_{t−1} + ζ_t ,                   ζ_t ~ N(0, σ_ζ²)
```

**Actual code** (`ces/model.py:159`)
```python
X = np.column_stack([a_t, x_t / KAPPA_SCALE])
```
Only α and κ are in the joint block; λ_eζ and φ_1 are drawn separately (see §F.1).

CES has **no** firm-count term and no latent state, so its likelihood and posterior are invariant
to the PCHIP-vs-annual-Q4 choice. The report reuses one set of 16 CES cells across both designs
(`scripts/12_build_cpi_ppi_report.py:1271-1277`).

**Stored variance convention**: CES stores `sigma_e2` and `sigma_zeta2` as **variances**; the HSA
models store `sigma_e`, `sigma_zeta`, … as **standard deviations**. `_extract_draws_from_result`
(`wrappers.py:313-316`) square-roots the CES ones on load so the saved `posterior.nc` is uniform.

---

### E.2 HSA steady — the main model

| | |
|---|---|
| **Production code** | `gibbs/hsa_steady/model.py`, `func_nkpc_hsa_decomp_tv_kappa_kalman` (line 401); alias `..._noerror` (line 946) is what `wrappers` dispatches to |
| **Estimates** | α, κ₀, δ, φ_1, λ_eζ, ρ₁, ρ₂, n, σ_η², σ_ζ², σ_u², σ_ε², σ_N² |
| **Latent states** | N̄_{0:T−1}, N̂_{0:T−1} (+ N̂_{−1}) |
| **State sampler** | **exact joint FFBS** — `common/joint_ffbs.py` |
| **θ** | restricted to 0: `N̂` does **not** enter the inflation equation |

**Inflation equation**
```
y_t = α a_t + (κ₀ + δ N̄_t) x_t + λ_eζ ζ_t + η_t
```

**Actual code** (`hsa_steady/model.py:632-638`)
```python
X = np.column_stack([
    a_t,
    x_t / KAPPA_SCALE,
    (x_t * Nbar) / KAPPA_SCALE,
])
```
| column | regressor for |
|---|---|
| `a_t` | α |
| `x_t / 100` | κ₀ (internal units) |
| `(x_t * Nbar) / 100` | δ (internal units) |

**Observation loading**: `h_nhat = 0`, `h_nbar = δ x_t / 100` (`hsa_steady/model.py:386-387`).

**Why linear-Gaussian**: with θ=0 the only state entering the inflation row is `N̄_t`, linearly,
with a coefficient that depends on data (`x_t`) but not on the state. Hence one exact joint draw.

---

### E.3 HSA dynamic

| | |
|---|---|
| **Production code** | `gibbs/hsa_dynamic/model.py`, `func_nkpc_hsa_decomp` → `func_nkpc_hsa_decomp_joint_fullSigma` (line 1027) |
| **Estimates** | α, κ, θ, φ_1, ρ₁, ρ₂, n, σ_N², and a 4×4 shock covariance Σ over `r_t = (e_t, ζ_t, u_t, ε_t)'` |
| **δ, γ** | restricted to 0: constant slope |
| **State sampler** | exact joint FFBS with **correlated** state/measurement shocks — `_sample_states_joint_ffbs_fullSigma` (line 817) |

**Inflation equation**
```
y_t = α a_t + κ x_t − θ N̂_t + e_t
```

**Actual code** (`hsa_dynamic/model.py:599-605`)
```python
X = np.column_stack([
    a_t,
    x_t / KAPPA_SCALE,
    -Nhat,
])
```

**This model is structurally different from the others.** Instead of the `e = λζ + η`
decomposition it carries a full covariance `Σ = Var(e, ζ, u, ε)` and draws each block from the
*conditional* Gaussian given the other three shocks — `_conditional_e_all` (line 361),
`_sample_beta_gaussian_weighted` (line 103). Under the production default
`covariance_structure = "e_zeta_only"` (`configs/models.yaml:11`), `_restrict_sigma_structure`
(line 45) keeps only the (e,ζ) off-diagonal:

```python
if structure == "e_zeta_only":
    out = np.diag(np.diag(S))
    out[0, 1] = out[1, 0] = S[0, 1]
    return _force_pd(out)
```

so `cov(u,ε) = cov(e,u) = cov(e,ε) = 0` and the FFBS cross-covariance block `C_base`
(lines 933-940) is zero. The filter then reduces to the same form as §D, with `H = [[1,0,1],
[−θ,0,0]]` and `R = diag(σ_N², var_e)` (lines 913-921, 977-983). Critically, `_sample_Sigma`
(line 301) draws the restricted parameters **directly on the constrained space** rather than
zeroing an unconstrained inverse-Wishart draw — report §3.7 claims exactly this, and it is true.

Prior on Σ: inverse-Wishart `IW(ν_Σ, S_Σ)` with `ν_Σ = 8`, `S_Σ = diag(3, 3, 0.06, 0.03)`
(`configs/priors_baseline.yaml:30-35`). The `u` and `ε` rows are in **squared ten-log-point**
units, hence 0.06/0.03 rather than O(1).

---

### E.4 HSA const-theta

| | |
|---|---|
| **Production code** | `gibbs/hsa_const_theta/model.py`, `func_nkpc_hsa_const_theta` (line 67) |
| **Estimates** | α, κ₀, δ, θ, φ_1, λ_eζ, ρ₁, ρ₂, n, σ_η², σ_ζ², σ_u², σ_ε², σ_N² |
| **γ** | restricted to 0 |
| **State sampler** | **exact joint FFBS** (`common/joint_ffbs.py`) |
| **Legacy** | `func_nkpc_hsa_full_static_theta` (`gibbs/hsa_full/model.py:960`) is the *old* alternating-FFBS implementation of the same model, retained as a deprecated alias only |

**Inflation equation**
```
y_t = α a_t + (κ₀ + δ N̄_t) x_t − θ N̂_t + λ_eζ ζ_t + η_t
```

**Actual code** (`hsa_const_theta/model.py:190-197`)
```python
X = np.column_stack([
    a_t,
    x_t / KAPPA_SCALE,
    (x_t * Nbar) / KAPPA_SCALE,
    -Nhat,
])
```
| column | regressor for | equation term |
|---|---|---|
| `a_t` | α | `α a_t` |
| `x_t / 100` | κ₀ | `κ₀ x_t` |
| `(x_t * Nbar) / 100` | δ | `δ N̄_t x_t` |
| `-Nhat` | θ | `−θ N̂_t` |

**Observation loading** (`hsa_const_theta/model.py:314-315`)
```python
h_nhat=np.full(T, -theta, dtype=float),
h_nbar=(delta / KAPPA_SCALE) * x_t,
```
so `H[1,·] = [−θ, 0, δ x_t/100]`.

**Why linear-Gaussian, and why this matters.** With γ=0 both state entries in the inflation row
have coefficients that depend on parameters and data but **not on the other state**, so the joint
system is linear-Gaussian and one exact FFBS sweep draws the whole path. Earlier revisions used
`hsa_full`'s two alternating blocks; because `N^obs = N̄ + N̂ + ν` pins the *sum* almost exactly
(posterior corr(N̄₀, N̂₀) ≈ −0.999) a two-block Gibbs moves the shared level with autocorrelation
ρ² ≈ 0.998 per sweep and did not mix. The module docstring records this.

---

### E.5 HSA full

| | |
|---|---|
| **Production code** | `gibbs/hsa_full_pg/model.py`, `func_nkpc_hsa_full_pg` (line 341) — **Particle Gibbs**, both observation designs |
| **Facade** | `models/hsa_full.py` re-exports it as `func_nkpc_hsa_full`; `run_model` imports from there |
| **Particle count** | `configs/models.yaml → defaults.n_particles` = 512, threaded through `run_model(n_particles=…)` → `opts["n_particles"]` |
| **Superseded** | `gibbs/hsa_full/model.py`, `func_nkpc_hsa_full` (line 591) — alternating FFBS; importable as `func_nkpc_hsa_full_alternating_ffbs` for validation only |
| **Estimates** | α, κ₀, δ, θ₀, γ, φ_1, λ_eζ, ρ₁, ρ₂, n, σ_η², σ_ζ², σ_u², σ_ε², σ_N² |

**Inflation equation**
```
y_t = α a_t + (κ₀ + δ N̄_t) x_t − (θ₀ + γ N̄_t) N̂_t + λ_eζ ζ_t + η_t
```

**Actual code** (`hsa_full/model.py:710-733`)
```python
columns = [
    a_t,
    x_t / KAPPA_SCALE,
    (x_t * Nbar) / KAPPA_SCALE,
    -Nhat,
]
...
if not static_theta:
    columns.append(-(Nhat * Nbar))
```
The fifth column `−(N̂_t N̄_t)` is the regressor for γ.

**Why NOT jointly linear-Gaussian.** The term `−γ N̄_t N̂_t` is **bilinear in the state**: the
coefficient on `N̂_t` depends on `N̄_t`, which is itself a state. `H_t` would have to depend on
`s_t`, which the Kalman recursion forbids. Hence either alternating conditional blocks (each of
which *is* linear-Gaussian given the other) or a Particle-Gibbs sweep.

**Which code runs where.** One path, both designs:

| Path | Sampler | Where it lands |
|---|---|---|
| `run_model("hsa_full", …)`, any `competition_measurement` | Particle Gibbs | `results/runs/` |
| Report PCHIP tables | Particle Gibbs | ✅ |
| Report annual-Q4 tables | Particle Gibbs | ✅ |

`scripts/12_build_cpi_ppi_report.py` enforces this at build time:

```python
        assert_expected_sampler(run_set, model="hsa_full", expected="particle_gibbs", label=label)
```

so a cell that silently fell back to the alternating sampler fails the build rather than
entering a table. An earlier revision reached Particle Gibbs only through a monkeypatch in
`scripts/appendix_pg_full_runs.py` and merged its output into the report run-set; that script
is retired and the merge is gone.

**Superseded alternating blocks** (`func_nkpc_hsa_full_alternating_ffbs`, validation only) (`hsa_full/model.py:824, 851`), each an exact conditional:

1. `N̂_{0:T} | N̄_{0:T}, ·` — AR(2) FFBS, `_sample_ar2_states_ffbs_tv_theta` (line 171).
   Observation rows: `[1,0]` on `N^obs − N̄`, and `[θ_t, 0]` on the inflation residual, where
   `θ_t = θ₀ + γ N̄_t` is **known** because `N̄` is conditioned on.
2. `N̄_{0:T} | N̂_{0:T}, ·` — random-walk FFBS, `_sample_rw_states_ffbs_tv_theta_kappa` (line 315),
   with loading `h₂ = δ x_t /100 − γ N̂_t` (line 373):
```python
y2 = (pi_t[t] - pi_expect[t] - alpha * (pi_tm1[t] - pi_expect[t])
      - kappa0 * x_t[t] + theta0 * Nhat[t] - obs_offset[t])
h2 = delta * x_t[t] - gamma * Nhat[t]
```
Both verified algebraically correct. The scheme is a **valid** Gibbs kernel; it was replaced
because it mixes badly, not because it is wrong: `N^obs = N̄ + N̂ + ν` pins the sum almost
exactly (posterior corr(N̄₀, N̂₀) ≈ −0.999), so a two-block Gibbs moves the shared level with
autocorrelation ρ² ≈ 0.998 per sweep.

---

## F. One complete MCMC iteration

See `docs/estimation_flow.md`, which also documents the `opts["fixed"]` block-pinning mechanism
(§0 there) — present in CES and HSA steady only, inert in production, and used by the
conditional-marginal-likelihood reduced runs.

---

## G. Parameter-by-parameter conditional posterior map

Every Gaussian coefficient block uses the same conjugate form, implemented once per model as
`_sample_beta_gaussian`:

```
V₁ = (V₀⁻¹ + X'X/σ_η²)⁻¹
b₁ = V₁ (V₀⁻¹ b₀ + X'y*/σ_η²)
β | · ~ N(b₁, V₁)
```

**Actual code** (`hsa_steady/model.py:76-79`)
```python
V0_inv = np.diag(1.0 / prior_var)
Vn = inv(X.T @ X / sigma2 + V0_inv)
mn = Vn @ (X.T @ y / sigma2 + V0_inv @ prior_mean)
return _mvnrnd(mn, Vn, rng)
```
The prior is **diagonal** (`np.diag`), i.e. independent across coefficients — which is what makes
the Savage–Dickey density ratio valid for `δ = 0`.

| Parameter | Models | Block | `y*` | Regressor | Conditional |
|---|---|---|---|---|---|
| α | all | joint β | `y − λζ` | `a_t` | Normal |
| κ | ces, dynamic | joint β | `y − λζ` / `y − mean_e` | `x_t/100` | Normal |
| κ₀ | steady, const-θ, full | joint β | `y − λζ` | `x_t/100` | Normal |
| δ | steady, const-θ, full | joint β | `y − λζ` | `x_t N̄_t/100` | Normal |
| θ | dynamic, const-θ | joint β | as above | `−N̂_t` | Normal |
| θ₀ | full | joint β | `y − λζ` | `−N̂_t` | Normal |
| γ | full | joint β | `y − λζ` | `−N̂_t N̄_t` | Normal |
| λ_eζ | ces, steady, const-θ, full | **own block** | `e_base` | `ζ_t` | Normal |
| φ_1 | all | **own block** | see below | `x_{t−1}` | Normal |
| ρ₁, ρ₂ | all HSA | **own block** | `N̂_{1:}` | `N̂_{0:−1}, N̂_{−1:−2}` | Normal **truncated** to the stationary triangle |
| n | all HSA | **own block** | `ΔN̄_t` | 1 | Normal |
| σ_ζ², σ_η², σ_u², σ_ε², σ_N² | as applicable | own blocks | residuals | — | Inverse-gamma |
| Σ | dynamic only | own block | `(e,ζ,u,ε)` | — | Inverse-Wishart / restricted conjugate |

⚠️ **λ_eζ is NOT in the joint β block.** It is drawn afterwards, conditional on the just-drawn
β. Any description that puts `ζ_t` as a fifth column of `X` describes a different sampler.

**λ_eζ conditional** (`hsa_const_theta/model.py:234-242`)
```python
e_base = y - alpha * a_t - kappa_t_eff * x_t + theta * Nhat
post_var_lambda = 1.0 / (lambda_prec0 + float(np.sum(zeta**2)) / sigma_eta2)
post_mean_lambda = post_var_lambda * (
    pri["mu_lambda"] * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
)
```
This is the regression of the NKPC shock `e_t` on `ζ_t`.

**φ_1 conditional** — uses *two* sources of information (`hsa_steady/model.py:273-309`)
```python
prec = (1.0 / sigma_phi**2
        + float(np.sum(x_tm1**2)) / sigma_zeta2
        + (lambda_ez**2) * float(np.sum(x_tm1**2)) / sigma_eta2)
mean_num = (mu_phi / sigma_phi**2
            + float(np.dot(x_tm1, x_t)) / sigma_zeta2
            - lambda_ez * float(np.dot(x_tm1, y_tilde - lambda_ez * x_t)) / sigma_eta2)
```
The first two terms are the AR(1) `x_t = φ x_{t−1} + ζ_t`; the third is the extra information in
`e_t = λ(x_t − φ x_{t−1}) + η_t`. Report §3.4 describes exactly this.

**AR(2) block** (`hsa_steady/model.py:112-175`) — the `initial_lag` branch is what production uses:
```python
y = Nhat[1:]
second_lag = np.concatenate([[float(initial_lag)], Nhat[:-2]])
X = np.column_stack([Nhat[:-1], second_lag])
```
`initial_lag = states[0,1]`, the **sampled** `N̂_{−1}`, so the first AR(2) likelihood row is not
dropped. Stationarity is imposed by **rejection sampling**, up to `ar2_max_tries = 2000`:
```python
for attempt in range(1, max_tries + 1):
    draw = _mvnrnd(post_mean, post_cov, rng)
    r1, r2 = float(draw[0]), float(draw[1])
    if _is_stationary_ar2(r1, r2):
        ...
        return r1, r2
```
with the triangle `|ρ₂| < 1, ρ₁+ρ₂ < 1, ρ₂−ρ₁ < 1` (`:33`). On exhaustion it falls back, in order,
to the current value, the posterior mean, the prior mean, then `(0,0)` (`:168-175`), and records
`fallbacks` in run metadata.

**Trend drift `n`** (`hsa_const_theta/model.py:288-293`)
```python
dNbar = Nbar[1:] - Nbar[:-1]
post_var_n = 1.0 / (1.0 / pri["sigma_n"] ** 2 + dNbar.size / sigma_eps2)
post_mean_n = post_var_n * (pri["mu_n"] / pri["sigma_n"] ** 2 + float(np.sum(dNbar)) / sigma_eps2)
```
i.e. the mean of `ΔN̄_t = n + ε_t` with a Normal prior.

**Variances** — all inverse-gamma, `IG(a + n/2, b + SSR/2)`:

| Variance | Residual | Code |
|---|---|---|
| σ_ζ² | `ζ_t = x_t − φ x_{t−1}` | `const_theta:260` |
| σ_η² | `y − αa − κ_t x + θN̂ − λζ` | `const_theta:263` |
| σ_u² | `N̂_t − ρ₁N̂_{t−1} − ρ₂N̂_{t−2}` | `const_theta:282-285` |
| σ_ε² | `N̄_t − n − N̄_{t−1}` | `const_theta:294-298` |
| σ_N² | `N^obs_t − N̂_t − N̄_t`, **finite quarters only** | `const_theta:302-305` |

`finite_N_residuals` (`common/competition.py:12-21`) is what makes the annual-Q4 σ_N² update use
only the 31 Q4 residuals.

---

## H. Kalman / FFBS implementation map

Single production implementation: `sample_joint_competition_states_ffbs`,
`src/nkpc_hsa/gibbs/common/joint_ffbs.py:54-168`. Used by HSA steady (§E.2) and HSA const-theta
(§E.4). HSA dynamic has its own correlated-shock variant (§E.3).

### Forward recursion

| Step | Math | Code (`joint_ffbs.py`) |
|---|---|---|
| Predict | `m_{t\|t−1} = c + F m_{t−1}`; `P_{t\|t−1} = F P_{t−1} F' + Q` | `:134-135` |
| Innovation | `v_t = z_t − H_t m_{t\|t−1}`; `S_t = H_t P_{t\|t−1} H_t' + R_t` | `:150, 148` |
| Gain | `K_t = P_{t\|t−1} H_t' S_t⁻¹` | `:153` |
| Update mean | `m_t = m_{t\|t−1} + K_t v_t` | `:155` |
| Update cov (Joseph) | `P_t = (I−K_tH_t) P_{t\|t−1} (I−K_tH_t)' + K_t R_t K_t'` | `:154-155` |

```python
S = force_pd(H @ P_pred[t] @ H.T + R)
K = P_pred[t] @ H.T @ inv(S)
innov = y_obs - H @ m_pred[t]
m_filt[t] = m_pred[t] + K @ innov
KH = K @ H
P_filt[t] = force_pd((I3 - KH) @ P_pred[t] @ (I3 - KH).T + K @ R @ K.T)
```

Joseph form is used rather than `(I−KH)P` because it is symmetric and positive-semidefinite by
construction. `force_pd` (`:42-48`) additionally symmetrises and clips eigenvalues at 1e-10.

At `t = 0` there is **no** transition step: `m_pred[0] = m₀`, `P_pred[0] = P₀` (`:131-132`).

### Backward sampling (FFBS)

```
s_{T−1} ~ N(m_{T−1}, P_{T−1})
for t = T−2 … 0:
    A_t = P_t F' P_{t+1|t}⁻¹
    s_t | s_{t+1} ~ N( m_t + A_t (s_{t+1} − c − F m_t),  P_t − A_t P_{t+1|t} A_t' )
```

**Actual code** (`joint_ffbs.py:158-166`)
```python
states[-1] = _mvnrnd(m_filt[-1], P_filt[-1], rng)
for t in range(T - 2, -1, -1):
    Ptp1 = force_pd(P_pred[t + 1])
    A = P_filt[t] @ F.T @ inv(Ptp1)
    mean_s = m_filt[t] + A @ (states[t + 1] - c - F @ m_filt[t])
    cov_s = force_pd(P_filt[t] - A @ Ptp1 @ A.T)
    states[t] = _mvnrnd(mean_s, cov_s, rng)
```
Note `− c −` in the mean: the drift must be removed before applying the smoothing gain. This
produces a draw from the **joint** smoothing posterior `p(s_{0:T−1} | z_{1:T}, θ)`, not a sequence
of marginal draws.

### Linear-Gaussianity, stated precisely

| Model | `H[1,·]` | Linear in `s_t`? |
|---|---|---|
| HSA steady | `[0, 0, δx_t/100]` | ✅ coefficient depends on data only |
| HSA const-theta | `[−θ₀, 0, δx_t/100]` | ✅ coefficients depend on parameters and data only |
| HSA full (γ≠0) | `[−(θ₀+γN̄_t), 0, δx_t/100]` | ❌ the loading on `N̂_t` **contains a state** |

### Validation

`tests/test_joint_ffbs.py` pins:
1. delegating HSA steady to this routine reproduces its previous in-line implementation
   **bit for bit** (`np.array_equal`), with and without missing observations;
2. FFBS draws match the analytic smoothing mean/sd obtained by assembling the joint Gaussian
   precision matrix directly — independent code — for both `θ₀ = 0` and `θ₀ ≠ 0`;
3. `θ₀ = 0` collapses the const-theta state block onto HSA steady's;
4. with every firm-count row missing, the inflation row alone still moves `N̄`.

---

## I. Particle-Gibbs implementation map

`sample_states_particle_gibbs`, `src/nkpc_hsa/gibbs/hsa_full_pg/model.py:102-234`.
`P = 512` in production, from `configs/models.yaml → defaults.n_particles`, threaded through
`run_model(n_particles=…)` → `opts["n_particles"]` (`gibbs/hsa_full_pg/model.py:396`).

### Reference path
The state path from the previous MCMC iteration, pinned into particle slot 0:
```python
pg = sample_states_particle_gibbs(..., Nbar_ref=Nbar, Nhat_ref=Nhat, Nhat_ref_lag=Nhat_initial_lag, ...)
```

### t = 0 (`:176-190`)
```python
Nhat0[0] = Nhat_ref[0]
Nlag0[0] = Nhat_ref_lag
Nbar0[0] = Nbar_ref[0]
if P > 1:
    Nhat0[1:] = m0_Nhat + np.sqrt(P0_Nhat) * rng.standard_normal(P - 1)
    ...
```
Slot 0 is the reference; slots 1…P−1 are drawn from the initial prior `N(m₀, P₀)`.

### t = 1 … T−1 (`:193-213`)
```python
a = rng.choice(P, size=P, p=W)   # conditional multinomial resampling
a[0] = 0                         # reference keeps its lineage (slot 0)
parent_Nhat = Nhat_store[t - 1, a]
parent_Nlag = Nlag_store[t - 1, a]
parent_Nbar = Nbar_store[t - 1, a]

new_Nhat = rho1 * parent_Nhat + rho2 * parent_Nlag + su * rng.standard_normal(P)
new_Nlag = parent_Nhat
new_Nbar = n_drift + parent_Nbar + se * rng.standard_normal(P)

new_Nhat[0] = Nhat_ref[t]        # pin the reference trajectory into slot 0
new_Nlag[0] = Nhat_ref[t - 1]
new_Nbar[0] = Nbar_ref[t]
```
Propagation is the **bootstrap proposal**: particles are drawn from the state transition itself
(§D), so the transition density cancels in the weight. `new_Nlag = parent_Nhat` preserves the
AR(2) genealogy — the lag travels with its own ancestor, not with the resampled index.

### Weights (`:51-87`)
```python
mu = (alpha * a_t + kappa0_eff * x_t + delta_eff * x_t * Nbar
      - theta0 * Nhat - gamma * Nbar * Nhat + lambda_ez * zeta_t)
ll = -0.5 * (y_t - mu) ** 2 / sigma_eta2
if np.isfinite(N_obs_t):
    ll = ll - 0.5 * (N_obs_t - Nbar - Nhat) ** 2 / sigma_N2
```
i.e. `w_t^i ∝ p(y_t | s_t^i, θ) · p(N^obs_t | s_t^i, θ)` — **both** observation equations. The
`−½log(2πσ²)` constants are common across particles at a given `t` and cancel under
normalisation, so they are omitted. Missing `N^obs` drops the second term only.

### Normalisation and ESS (`:90-96`)
```python
m = float(np.max(logw))
w = np.exp(logw - m)
s = float(np.sum(w))
W = w / s
ess = 1.0 / float(np.sum(W * W))
```
Log-sum-exp with the max subtracted, so no overflow.

### Terminal draw and ancestor tracing (`:216-224`)
```python
b[T - 1] = int(rng.choice(P, p=W))
for t in range(T - 2, -1, -1):
    b[t] = ancestors[t + 1, b[t + 1]]
Nhat_new = Nhat_store[idx, b]
Nbar_new = Nbar_store[idx, b]
```
Tracing ancestors backward returns a **single coherent trajectory**, not a per-time-point
concatenation of independently-chosen particles.

### Why this is conditional SMC, not an ordinary particle filter

An ordinary bootstrap PF is *not* a valid MCMC kernel: its output is not distributed as the exact
smoothing posterior for finite `P`, and using it in a Gibbs sweep would leave a different
stationary distribution. Conditional SMC (Andrieu, Doucet & Holenstein 2010) fixes this by
guaranteeing that the previous iteration's path **survives the whole sweep**: slot 0 is never
resampled away (`a[0] = 0`) and is re-inserted at every `t`. The terminal draw over `W_T` then
either re-selects the reference or moves to a genuine alternative, and the resulting kernel leaves
`p(s_{0:T} | θ, y, N^obs)` invariant **for any number of particles**. `P` affects efficiency, not
correctness.

### Diagnostics recorded
`ess_mean`, `ess_min`, `moved_frac` (fraction of periods the returned path leaves the reference)
are stored per draw under `pg_diagnostics` (`:609-614`). ⚠️ They are **not** written into
`posterior.nc` by `_extract_draws_from_result` (`wrappers.py:302-321` only keeps `state_draws` and
`{"draws": …}` entries), so they survive only in the appendix JSON.

---

## J. Priors

**Config** `configs/priors_{baseline,weak,tight}.yaml` — physical units.
**Mapper** `models/common.prior_specs_to_internal` (`models/common.py:154-197`) — the single
authoritative implementation.
**Resolver** each model's `_common_priors`, which supplies defaults for anything absent.

| Parameter | Baseline | Weak | Tight | Support | Internal units | Config→sampler |
|---|---|---|---|---|---|---|
| α | N(0.5, 0.2²) | N(0.5, 0.5²) | N(0.5, 0.1²) | ℝ | same | `mu_alpha, sigma_alpha` |
| κ | N(0.1, 0.2²) | N(0.1, 0.5²) | N(0.1, 0.1²) | ℝ | **×100** | `mu_kappa, sigma_kappa` |
| κ₀ | N(0.1, 0.2²) | N(0.1, 0.5²) | N(0.1, 0.1²) | ℝ | **×100** | `mu_kappa_0, sigma_kappa_0` |
| δ | **N(0, 0.02²)** | N(0, 0.1²) | N(0, 0.01²) | ℝ | **×100** → N(0, 2²) | `mu_delta, sigma_delta` |
| θ, θ₀ | N(0.1, 0.2²) | N(0, 0.5²) | N(0.1, 0.1²) | ℝ | same | `mu_theta, sigma_theta` |
| γ | N(0, 0.02²) | N(0, 0.1²) | N(0, 0.01²) | ℝ | **same** (unscaled regressor) | `mu_gamma, sigma_gamma` |
| φ_1 | N(0.7, 0.2²) | N(0.7, 0.5²) | N(0.7, 0.1²) | ℝ | same | `mu_phi_1, sigma_phi_1` |
| ρ₁ | N(0.5, 0.2²) | N(0.5, 0.5²) | N(0.5, 0.1²) | **stationary triangle** | same | `mu_rho1, sigma_rho1` |
| ρ₂ | N(−0.5, 0.2²) | N(−0.5, 0.5²) | N(−0.5, 0.1²) | **stationary triangle** | same | `mu_rho2, sigma_rho2` |
| n | N(0, 0.1²) | N(0, 0.25²) | N(0, 0.05²) | ℝ | same | `mu_n, sigma_n` |
| λ_eζ | **N(0, 0.5²)** | *same* | *same* | ℝ | same | ⚠️ **hard-coded default**, see below |
| σ_η² | IG(2, 2) | IG(1, 1) | IG(4, 4) | ℝ₊ | same | `a_e, b_e` |
| σ_ζ² | IG(.001, .001) | IG(.001,.001) | IG(.01, .01) | ℝ₊ | same | `a_z, b_z` |
| σ_u² | IG(2, 0.02) | IG(1, 0.02) | IG(4, 0.06) | ℝ₊ | same | `a_u, b_u` |
| σ_ε² | IG(2, 0.01) | IG(1, 0.01) | IG(4, 0.03) | ℝ₊ | same | `a_eps, b_eps` |
| σ_N² | IG(2, 0.01) | IG(1, 0.01) | IG(4, 0.03) | ℝ₊ | same | `a_N, b_N` |
| Σ (dynamic) | IW(8, diag(3,3,.06,.03)) | IW(6, ·) | IW(12, ·) | PD | same | `nu_Sigma, S_Sigma` |
| s₀ | N(0, 10·I₃) | *same* | *same* | ℝ³ | same | ⚠️ **hard-coded default** |
| *P* (particles) | 512 | *same* | *same* | — | — | `configs/models.yaml → defaults.n_particles`; not a prior, listed here as the one remaining sampler tuning constant |

**KAPPA_SCALE conversion, explicitly** (`models/common.py:183-190`):
```python
for key, target in [("kappa", …), ("kappa_0", …), ("delta", …)]:
    p = pair(key)
    if p is not None:
        out[target[0]], out[target[1]] = kappa_prior_physical_to_internal(*p)
```
with `kappa_prior_physical_to_internal(mean, sd) = (mean*100, sd*100)` (`:86-87`). So the physical
`δ ~ N(0, 0.02²)` becomes the internal `δ_int ~ N(0, 2²)`, and since `δ_int = 100·δ` and the
regressor is `x·N̄/100`, the two reparameterisations are exactly equivalent. `theta`, `theta_0`,
`gamma`, `n` etc. go through the unscaled `pair()` loop at `:167-181`.

⚠️ **Two priors are hard-coded, not configured.** `λ_eζ ~ N(0, 0.5²)` and `s₀ ~ N(0, 10I₃)` appear
in **no** `priors_*.yaml`; they come from `_common_priors` defaults (`hsa_steady/model.py:228-229,
266-268`). Consequence: they are **identical across baseline/weak/tight**, so the prior-sensitivity
exercise does not vary them. Since the `s₀` prior is one of the few things pinning the N̄/N̂ level
split, the reported prior sweep does not test sensitivity of the trend level or of κ₀.

**Variance-scale note.** `b_u`/`b_eps`/`b_N` are in *squared ten-log-point* units and must stay
near the 0.01 decade; `tests/test_prior_wiring.py::test_state_variance_scales_stay_in_the_transformed_n_decade`
asserts `< 0.5`.

**Truncation and normalising constants.** The AR(2) prior is truncated by rejection in the sampler.
No normalising constant is needed for *sampling*, but it is needed for any density evaluation; the
marginal-likelihood module computes it by fixed-seed Monte Carlo
(`gibbs/conditional_ml.py:135-154`).

**Coefficient constraints.** `configs/models.yaml:14-18` sets `enabled: false`, so
`draw_with_constraints` is a pass-through in every production run. When enabled, kappa-like bounds
are converted to internal units by `coefficient_constraints_to_internal` (`models/common.py:110-151`).

---

## K. Diagnostics

Implemented in `scripts/12_build_cpi_ppi_report.py:301-400`. Thresholds `RHAT_LIMIT = 1.01`,
`ESS_LIMIT = 400.0` (`:33-34`).

### Three groups (`:53-74`)

```python
SCALAR_PARAMETERS = [
    "alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma",
    "rho_1", "rho_2", "n", "phi_1", "lambda_ez",
    "sigma_e", "sigma_eta", "sigma_zeta", "sigma_u", "sigma_eps", "sigma_N",
]   # one entry per line in the source; wrapped here only for width
STATE_PATH_PARAMETERS = ["Nbar", "Nhat"]
DERIVED_PATH_PARAMETERS = ["kappa_t", "theta_t"]
```

`_group_diagnostics` (`:301-348`) takes max `R̂` and min bulk ESS over each group, **skipping**
variables that are constant across draws (`np.nanstd(values) <= 0`) because their `R̂` is
undefined and would poison the max.

### Flag semantics (`:351-400`)

| Flag | Meaning |
|---|---|
| `converged` | **scalar** group only — this is what the `†` mark means |
| `state_converged` | `Nbar`, `Nhat` paths |
| `derived_converged` | `kappa_t`, `theta_t` paths |
| `joint_converged` | all present groups |

```python
def _conv_status(diagnostics: dict[str, object], *, japanese: bool = False) -> str:
    watch = "要注意" if japanese else "watch"
    if not bool(diagnostics["converged"]):
        return watch
    if diagnostics["has_states"] and not bool(diagnostics["joint_converged"]):
        return "OK (coef)"
    return "OK"
```

So a table cell reads `OK` (all three groups), `OK (coef)` (coefficients pass, state paths do not),
or `watch`/`†` (coefficients fail).

⚠️ **Historical note.** Before August 2026 `_diagnostics` inspected only nine coefficients —
`n`, `φ_1`, `λ_eζ` and every variance were excluded, as were all paths. `n` is the worst-mixing
scalar in essentially every HSA cell, so the earlier flags were systematically optimistic.

### Where the flags surface
`build_model_table`, `build_output_gap_model_tables`, `build_prior_table`, `build_diagnostics_table`,
`build_run_manifest`, `build_group_convergence_diagnostics` (all in script 12), and
`scripts/make_headline_results_table.py`.

---

## L. Derived quantities and report outputs

### κ_t — **not separately estimated**

```
κ_t^(m) = κ₀^(m) + δ^(m) · N̄_t^(m)      for each retained draw m
```

**Actual code** (`hsa_const_theta/model.py:328, 353`)
```python
kappa_t = kappa0 + delta * Nbar
...
kappa_t_draws[store_idx] = kappa_t / KAPPA_SCALE
```
Computed from the draw's **own** κ₀, δ and N̄ path — so the stored `kappa_t` carries the joint
posterior uncertainty of coefficients *and* states, not a plug-in.

⚠️ **Ordering subtlety.** In `hsa_steady` and `hsa_const_theta`, `kappa_t` is recomputed
**after** the state draw (`hsa_steady:843`, `const_theta:328`), so the stored `κ_t` pairs the
iteration's coefficients with the iteration's *new* `N̄`. In `hsa_full` the same is true
(`hsa_full:872-874`). Consistent across models.

### θ_t

`theta_t = theta_0 + gamma * Nbar` in `hsa_full` (`:874`); in `hsa_const_theta` it is stored as a
constant path `np.full(T, theta)` (`:355`) so the schema matches.

### Report summaries

| Statistic | Code | Definition |
|---|---|---|
| posterior mean | `_summary` (`script12:244-254`) | `np.mean` over pooled chain×draw |
| 95 % CI | `_summary` | `np.quantile(·, 0.025)` and `0.975` |
| κ_t start / end | `_path_summary` (`:256-267`) | `nanmean(paths[:, 0])`, `nanmean(paths[:, -1])` |
| BF₁₀(δ) | `_bf10` (`:269-285`) | Savage–Dickey: prior density at 0 ÷ Gaussian-KDE posterior density at 0 |

```python
posterior_at_zero = float(gaussian_kde(values)([0.0])[0])
prior_at_zero = float(norm.pdf(0.0, loc=prior_mean, scale=prior_sd))
bf01 = posterior_at_zero / max(prior_at_zero, 1e-300)
return float(1.0 / max(bf01, 1e-300))
```
The prior comes from the run's own `priors.json`, injected as `idata.attrs["run_priors"]` by
`_load_runs` (`:147-150`), so prior and posterior always match the run.

⚠️ `gaussian_kde` uses Scott's rule with no bandwidth argument. For the main cell the posterior of
δ has **no draws within 3 s.d. of zero**, so this is a tail extrapolation.

### Economic magnitude
The report's "start-to-end response differs by about 0.126 point" (§10) is a **hard-coded literal**
in `report/nkpc_hsa_report.tex`, not a macro. The macro-driven components are
`\CoreUnempKappaStart` (+0.120) and `\CoreUnempKappaEnd` (−0.005), whose difference is 0.125.

---

## M. Code-to-equation crosswalk

See `docs/code_equation_crosswalk.md`.

---

## N. Identification map

| Parameter | Appears in | Identifying variation | Posterior dependence / confounding |
|---|---|---|---|
| **α** | `α a_t` | covariance of `y_t` with `π_{t−1} − E_tπ_{t+1}` | Inflated by construction: `π` is a 4-quarter change sampled quarterly, so `π_t` and `π_{t−1}` overlap in 3 of 4 quarters. α ≈ 0.79 is partly a measurement artifact. |
| **κ / κ₀** | `κ x_t` | covariance of the inflation residual with `x_t` | **corr(mean N̄, κ₀) ≈ −0.55.** κ₀ is the slope at `N̄ = 0`, and the N̄ *level* is pinned largely by the `s₀ ~ N(0,10I)` prior, not by data. So κ₀ is identified only up to that normalisation. |
| **δ** | `δ x_t N̄_t` | covariance of the inflation residual with the **interaction** `x_t·N̄_t`, given `a_t` and `x_t` | **corr(mean N̄, δ) ≈ −0.13** — a level shift in N̄ moves κ₀, not δ. This is why δ is the robust parameter. But `N̄_t` is near-collinear with a linear trend, so δ cannot be separated from anything else trending over 1982–2012 (report §8 says this). |
| **θ / θ₀** | `−θ N̂_t` | covariance of the inflation residual with the *cycle* `N̂_t` | `N̂` is itself barely identified: corr(N̄₀, N̂₀) ≈ −0.999, and its dynamics change completely between observation schemes (ρ₁ ≈ 1.80 under PCHIP vs ≈ 0.20 under annual-Q4). Anything loading on `N̂` inherits that. Theory implies **θ > 0**; the report states the predicted sign for δ but never for θ. |
| **γ** | `−γ N̂_t N̄_t` | covariance with `N̂_t·N̄_t` given `N̂_t` | corr(`N̂`, `N̂N̄`) ≈ 0.56–0.83, condition numbers 2.6–5.6. Mathematically identified; weakly so empirically, and inheriting `N̂`'s indeterminacy. |
| **λ_eζ** | `λ ζ_t` | covariance of the NKPC residual with the activity innovation | Prior N(0,0.5²) is hard-coded and never varied. |
| **φ_1** | `x_t = φ x_{t−1} + ζ_t` | autocovariance of `x_t`, **plus** the `e = λζ + η` channel | Posterior ≈ 0.95 with CI upper > 1: near unit root. |
| **ρ₁, ρ₂** | AR(2) of `N̂` | autocovariance of the sampled `N̂` path | **Almost entirely an artifact of the observation scheme**, not economics: PCHIP interpolation manufactures quarterly smoothness (ρ₁ ≈ 1.80); annual-Q4 with the `ρ₂ ~ N(−0.5,0.2²)` prior manufactures a ~4-quarter oscillation (ρ₁ ≈ 0.20, ρ₂ ≈ −0.89). |
| **n** | `N̄_t = n + N̄_{t−1} + ε_t` | mean of `ΔN̄_t` | The worst-mixing scalar in every HSA cell — this is the latent level ridge appearing in the parameter block. |
| **σ_η²** | inflation residual | SSR of `η_t` | Understated relative to a serial-correlation-robust estimate (Ljung–Box(8) on the fitted residuals gives p ≈ 0.0002), but the posterior s.d. of δ is inflated by state uncertainty by roughly the offsetting amount. |
| **σ_u², σ_ε², σ_N²** | state/measurement | SSRs of the sampled paths | `σ_N` ≈ 0.023 under PCHIP: the interpolated series is fitted almost exactly, which is what pins `N̄+N̂` while leaving the split free. |
| **Σ** (dynamic) | shock covariance | cross-products of `(e,ζ,u,ε)` residuals | Only the (e,ζ) off-diagonal is free under the default restriction. |

**The central identification fact, and that it is design-dependent.**
`N^obs = N̄ + N̂ + ν` constrains the *sum* but not the split: adding `c` to `N̄` and subtracting it
from `N̂` leaves the firm-count equation, the trend equation, and — via `κ₀ → κ₀ − δc` — the
inflation equation all exactly unchanged. The only resistance is the AR(2)'s mean-reversion term
`1 − ρ₁ − ρ₂` (it has no intercept), plus the weak `s₀` prior.

Under **PCHIP** the interpolated series is so smooth that σ_N collapses to 0.023 and the AR(2) is
pushed to a near-unit root, `ρ₁+ρ₂ = 0.986`, so the anchor `1 − ρ₁ − ρ₂ = 0.014` effectively
vanishes and corr(N̄₀, N̂₀) = **−0.9996**. Under **annual-Q4** the same quantity is **1.69** and the
correlation is **+0.13**. Across all 122 estimated HSA cells the two correlate at **+0.92**, which
identifies the mechanism rather than merely describing it.

Consequences: `κ₀` is identified only up to the `N̄` location normalisation, and under PCHIP that
normalisation comes substantially from the `s₀` prior (corr(N̄ level, κ₀) = −0.55, versus −0.32
under annual-Q4). `δ` is unaffected either way, because a level shift in `N̄` is absorbed by `κ₀`
and leaves the interaction `x_t·N̄_t` unchanged. Under PCHIP the posterior mean `N̂₀ ≈ −3.3`
(−33 log points) is not an economically interpretable business cycle; it is the ridge.

---

## O. File / function dependency map

```
configs/
  models.yaml ────────────► config.load_model_config / configured_data_specs
  priors_*.yaml ──────────► models.common.prior_specs_to_internal ──► every sampler
  periods.yaml ───────────► inference.period_robustness (subsample windows)
  # paths.yaml and periods_tnic.yaml are gone: nothing read them, and paths.yaml's
  # entries had gone stale (it still pointed at reports/main.tex and results/tables/).
  # Output locations are module-level constants in the scripts that write them.

src/nkpc_hsa/
  paths.py, config.py
  data/
    func_data_build.py     raw → quarterly panel (legacy builder, still production)
    build.py               + HP output gap, + labor-share gap  → model_ready.csv
    transforms.py          transform_competition_series  (log100_centered10)
    competition.py         PCHIP vs annual-Q4 observation vectors
    load.py
  models/
    common.py              KAPPA_SCALE, prior_specs_to_internal,
                           coefficient_constraints_to_internal, sample_beta_gaussian
    ces.py / hsa_*.py      thin facades over gibbs/*
  gibbs/
    common/
      joint_ffbs.py        ★ shared exact joint Kalman/FFBS  (steady, const-theta)
      competition.py       finite_N_residuals, initial_competition_path
      constraints.py       draw_with_constraints
      state_space.py       re-exports two UNUSED hsa_full helpers
      marginal_likelihood.py
    ces/model.py           func_nkpc_ces
    hsa_steady/model.py    func_nkpc_hsa_decomp_tv_kappa_kalman  → joint_ffbs
    hsa_dynamic/model.py   func_nkpc_hsa_decomp_joint_fullSigma  (own correlated FFBS)
    hsa_const_theta/model.py  func_nkpc_hsa_const_theta          → joint_ffbs
    hsa_full/model.py      func_nkpc_hsa_full  (alternating FFBS)
                           + func_nkpc_hsa_full_static_theta  [legacy const-theta]
    hsa_full_pg/model.py   func_nkpc_hsa_full_pg  (Particle Gibbs)
                           + sample_states_joint_ffbs_gamma0  [validation only]
    conditional_ml.py      corrected conditional marginal likelihood
    gibbs_marginal_likelihood.py  [older Chib code; outputs quarantined]
    gibbs_wrappers.py      [DEPRECATED]
  inference/
    wrappers.py            ★ run_model — dispatch, data prep, save
                             (run_ces/run_hsa_* are one-line aliases, :692-709)
    prior_sensitivity.py   calls run_model (:38)  — re-estimates under each prior set
    period_robustness.py   calls run_model (:124) — subsample re-estimation
    diagnostics.py, model_comparison.py, identification.py, prior_robustness.py
  reporting/
    cpi_ppi_spec.py        MODEL_ORDER, INFLATION_SPECS, report_run_keys()
    tables.py, figures.py, estimation_results.py, data_model_report.py
    # latex.py is gone. It generated the superseded main.tex report and defaulted
    # to paper/main.tex and results/report/, none of which exist; the fragments it
    # \input-ed were deleted with scripts/07 and 08. Everything the surviving
    # report needs is written by scripts/12 and the make_* scripts instead.

scripts/
  # --- pipeline ---
  01_build_data.py … 06_model_comparison.py    data, estimation, robustness, comparison
  09_identification_diagnostics.py
  11_additional_report_evidence.py   -> results/evidence/tables/report_additions/; the report quotes
                                     these numbers in prose rather than \input-ing them
  # 07, 08 and 10 are gone: they built the superseded report (main.tex) and the HTML
  # edition. The deliverable is report/nkpc_hsa_report.{tex,pdf} only.
  12_build_cpi_ppi_report.py  ★ load_report_runs, diagnostics, all table builders
  13_estimate_cpi_ppi_report.py    estimation driver (design default from models.yaml)
  make_headline_results_table.py   headline_results / ppi_results / model_comparison_unemp,
                                   once per observation design
  make_fit_comparison_table.py     tab:fit-comparison (plug-in next to LPD1/WAIC/LOO)
  make_data_series_figure.py       fig:data
  # --- estimation re-runs (each writes new run dirs; nothing is overwritten) ---
  rerun_hsa_full_particle_gibbs.py   all 30 hsa_full cells, Particle Gibbs, both designs
  rerun_const_theta_joint_ffbs.py    all 30 const-theta cells, exact joint FFBS
  rerun_hsa_steady_no_inertia.py     all 9 annual-Q4 cells under the alpha == 0 restriction;
                                     written with constraint_spec = "alpha_zero" so
                                     load_report_runs cannot select them
  # --- model comparison / evidence ---
  predictive_comparison.py         prequential LPD1 + WAIC + PSIS-LOO + the plug-in score,
                                   both designs; CES shared across designs
  chib_marginal_likelihood.py      corrected conditional ML driver -> gibbs/conditional_ml.py
  appendix_particle_gibbs_hsa_full.py   PG validation / pilot / production summaries
```

Two groups of scripts that earlier revisions listed here are no longer under `scripts/`.

**Deleted.** `build_english_tables.py`, `appendix_pg_full_runs.py` and
`appendix_pg_full_tables.py` had already been reduced to no-op stubs — the tables are
English at source, and Particle Gibbs is the dispatched sampler rather than a monkeypatch —
so the stubs themselves were removed.

**Moved to `archive/legacy_scripts/`** (git-ignored; nothing in `src/` or `scripts/` imports
them). These drove the review of this revision rather than any artifact in the paper; no
output of theirs is cited by `report/nkpc_hsa_report.tex`, which is what distinguishes them
from the re-run drivers above. Their outputs moved with them, to
`archive/legacy_results/`:

| Script | Output | Purpose |
|---|---|---|
| `const_theta_joint_ffbs_pilot.py` | `const_theta_pilot/` | old-vs-new const-theta gate, run before the re-run |
| `prior_decomposition_rho_delta.py` | `prior_decomposition/` | δ-prior × AR(2)-prior factorial |
| `fix_attribution.py` | — | attributes each changed number to T1/T2/T3 |
| `report_artifact_diff.py` | `_review_baseline/` | before/after artifact diff |

★ = the two files a reader should open first.

---

## P. How to read this specification

**Which design is a number from?** Macros prefixed `Annual` and tables under
`annual_q4/` are the **main** mixed-frequency results (paper §4–§6, §9–§10, Appendices A
and C). Unprefixed macros and base-directory tables are the **interpolated comparison**
(paper §7, Appendix B). The prefix is historical: it predates the decision to make mixed
frequency primary, and was kept rather than renamed so that no existing macro silently
changes meaning.

**To trace a reported number backwards** (e.g. `\AnnualCoreUnempDelta`):
1. `docs/code_equation_crosswalk.md` §"Report traceability" → posterior variable and run key.
2. §L here → how the summary statistic is computed from draws.
3. §G → the conditional posterior that produced those draws.
4. §E → the equation the conditional comes from.
5. `docs/data_dictionary.md` → the observed series in that equation.

**To trace a code line forwards** (e.g. `h_nbar=(delta / KAPPA_SCALE) * x_t`):
1. `docs/code_equation_crosswalk.md` → the equation term (`δ N̄_t x_t`).
2. §D → where it sits in `H_t`.
3. §C.5 → why the `/100` is there.
4. §N → what identifies `δ`.
5. §L → that `δ` also feeds the derived `κ_t` path.

**To reproduce one MCMC iteration by hand**: `docs/estimation_flow.md`, which gives the exact
execution order per model with the conditional posterior and code for every step.

**Conventions used throughout**
- Line numbers refer to the post-August-2026 codebase.
- Run directories: `_default_run_dir` always appends the observation design, since a
  conditional suffix would have inverted meaning when the project default changed to
  `annual_q4`. Directories created earlier, and any created with an explicit `run_dir`,
  may omit it; run metadata is authoritative for selection either way.
- ⚠️ marks a discrepancy, hard-coded value, or fragile convention — not necessarily a defect.
- **UNVERIFIED** marks something not recoverable from code or config.
- "Production" means reachable from `run_model`; anything else is labelled per §A.

---

## Q. Final review — verification record

### Q.1 What was verified, and how

Every line-number and code-snippet claim in these four documents was checked programmatically
against the working tree: **52 line-range claims** (all in range, no unresolved paths) and
**22 exact snippet claims** (all matching within ±3 lines). Substantive claims were re-derived
from live code and data:

| Claim | Result |
|---|---|
| `KAPPA_SCALE == 100` in all five model modules | ✅ |
| physical δ prior 0.02 → internal 2.0 (×100) | ✅ |
| γ prior 0.02 → internal 0.02 (**not** rescaled) | ✅ |
| `n` prior 0.1 → internal 0.1 (not rescaled) | ✅ |
| `m₀ = (0,0,0)`, `P₀ = 10·I₃` from resolver defaults | ✅ |
| `λ_eζ ~ N(0, 0.5²)` hard-coded, absent from all `priors_*.yaml` | ✅ |
| `s₀` hyperparameters absent from all `priors_*.yaml` | ✅ |
| stationary triangle accepts (1.80, −0.81), rejects (1.5, 0.0) | ✅ |
| transformed `N` has mean 0, range [−2.69, +2.16], T = 124 | ✅ |
| annual-Q4 produces exactly **31** finite observations | ✅ |
| `n` and all variances are in `SCALAR_PARAMETERS` | ✅ |
| `STATE_PATH_PARAMETERS == ["Nbar","Nhat"]`, `DERIVED == ["kappa_t","theta_t"]` | ✅ |
| thresholds `R̂ ≤ 1.01`, bulk ESS ≥ 400 | ✅ |
| `_conv_status` returns `OK` / `OK (coef)` / `watch` as documented | ✅ |
| `hsa_const_theta` stores `theta` and `theta_t`, never `gamma` | ✅ |
| new const-theta runs declare `state_sampler = "joint_ffbs"` in metadata | ✅ |
| `hsa_steady`'s saved schema excludes `Nhat_lag` (opt-in only) | ✅ |
| CES's `sigma_e2` is square-rooted to `sigma_e` on load | ✅ |

### Q.2 Per-model confirmation

| | CES | HSA steady | HSA dynamic | HSA const-theta | HSA full |
|---|---|---|---|---|---|
| estimated equation | ✅ §E.1 | ✅ §E.2 | ✅ §E.3 | ✅ §E.4 | ✅ §E.5 |
| state vector | n/a | ✅ | ✅ | ✅ | ✅ |
| transition `F, c, Q` | n/a | ✅ | ✅ (+`cov(u,ε)`) | ✅ | ✅ |
| observation `H_t, R_t` | n/a | ✅ | ✅ (+cross-cov `C`) | ✅ | ✅ (per block) |
| conditional blocks | ✅ 4 | ✅ 9 | ✅ 7 | ✅ 9 | ✅ 8 |
| prior source | ✅ | ✅ | ✅ (+Σ) | ✅ | ✅ |
| KAPPA_SCALE handling | ✅ | ✅ | ✅ | ✅ | ✅ |
| sampler identified | conjugate | joint FFBS | joint FFBS | joint FFBS | alt. FFBS / PG |
| saved posterior variables | ✅ | ✅ | ✅ | ✅ | ✅ |

### Q.3 Search sweep for missed branches

`grep` over `src/` and `scripts/` for every `func_nkpc*` definition, every `run_*` entry point,
and every monkeypatch. Findings folded into §A's classification table:

- **`src/nkpc_hsa/gibbs/gibbs_ces.py`** — a second CES sampler that returns its own
  `"deprecated"` key (`:206-209`) and is imported by nothing. Not production.
- **`gibbs/gibbs_utils.py`, `gibbs/gibbs_notebook_utils.py`** — reached only through the
  re-export shims `gibbs/common/math.py` and `gibbs/common/notebook.py`.
- **`inference/prior_sensitivity.py:38`, `inference/period_robustness.py:124`** — two further
  `run_model` callers; they change only priors / sample window, not the model or sampler.
- **No monkeypatch remains.** `scripts/appendix_pg_full_runs.py` used to rebind
  `func_nkpc_hsa_full` to the Particle-Gibbs implementation for one process; Particle Gibbs is
  now the dispatched production sampler and the script is a retired no-op.

No other production path exists.

### Q.4 Code / report discrepancies

Both sides shown; the documentation follows the code.

| # | Code says | Report says | Where |
|---|---|---|---|
| 1 | `hsa_full` is Particle Gibbs for **both** designs | §3.1/§3.2 say so | ✅ **resolved** |
| 1b | `configs/models.yaml`, `DEFAULT_COMPETITION_MEASUREMENT` and `scripts/13_estimate_cpi_ppi_report.py`'s CLI default all resolve to `annual_q4` | The report's main results are `annual_q4` | ✅ **resolved** — pinned by `tests/test_observation_design_default.py`. Metadata *readers* deliberately still fall back to `quarterly_interpolated` for runs predating the field. |
| 2 | The `(α, 1−α)` weights on `π_{t−1}` and `E_tπ_{t+1}` are forced to sum to one | §1.4 says "we impose no cross-coefficient restrictions from the theory" and lists only "we add backward-looking inertia (α)" | ⚠️ **open** — §C.1 here |
| 3 | Theory in §1.1 implies `θ > 0` for the entry channel | The report states the predicted sign for δ but never for θ | ⚠️ **open** — §N here |
| 4 | headline CPI and PPI use `pct_yoy`; core CPI uses `log_yoy` | not mentioned | ⚠️ **open** — `data_dictionary.md` |
| 5 | `λ_eζ` and `s₀` priors are hard-coded and identical across baseline/weak/tight | Table 3 lists `λ_eζ ~ N(0,0.5²)` among the "baseline priors" without saying it is not varied | ⚠️ **open** — §J here |
| 6 | The "0.126" economic magnitude is a hard-coded literal; the macros give 0.120 − (−0.005) = 0.125 | §10 states "about 0.126 point" | ⚠️ **cosmetic** — §L here |
| 7 | `π_t` is a four-quarter change sampled quarterly, so `π_t` and `π_{t−1}` overlap in three quarters; the likelihood assumes i.i.d. `η_t` | not mentioned | ⚠️ **open** — `data_dictionary.md`, §N |

None of these was silently reconciled.

### Q.5 UNVERIFIED items

| Item | Why |
|---|---|
| Horizon of the `Epi` series | The Cleveland Fed CSV column is ` Epi`; no maturity is recorded in code, config or report. The equation calls it `E_tπ_{t+1}`. |
| Upstream BN decompositions (`output_gap_BN`, `markup_BN_inv`, `N_Gustavo` trend/cycle) | Computed outside this repository; only their outputs are read. |
| Provenance of `BN_N_Gustavo_26.csv` | The report describes it as the inverse HHI of U.S. listed firms; the construction is not in this repository. |
| Whether `nekarda_ramey_markups.xlsx`'s `mu_bus` is ever intended for use | Loaded as `markup` but referenced by no configured spec. |

### Q.5b Test inventory

The suite is 84 tests. The ones that pin decisions made in this revision:

| Test file | Pins |
|---|---|
| `tests/test_joint_ffbs.py` | the shared joint FFBS reproduces the previous in-line HSA-steady sampler bit-for-bit; FFBS draws match an independently-assembled analytic smoother; `θ₀=0` collapses const-theta onto HSA steady; the inflation row alone still moves `N̄` when every firm-count row is missing |
| `tests/test_particle_gibbs_missing_n.py` | Particle Gibbs handles the annual-Q4 missing-N pattern: a missing `N^obs` drops only the firm-count term; one-step invariance against the exact γ=0 joint FFBS, with the tolerance **calibrated from a split-half null** rather than a fixed constant |
| `tests/test_prior_wiring.py` | every configured prior field reaches the sampler; `γ` is not KAPPA_SCALE-rescaled; state-variance scales stay in the transformed-N decade |
| `tests/test_conditional_ml.py` | the conditional marginal likelihood's identity, ordinate factors and truncation constant |
| `tests/test_observation_design_default.py` | config, library default and the estimation driver all resolve to `annual_q4`; metadata *readers* still fall back to `quarterly_interpolated`; run directories always carry the design |
| `tests/test_competition_measurement.py`, `test_transforms.py`, `test_common.py` | data-side transforms and the observation-vector construction |
| `tests/test_wrappers_and_diagnostics.py`, `test_report_and_robustness.py`, `test_identification.py`, `test_statistical_updates.py` | wrapper plumbing, report builders, identification helpers |

### Q.6 Documents created

| File | Contents |
|---|---|
| `docs/estimation_specification.md` | this document — §A–§Q |
| `docs/data_dictionary.md` | every observed series: symbol, code name, source, transform, units, sign, missing-data treatment, models |
| `docs/estimation_flow.md` | one complete MCMC iteration per model, step by step |
| `docs/code_equation_crosswalk.md` | compact equation ↔ code tables + report traceability |

No pre-existing documentation was overwritten (`docs/` did not exist before).
