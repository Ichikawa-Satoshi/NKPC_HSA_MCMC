# Code Review and Identification Report: NKPC-HSA State-Space MCMC

**Date:** 2026-07-02.
**Scope:** Full review of the repository against Fujiwara & Matsuyama, "Competition and the Phillips Curve" (JME 2026), plus re-estimation and identification diagnostics after fixes.

---

## 0. Summary of conclusions

1. **The samplers correctly implement the intended empirical model.** All sign conventions, timing, lag alignment, FFBS recursions, conditional posteriors, and unit conversions in the `ces`, `hsa_steady`, and `hsa_dynamic` Gibbs samplers are correct. Four implementation problems were found and fixed (Section 3): an inexact state sampler in `hsa_full`, mismatched priors in the Chib marginal-likelihood module, state-variance priors mis-scaled by two orders of magnitude, and stale "restricted" results produced before the `kappa_t` constraint existed.
2. **The negative/zero `delta` in the output-gap and markup specifications is not a coding error.** It is genuine (weak) likelihood information: in this sample (1982Q1–2012Q4, T=124), the Phillips-curve slope measured on the *unemployment gap* fell as trend competition fell (`delta > 0`, the paper's prediction), but the slope measured on the *output gap* did not flatten — sub-period OLS shows it rising from 0.04 to 0.10 (BN gap), which mechanically implies `delta < 0`. The inverse-markup and labor-share gaps carry almost no inflation signal at all, so `kappa_0` (and hence `delta`) is weakly identified there.
3. **The unemployment gap "looks good" because unemployment carries the strongest Phillips-curve signal in this sample, not because of the state-space structure.** The latent N decomposition (`Nbar`/`Nhat`) is well identified in *every* specification (decomposition RMSE 0.003–0.01, tight state paths); what differs across specifications is only the informativeness of `x_t` for inflation surprises.
4. After the fixes, the baseline **unemployment-gap results are decisively theory-consistent**: `delta = +0.030 (sd 0.011, P(δ>0) ≈ 1.00)`, `kappa_0 = +0.069 (0.034)`, and the implied slope path `kappa_t` declines from ≈0.16 (1982) to ≈0.00 (2012) — the flattening-with-concentration story of the paper.

---

## 1. The intended empirical model (reconstruction from the paper)

The paper is theoretical ("No data was used"). Its estimable object is the log-linearized NKPC under HSA with Rotemberg pricing, Eq. (21):

π̂_t = β(1−δ_exit)·E_t π̂_{t+1} + [(ζ(z)−1)/χ]·mc_t − (1/χ)·[(1−ρ(z))/ρ(z)]·N̂_t

with two testable implications:

- **Steady-state effect (Second law):** the slope (ζ(z)−1)/χ is *increasing in the number of firms N* (more firms → higher z → higher price elasticity ζ). Parameterizing the slope as κ_t = κ₀ + δ·N̄_t, theory predicts **δ > 0**.
- **Dynamic effect of entry:** the cost-push term enters with a **negative sign on N̂_t**; writing it as −θ_t·N̂_t with θ_t = θ₀ + γ·N̄_t, theory predicts **θ > 0**, and under the Third law (ρ′(z)>0) the magnitude (1−ρ)/ρ shrinks with N, so **γ < 0**.

The repository implements a hybrid empirical version with survey expectations (Cleveland Fed 1-year `Epi`), a hybrid backward/forward weighting (α, 1−α), an AR(1) for the activity variable x_t, and an unobserved-components model for competition:

- Inflation: π_t − Eπ_t = α(π_{t−1} − Eπ_t) + κ_t·x_t − θ_t·N̂_t + e_t, with e_t = λ·ζ_t + η_t
- Activity: x_t = φ₁·x_{t−1} + ζ_t
- Competition: N_obs,t = N̂_t + N̄_t + ν_t; N̂_t AR(2) (stationarity enforced); N̄_t random walk with drift n
- κ_t = κ₀ + δ·N̄_t; θ_t = θ₀ + γ·N̄_t (model-dependent)

with N in centered ten-log-point units, `N_Gustavo` = number of US listed firms (thousands, annual, PCHIP-interpolated to quarterly). This is a reasonable empirical operationalization of Eq. (21); the sign conventions in the code match the paper (the θ regressor is −N̂, so positive θ = paper's negative dynamic term).

## 2. Does the code implement this correctly?

Yes, after review of every sampler block:

- **Regression blocks** (`alpha, kappa_0, delta[, theta_0, gamma]`): conjugate Gaussian conditionals with correct design columns `[a_t, x/100, x·N̄/100, −N̂, −N̂·N̄]`, correct internal KAPPA_SCALE=100 handling, correct λζ offset, correct storage in physical units.
- **FFBS blocks**: `hsa_steady` uses an exact joint 3-state Kalman/FFBS with the inflation measurement row `[0, 0, (δ/100)x_t]` — correct. `hsa_dynamic` uses an exact correlated-noise filter; the innovation covariance S = HPH′+R+HC+C′H and cross term PH′+C are the correct correlated-noise formulas, and under the default `e_zeta_only` restriction the cross-covariance C is exactly zero, so the standard backward recursion is exact.
- **φ conditional** correctly combines the x-equation and inflation-equation information under the λ-coupling (verified algebraically).
- **Constraint machinery**: `kappa_t` is enforced as a path constraint via validator; bounds are converted physical→internal exactly once.
- **Unit conversions**: priors physical→internal (×100 for κ-like), draws internal→physical (÷100), tables/figures/SDDR consume physical draws with physical priors — no double scaling anywhere.
- **Chains/seeds/burn-in/thinning**: independent chains via `SeedSequence.spawn`, burn-in discarded before storage, thinning via `store_every`, R-hat/ESS via ArviZ — all sound.

## 3. Bugs found and fixes applied

### 3.1 `hsa_full` state sampler was not a valid Gibbs sampler (fixed)

**File:** `analysis/gibbs/func_gibbs/hsa_full/model.py`.
**Issue:** The N̂ and N̄ FFBS blocks used fabricated pseudo-measurement variances (`sigma_u2*target_scale`, `sigma_eps2*rw_scale`, scale=0.1 fixed) to tie states to `N_obs`. The stated model had no N measurement error, and the invariant distribution of the sampler was not the posterior of any coherent model.
**Fix:** Introduced the same measurement equation as the other HSA models, `N_obs_t = N̂_t + N̄_t + ν_t`, ν_t ~ N(0, σ_N²), with σ_N² sampled (IG(a_N, b_N)) and used as the measurement variance in *both* conditional FFBS blocks. Conditional on one state path, each block is now an **exact** linear-Gaussian conditional, so the two-block Gibbs is exact for the stated model. σ_N draws are stored.
**Why it matters:** The pseudo-variances injected arbitrary information into the states. Under the old sampler `theta_0` was spuriously "significant" (≈0.22–0.24 with P(θ₀>0)≈0.98–1.00 for the HP output gap and labor share); under the exact sampler these collapse to ≈0.00–0.07 — the apparent significance was an artifact.

### 3.2 Chib marginal likelihood used wrong priors and a wrong measurement variance (fixed)

**File:** `analysis/gibbs/func_gibbs/gibbs_marginal_likelihood.py` (and `src/nkpc_hsa/inference/model_comparison.py`).
**Issues:** (i) prior and posterior-ordinate terms were hard-coded with priors that do not match the sampling priors — most severely ρ₁, ρ₂ ~ N(0.2, 0.2) instead of N(0.5, 0.2)/N(−0.5, 0.2), δ ~ N(0.1, 0.2) instead of the configured N(0.0, 0.02), and a fixed sd of 0.2 for every coefficient; (ii) the Kalman likelihood fixed the N measurement variance at 1e-6 while the posterior draws were generated with an estimated σ_N²; σ_N² appeared in neither the prior nor the ordinate. The Chib identity logm = loglik + logprior − logordinate then does not hold for the model that was actually estimated.
**Fix:** All prior and ordinate terms now take a `priors` argument (physical units, `priors_*.yaml` shape) with sampler-consistent defaults; `model_comparison.py` passes each run's saved `priors.json`. The Kalman likelihoods use the posterior σ_N², and a σ_N² prior + Rao-Blackwellized ordinate term was added. Remaining known approximation (documented in the output notes): the AR(2) stationarity truncation and hard constraints are not reflected in the prior normalization.
**Why it matters:** biased log marginal likelihoods and Bayes factors; irrelevant for parameter posteriors.

### 3.3 State-variance priors mis-scaled by ~2 orders of magnitude (fixed)

**Files:** `configs/priors_baseline.yaml`, `configs/priors_weak.yaml`, `configs/priors_tight.yaml`.
**Issue:** When the N transform changed to centered ten-log-point units (June 27), the IG priors on σ_u², σ_ε², σ_N² stayed at IG(2, 2). The transformed N series has *total* quarterly-change variance ≈ 0.011, but IG(2, 2) has essentially zero mass below 0.01 and prior mean 2. With T=124 the prior scale term (b=2) dominates the likelihood term (0.5·Σresid² ≈ 0.7), forcing posterior state-shock variances to ≈0.1 (posterior σ_u, σ_ε, σ_N ≈ 0.33, 0.33, 0.29 in the old runs) — 10× too large in standard deviation. The same held for the u/ε diagonal of the `hsa_dynamic` inverse-Wishart scale.
**Fix:** rescaled to the data decade (baseline: b_u = 0.02, b_ε = 0.01, b_N = 0.01, S_Σ = diag(3, 3, 0.06, 0.03); weak/tight analogously), with comments in the YAML and README/CLAUDE.md documenting the unit convention.
**Why it matters (econometrically):** with inflated state noise, the N̄/N̂ decomposition becomes spuriously volatile (old runs: posterior N̄ wandering over [−10.9, 7] against a data range of [−2.7, 2.2]) and the interaction regressor x·N̄ inherits draw-specific noise. This is classical errors-in-variables at the level of the Gibbs draw: it attenuates δ toward its prior mean of 0 and destabilizes its sign. After the fix, σ_N ≈ 0.023–0.029, the decomposition is tight, and δ moved from −0.011 to −0.002 for the HP output gap (i.e., part of the "wrong sign" *was* this artifact) and sharpened from +0.029 (sd 0.014) to +0.030 (sd 0.011) for the unemployment gap.

### 3.4 Stale "restricted" results: constraint silently ignored (re-ran; now enforced)

All runs in `results/runs/` predated the June-27 commit that introduced the `kappa_t` path-constraint validator. In the stored `restricted_kappa` runs, **75% of the kept draws violate the constraint** and the sampler logged zero rejections — the constraint was never applied, and the "restricted" tables were byte-identical to unrestricted ones. With current code the constraint demonstrably binds (66–74% rejection rate on the labor-share spec). All estimation was re-run (Section 5). One informative failure: for `inv_markup` the κ_t ≥ 0 restricted posterior is numerically infeasible (rejection sampling exhausts 1000 tries) because the unrestricted posterior puts essentially no mass on an all-positive κ_t path — that spec has no positive-slope information at all.

### 3.5 Not bugs (checked and cleared)

- `N_Gustavo` "original_series" is the number of listed firms in thousands (7.5→9.5→6.0); the single log transform is correct (no double-log).
- `unemp_gap = NROU − UNRATENSA` is a positive activity measure, so κ > 0 is the correct expected sign.
- Prior/posterior overlays, SDDR, tables: physical units consistently; no double ×10 scaling of δ/γ.
- CES sampler, `hsa_steady` FFBS, correlated-noise filter in `hsa_dynamic`: algebra verified.
- Tests: all pass; two regression tests added (`sigma_N` in `hsa_full`; Chib prior threading).

## 4. Why the unemployment gap looks good but δ and κ do not (diagnostic evidence)

**(a) The state block is identified everywhere; the Phillips-curve block is not.** In all five specs the posterior N̄+N̂ reconstruction matches N_obs with RMSE 0.003–0.01 and R-hat ≈ 1.00 — the latent decomposition is pinned down by the N measurement equation, essentially independent of the inflation equation. So "good-looking states" carry no evidence that δ/κ are well estimated.

**(b) δ's likelihood information is weak and proxy-dependent.** OLS on observables (y = π−Eπ on [π₋₁−Eπ, x, x·N̄_HP]) gives:

| spec | OLS δ (t-stat) | sub-period κ: 1st half → 2nd half | baseline posterior δ (P>0) | weak-prior δ (P>0) |
|---|---|---|---|---|
| unemployment_gap | +0.057 (2.8) | 0.185 → −0.013 | **+0.030 (1.00)** | +0.049 (0.94) |
| labor_share_gap_hp | +0.081 (2.2) | 0.043 → −0.056 | +0.009 (0.70) | +0.032 (0.81) |
| output_gap_bn | −0.076 (−2.7) | 0.039 → 0.103 | −0.013 (0.18) | −0.031 (0.11) |
| output_gap_hp | −0.042 (−1.2) | 0.176 → 0.190 | −0.002 (0.43) | −0.005 (0.44) |
| inv_markup | −0.009 (−0.2) | −0.024 → −0.036 | −0.001 (0.48) | −0.002 (0.49) |

The MCMC posterior means equal the precision-weighted compromise between the OLS estimate and the N(0, 0.02) prior to within ≈0.003 in every spec — the sampler is faithfully transmitting exactly the (weak) likelihood information. Design collinearity is *not* the problem: corr(x, x·N̄) ∈ [−0.65, −0.28], condition numbers 1.2–5.1, posterior corr(κ₀, δ) ≤ 0.31.

**(c) The sign of δ is an economic fact about which activity measure flattened.** Over 1982–2012 the unemployment-based slope collapsed (consistent with the paper: fewer firms → flatter curve, δ>0), but the output-gap-based slope did not — the 2008–09 recession produced a large negative output gap together with disinflation, keeping the late-sample output-gap slope alive (rolling-window figure: `results/figures/identification_rolling_kappa_vs_Nbar.png`). Since N̄ falls in the second half, a non-falling slope mechanically maps to δ ≤ 0. This is the well-known unemployment/output Okun divergence in this period, not a coding issue.

**(d) κ₀ weak identification is a data problem for two specs.** The inverse-markup gap and the labor-share gap have essentially zero correlation with inflation surprises in this sample (OLS κ ≈ 0, posterior learning about κ₀ mostly from the prior). Any interaction coefficient built on a zero main effect is unidentified. The output gaps and the unemployment gap all deliver κ₀ ≈ 0.06–0.14 with P(κ₀>0) ≥ 0.92.

**(e) θ and γ (dynamic entry effect) are only marginally identifiable with this N series.** `N_Gustavo` is annual and PCHIP-interpolated (quarterly ΔN autocorrelation 0.95): quarterly N̂ variation is largely an interpolation artifact, so high-frequency identification of θ is illusory. After the `hsa_full` fix, θ₀ is ≈0 everywhere except a weak positive for the unemployment gap (+0.066, P=0.85, theory-consistent), and γ is mildly negative everywhere (−0.02 to −0.03, P(γ>0) ≈ 0.04–0.19), which matches the Third-law prediction but with learning rates of only 0.11–0.17 (posterior variance barely below prior variance). `hsa_dynamic`'s constant θ is slightly negative and insignificant — pooled over specs, θ should be treated as unidentified at quarterly frequency.

**(f) MCMC health (corrected runs).** R-hat ≤ 1.01 for all `hsa_steady`/`ces` parameters (δ ESS ≈ 3000 of 3200 stored); independent seeds (12345 vs 99999) agree to the third decimal. In `hsa_full` the headline parameters are healthy (δ, κ₀: R-hat ≤ 1.03, ESS 700–3200), but the nuisance block mixes slowly under the two-block state sampler: γ (ESS 33–70, R-hat ≤ 1.04) and especially the trend drift `n` (ESS ≈ 11, R-hat up to 1.34) and ρ₁, ρ₂ (R-hat ≈ 1.10) — the states and `n` are strongly posterior-dependent, and the block scheme moves them jointly only slowly. For `hsa_full` conclusions about `n`/ρ, run longer chains or add an ancillarity-sufficiency interweaving step; `hsa_steady`, whose joint 3-state FFBS mixes well, is the more reliable variant for the κ_t path. Trace/autocorrelation: `results/figures/trace_hsa_steady_unemp_corrected.png`; prior-vs-posterior: `results/figures/delta_prior_posterior_by_spec.png`; full tables: `results/tables/mcmc_diagnostics.csv`.

**(g) Robustness (corrected code).**

| δ, hsa_steady | baseline N(0,.02) | weak N(0,.1) | tight N(0,.01) | κ_t≥0 restricted |
|---|---|---|---|---|
| unemployment_gap | +0.030 (P 1.00) | +0.049 (P .94) | +0.016 (P .97) | +0.023 (P .99) |
| output_gap_bn | −0.013 (P .18) | −0.031 (P .11) | −0.007 (P .20) | −0.009 (P .23) |
| output_gap_hp | −0.002 (P .43) | −0.005 (P .44) | −0.002 (P .43) | −0.002 (P .44) |
| labor_share_gap_hp | +0.009 (P .70) | +0.032 (P .81) | +0.004 (P .67) | +0.004 (P .63) |
| inv_markup | −0.001 (P .48) | −0.002 (P .49) | −0.001 (P .48) | infeasible |

The unemployment-gap δ>0 and the BN-output-gap δ<0 are robust across prior scales, constraints, seeds, and the steady/full model variants; the other three specs are prior-dominated.

## 5. Overall attribution

The problem the user observed is a **combination**, in this order of importance:

1. **Data/specification reality (largest):** δ>0 is supported only by unemployment-based (and weakly labor-share-based) slopes in 1982–2012; output-gap slopes did not flatten in this sample, and two activity proxies have no inflation signal. The competition series is annual/interpolated and ends ~2012, cutting off the post-2012 flattening evidence and any high-frequency N̂ identification.
2. **Mis-scaled state-variance priors (moderate, fixed):** inflated state noise attenuated δ and pushed HP-output-gap δ spuriously negative.
3. **`hsa_full` sampler approximation (moderate, fixed):** produced spuriously significant θ₀.
4. **Stale outputs (bookkeeping, re-run):** all previous "restricted" results were actually unrestricted; the whole results tree predated the June-27 transform/constraint fixes.
5. **Chib module priors (fixed):** affected model comparison only.

## 6. Supporting artifacts

- `results/figures/identification_rolling_kappa_vs_Nbar.png` — rolling 10-year OLS slope vs. N trend, per spec (the single most informative picture).
- `results/figures/kappa_t_paths_corrected.png` — posterior κ_t paths; unemployment gap shows 0.16→0.00 flattening.
- `results/figures/delta_prior_posterior_by_spec.png` — prior-vs-posterior for δ.
- `results/figures/trace_hsa_steady_unemp_corrected.png` — traces/autocorrelations.
- `results/diagnostics/identification/*.csv` — data-scale, period-OLS, identification, and key-diagnostics tables (regenerated).
- Corrected runs: `results/runs/*_20260702_*` (37 runs: 4 models × 5 specs baseline; weak/tight/restricted/seed robustness for `hsa_steady`).

## 7. Recommended next steps

1. **Lead with the unemployment-gap (and secondarily labor-share) specification** when presenting δ; report the output-gap specs as the honest caveat that the flattening is a labor-market phenomenon in this sample. Do not average across specs.
2. **Extend the competition series past 2012** (the flattest Phillips-curve decade is missing) and/or use the TNIC-based series already in `data/raw/competition/` as a second measure; consider estimating on annual averages or adding a measurement equation that only loads N_obs in the fourth quarter, so interpolation noise cannot masquerade as N̂ identification.
3. **Drop or de-emphasize θ/γ at quarterly frequency** unless a genuinely quarterly competition measure is available; alternatively restrict θ_t = θ ≥ 0 as a theory-signed robustness spec.
4. **Prior for δ:** the baseline N(0, 0.02) is defensible but deliberately conservative — it halves the likelihood estimate. Report weak-prior (N(0, 0.1)) results alongside, as the likelihood is proper and the weak posterior is data-dominated (learning rate ≈ 0.8).
5. If marginal likelihoods will be reported, note the remaining Chib caveat (stationarity truncation not in the prior normalization) or switch to bridge sampling for the final paper.
6. Consider a single joint FFBS for `hsa_full` (linearizing the γ·N̄·N̂ term around the current draw with a Metropolis correction) if exactness beyond the two-block scheme is desired; the current two-block sampler is exact but can mix slowly for γ.
