# Code ↔ equation crosswalk

Compact lookup table. Full narrative in `docs/estimation_specification.md`;
per-iteration detail in `docs/estimation_flow.md`.
Paths are relative to the repository root. Line numbers are post-August-2026.

---

## 1. Data construction

| Mathematical object | Code expression | File | Function | Used by | Meaning |
|---|---|---|---|---|---|
| `π_t = 100(P_t/P_{t−4} − 1)` | `100 * (series_q / series_q.shift(4) - 1)` | `src/nkpc_hsa/dataprep/func_data_build.py:22-25` | `yoy_pct` | headline CPI, PPI | four-quarter inflation |
| `π_t = 100(log P_t − log P_{t−4})` | `100 * (np.log(series_q) - np.log(series_q).shift(4))` | `func_data_build.py:27-29` | `log_yoy` | **core CPI only** | four-quarter log inflation |
| `π_{t−1}` | `out[f"{col}_prev"] = out[col].shift(1)` | `func_data_build.py:252` | `build_dataset` | all | one-quarter lag |
| `x_t = u*_t − u_t` | `tt_gap["NROU"] - tt_gap["UNRATE"]` | `func_data_build.py:231` | `load_labor_market_series` | unemployment specs | **negative** unemployment gap; `UNRATE` is the SA series (see data dictionary) |
| HP gap | `hp_filter_series(100.0 * out["output"], lamb=1600)` | `src/nkpc_hsa/dataprep/build.py:61-63` | `add_hp_output_gap` | HP specs | 100-log-point output gap |
| labor-share gap | `hp_filter_series(out["labor_share_100log"], lamb)` | `build.py:102` | `load_labor_share_gap` | `labor_share_gap_hp` | 100-log-point labor-share gap |
| `N^obs_t = (100 log N_t − mean)/10` | `raw = 100.0*np.log(arr); (raw - np.mean(raw))/10.0` | `src/nkpc_hsa/dataprep/transforms.py:44-46` | `transform_competition_series` | all HSA | ten-log-point firm count |
| annual → quarterly N | `PchipInterpolator(xx, yy)` | `func_data_build.py:63` | `annual_to_quarterly_pchip` | PCHIP design | interpolated firm count |
| Q4-only N | `if int(period.quarter) == 4 …: out[i] = …` else `nan` | `src/nkpc_hsa/dataprep/competition.py:131-141` | `build_competition_observation` | annual-Q4 design | mixed-frequency observation |
| annual N centered on PCHIP mean | `center = np.mean(100.0*np.log(reference))` | `src/nkpc_hsa/inference/wrappers.py:174-178` | `_transform_annual_competition_like_quarterly` | annual-Q4 | keeps units comparable |
| complete-case sample | `data[[…]].dropna()` | `wrappers.py` | `_coerce_model_data` | production report | T = 124 |
| establishment experiment sample | configured 1993Q2–2012Q4 window plus joint complete cases | `wrappers.py` | `_coerce_model_data` | HSA const-theta experiment | T = 79 |
| quarterly establishment signal | `Ehat_obs_t = lambda_E*Nhat_t + omega_t` | `gibbs/common/joint_ffbs.py` | `sample_joint_competition_states_ffbs` | HSA const-theta experiment | theta, lambda_E and sigma_E sampled |

---

## 2. Within-sampler transformations

| Mathematical object | Code expression | File:line | Models | Meaning |
|---|---|---|---|---|
| `y_t = π_t − E_tπ_{t+1}` | `y = pi_t - pi_expect` | `hsa_steady/model.py:562` | all | inflation net of expectations |
| `a_t = π_{t−1} − E_tπ_{t+1}` | `a_t = pi_tm1 - pi_expect` | `hsa_steady/model.py:561` | all | inertia regressor |
| `ζ_t = x_t − φ_1x_{t−1}` | `zeta = x_t - phi_1 * x_tm1` | `hsa_steady/model.py:630, 735` | all | activity innovation |
| `y*_t = y_t − λζ_t` | `y_adj = y - lambda_ez * zeta` | `hsa_steady/model.py:630` | ces, steady, const-θ, full | LHS net of the simultaneity term |
| `e_t = y − αa − κ_t^eff x + θN̂` | `e_base = y - alpha*a_t - kappa_t_eff*x_t + theta*Nhat` | `hsa_const_theta/model.py:235` | const-θ | NKPC shock |
| `η_t` | `eta = y - alpha*a_t - kappa_t_eff*x_t + theta*Nhat - lambda_ez*zeta` | `hsa_const_theta/model.py:259` | const-θ | orthogonal inflation residual |
| `KAPPA_SCALE = 100` | `x_t / KAPPA_SCALE` | every model's `X` | all | internal κ-scaling |
| physical→internal prior | `(mean*100, sd*100)` | `models/common.py:86-87` | κ, κ₀, δ **only** | prior rescaling |
| internal→physical draw | `kappa0 / KAPPA_SCALE` | `hsa_steady/model.py:861` | κ, κ₀, δ **only** | stored in physical units |

---

## 3. Inflation-equation regressors

| Equation term | Code column | File:line | Models | Coefficient |
|---|---|---|---|---|
| `α a_t` | `a_t` | `hsa_const_theta/model.py:191` | all | α |
| `κ x_t` | `x_t / KAPPA_SCALE` | `ces/model.py:159`; `hsa_dynamic/model.py:602` | ces, dynamic | κ |
| `κ₀ x_t` | `x_t / KAPPA_SCALE` | `hsa_steady/model.py:635`; `hsa_const_theta/model.py:192` | steady, const-θ, full | κ₀ |
| `δ N̄_t x_t` | `(x_t * Nbar) / KAPPA_SCALE` | `hsa_steady/model.py:636`; `hsa_const_theta/model.py:193` | steady, const-θ, full | δ |
| `−θ N̂_t` | `-Nhat` | `hsa_dynamic/model.py:603`; `hsa_const_theta/model.py:194` | dynamic, const-θ | θ |
| `−θ₀ N̂_t` | `-Nhat` | `hsa_full/model.py:714` | full | θ₀ |
| `−γ N̄_t N̂_t` | `-(Nhat * Nbar)` | `hsa_full/model.py:730` | full | γ |
| `λ_eζ ζ_t` | *separate block*, regress `e_base` on `zeta` | `hsa_const_theta/model.py:234-242` | ces, steady, const-θ, full | λ_eζ |

⚠️ `λ_eζ` is **not** a column of `X`. It is drawn in its own block after β.

---

## 4. State-space matrices

| Mathematical object | Code expression | File:line | Models |
|---|---|---|---|
| `s_t = (N̂_t, N̂_{t−1}, N̄_t)'` | `states[:, 0], states[:, 1], states[:, 2]` | `common/joint_ffbs.py:168` (returned as `Nbar, Nhat, states`) | all HSA |
| `F = [[ρ₁,ρ₂,0],[1,0,0],[0,0,1]]` | `np.array([[rho1, rho2, 0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]])` | `joint_ffbs.py:111-118` | steady, const-θ |
| `c = (0,0,n)'` | `np.array([0.0, 0.0, n_drift])` | `joint_ffbs.py:119` | steady, const-θ |
| `Q = diag(σ_u²,0,σ_ε²)` | `np.diag([sigma_u2, 0.0, sigma_eps2])` | `joint_ffbs.py:120` | steady, const-θ |
| `N^obs_t = N̄_t + N̂_t + ν_t` | `[1.0, 0.0, 1.0]` | `joint_ffbs.py:141` | steady, const-θ |
| inflation row (steady) | `h_nhat=np.zeros(T)`, `h_nbar=(delta/KAPPA_SCALE)*x_t` | `hsa_steady/model.py:386-387` | steady |
| inflation row (const-θ) | `h_nhat=np.full(T,-theta)`, `h_nbar=(delta/KAPPA_SCALE)*x_t` | `hsa_const_theta/model.py:314-315` | const-θ |
| `R = diag(σ_N², σ_η²)` | `np.diag([sigma_N2, sigma_eta2])` | `joint_ffbs.py:142` | steady, const-θ |
| `ỹ_t` (obs net of known part) | `y - alpha*a_t - (kappa0/KAPPA_SCALE)*x_t - lambda_ez*zeta` | `hsa_const_theta/model.py:310` | const-θ |
| missing-N row drop | `if np.isfinite(N_obs[t]): … else: …` | `joint_ffbs.py:137-146` | all HSA |
| `s_0 ~ N(m₀, P₀)` | `m_pred[0]=m0; P_pred[0]=force_pd(P0)` | `joint_ffbs.py:130-135` | steady, const-θ |
| `H` (dynamic) | `[[1.0,0.0,1.0],[-theta,0.0,0.0]]` | `hsa_dynamic/model.py:977-983` | dynamic |
| `Q` with `cov(u,ε)` | `[[var_u,0,cov_u_eps],[0,0,0],[cov_u_eps,0,var_eps]]` | `hsa_dynamic/model.py:913-921` | dynamic |
| state/meas. cross-cov `C` | `[[0,cov_eu],[0,0],[0,cov_eeps]]` | `hsa_dynamic/model.py:933-940` | dynamic |

---

## 5. Kalman / FFBS recursions

| Mathematical object | Code expression | File:line |
|---|---|---|
| `m_{t\|t−1} = c + F m_{t−1}` | `m_pred[t] = c + F @ m_filt[t - 1]` | `joint_ffbs.py:134` |
| `P_{t\|t−1} = F P_{t−1} F' + Q` | `P_pred[t] = force_pd(F @ P_filt[t-1] @ F.T + Q)` | `joint_ffbs.py:135` |
| `S_t = H P_{t\|t−1} H' + R` | `S = force_pd(H @ P_pred[t] @ H.T + R)` | `joint_ffbs.py:148` |
| `K_t = P_{t\|t−1} H' S⁻¹` | `K = P_pred[t] @ H.T @ inv(S)` | `joint_ffbs.py:149` |
| `v_t = z_t − H m_{t\|t−1}` | `innov = y_obs - H @ m_pred[t]` | `joint_ffbs.py:150` |
| `m_t = m_{t\|t−1} + K v_t` | `m_filt[t] = m_pred[t] + K @ innov` | `joint_ffbs.py:151` |
| Joseph: `(I−KH)P(I−KH)' + KRK'` | `force_pd((I3-KH) @ P_pred[t] @ (I3-KH).T + K @ R @ K.T)` | `joint_ffbs.py:154-155` |
| `s_{T−1} ~ N(m_{T−1},P_{T−1})` | `states[-1] = _mvnrnd(m_filt[-1], P_filt[-1], rng)` | `joint_ffbs.py:159` |
| `A_t = P_t F' P_{t+1\|t}⁻¹` | `A = P_filt[t] @ F.T @ inv(Ptp1)` | `joint_ffbs.py:163` |
| `m_t + A(s_{t+1} − c − F m_t)` | `mean_s = m_filt[t] + A @ (states[t+1] - c - F @ m_filt[t])` | `joint_ffbs.py:164` |
| `P_t − A P_{t+1\|t} A'` | `cov_s = force_pd(P_filt[t] - A @ Ptp1 @ A.T)` | `joint_ffbs.py:165` |

---

## 6. Particle Gibbs (HSA full)

| Mathematical object | Code expression | File:line |
|---|---|---|
| reference path pinned at `t=0` | `Nhat0[0] = Nhat_ref[0]; Nbar0[0] = Nbar_ref[0]` | `hsa_full_pg/model.py:179-181` |
| ancestor resampling | `a = rng.choice(P, size=P, p=W)` | `hsa_full_pg/model.py:194` |
| reference lineage fixed | `a[0] = 0` | `hsa_full_pg/model.py:195` |
| `N̂_t^i = ρ₁N̂_{t−1}^i + ρ₂N̂_{t−2}^i + u_t^i` | `new_Nhat = rho1*parent_Nhat + rho2*parent_Nlag + su*rng.standard_normal(P)` | `hsa_full_pg/model.py:200` |
| AR(2) genealogy | `new_Nlag = parent_Nhat` | `hsa_full_pg/model.py:201` |
| `N̄_t^i = n + N̄_{t−1}^i + ε_t^i` | `new_Nbar = n_drift + parent_Nbar + se*rng.standard_normal(P)` | `hsa_full_pg/model.py:202` |
| reference re-inserted at `t` | `new_Nhat[0] = Nhat_ref[t]; new_Nbar[0] = Nbar_ref[t]` | `hsa_full_pg/model.py:205-207` |
| `μ_t^i` | `alpha*a_t + kappa0_eff*x_t + delta_eff*x_t*Nbar - theta0*Nhat - gamma*Nbar*Nhat + lambda_ez*zeta_t` | `hsa_full_pg/model.py:76-83` |
| `log p(y_t\|s_t^i)` | `ll = -0.5 * (y_t - mu)**2 / sigma_eta2` | `hsa_full_pg/model.py:84` |
| `log p(N^obs_t\|s_t^i)` | `ll - 0.5 * (N_obs_t - Nbar - Nhat)**2 / sigma_N2` | `hsa_full_pg/model.py:86` |
| log-sum-exp normalisation | `w = np.exp(logw - m); W = w / s` | `hsa_full_pg/model.py:91-94` |
| `ESS_t = 1/Σ(W^i)²` | `ess = 1.0 / float(np.sum(W * W))` | `hsa_full_pg/model.py:95` |
| terminal draw `J ~ Cat(W_T)` | `b[T-1] = int(rng.choice(P, p=W))` | `hsa_full_pg/model.py:217` |
| ancestor tracing | `b[t] = ancestors[t + 1, b[t + 1]]` | `hsa_full_pg/model.py:219` |

---

## 7. Alternating FFBS (HSA full) — SUPERSEDED, validation only

Production is Particle Gibbs (§6). These rows document the retained
`func_nkpc_hsa_full_alternating_ffbs` implementation.

| Mathematical object | Code expression | File:line |
|---|---|---|
| `N̂ \| N̄` observation | `H = np.array([[1.0, 0.0], [theta_series[t], 0.0]])` | `hsa_full/model.py:238` |
| `N̂ \| N̄` target | `y_pi = alpha*pi_tm1[t] + (1-alpha)*pi_expect[t] + kappa_series[t]*x_t[t] + obs_offset[t] - pi_t[t]` | `hsa_full/model.py:230-236` |
| `N̄ \| N̂` target | `y2 = pi_t[t] - pi_expect[t] - alpha*(...) - kappa0*x_t[t] + theta0*Nhat[t] - obs_offset[t]` | `hsa_full/model.py:365-372` |
| `N̄ \| N̂` loading | `h2 = delta * x_t[t] - gamma * Nhat[t]` | `hsa_full/model.py:373` |
| block 1 call | `Nhat_states = _sample_ar2_states_ffbs_tv_theta(y_target=N_obs - Nbar, …)` | `hsa_full/model.py:824-848` |
| block 2 call | `Nbar = _sample_rw_states_ffbs_tv_theta_kappa(y_target=N_obs - Nhat, …)` | `hsa_full/model.py:851-871` |

---

## 8. Conditional posteriors

| Mathematical object | Code expression | File:line |
|---|---|---|
| `V₁ = (V₀⁻¹ + X'X/σ²)⁻¹` | `Vn = inv(X.T @ X / sigma2 + V0_inv)` | `hsa_steady/model.py:78` |
| `b₁ = V₁(V₀⁻¹b₀ + X'y/σ²)` | `mn = Vn @ (X.T @ y / sigma2 + V0_inv @ prior_mean)` | `hsa_steady/model.py:79` |
| λ precision | `lambda_prec0 + np.sum(zeta**2)/sigma_eta2` | `hsa_const_theta/model.py:234` |
| φ precision | `1/sigma_phi**2 + Σx²_{t−1}/σ_ζ² + λ²Σx²_{t−1}/σ_η²` | `hsa_steady/model.py:297-301` |
| AR(2) design (initial-lag) | `y = Nhat[1:]; X = np.column_stack([Nhat[:-1], second_lag])` | `hsa_steady/model.py:136-138` |
| stationary triangle | `(abs(r2)<1) and ((r1+r2)<1) and ((r2-r1)<1)` | `hsa_steady/model.py:33-34` |
| rejection loop | `for attempt in range(1, max_tries+1): … if _is_stationary_ar2(...)` | `hsa_steady/model.py:155-162` |
| `n` posterior variance | `1.0/(1.0/sigma_n**2 + dNbar.size/sigma_eps2)` | `hsa_const_theta/model.py:289` |
| `σ² ~ IG(a+n/2, b+SSR/2)` | `_sample_invgamma(a + 0.5*n, b + 0.5*np.sum(resid**2), rng)` | `hsa_steady/model.py:48-51` |
| σ_N² finite-only residuals | `finite_N_residuals(N_obs, Nhat, Nbar)` | `common/competition.py:12-21` |
| Σ restriction (`e_zeta_only`) | `out = np.diag(np.diag(S)); out[0,1]=out[1,0]=S[0,1]` | `hsa_dynamic/model.py:58-61` |

---

## 9. Derived quantities

| Mathematical object | Code expression | File:line | Meaning |
|---|---|---|---|
| `κ_t = κ₀ + δN̄_t` | `kappa_t = kappa0 + delta * Nbar` | `hsa_steady/model.py:843`; `hsa_const_theta/model.py:328` | Phillips-curve slope path |
| stored `κ_t` (physical) | `kappa_t_draws[store_idx] = kappa_t / KAPPA_SCALE` | `hsa_steady/model.py:882` | per-draw, uses that draw's own `N̄` |
| `θ_t = θ₀ + γN̄_t` | `theta_t = theta0 + gamma * Nbar` | `hsa_full/model.py:874` | entry-coefficient path |
| `θ_t = θ` (const) | `np.full(T, theta, dtype=float)` | `hsa_const_theta/model.py:358` | constant path, schema parity |
| `σ_e = √(λ²σ_ζ² + σ_η²)` | `float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))` | `hsa_const_theta/model.py:331` | total NKPC shock sd |
| `ρ_eζ = λσ_ζ/σ_e` | `float((lambda_ez*np.sqrt(sigma_zeta2)) / max(sigma_e, 1e-12))` | `hsa_const_theta/model.py:332-336` | shock correlation |

---

## 9b. Predictive fit scores

| Mathematical object | Code expression | File | Meaning |
|---|---|---|---|
| plug-in score `S` | `-0.5*T*np.log(2*np.pi*sigma2) - 0.5*np.sum(resid**2)/sigma2` | `scripts/predictive_comparison.py` `plugin_score()` | in-sample, posterior-mean parameters **and** states |
| `LPD₁ = Σ_t log p(π_t \| π_{1:t−1}, x, N)` | `scores(ll)` → `logsumexp` over draws, summed over t | `predictive_comparison.py` `prequential_ces` / `prequential_hsa` | one-step-ahead, integrated over the posterior |
| WAIC, PSIS-LOO | `arviz` on the pointwise log-likelihood `ll` (S×T) | `predictive_comparison.py` `scores()` | max Pareto k̂ returned alongside |
| CES shared across designs | `find_run(model, spec, "quarterly_interpolated" if model == "ces" else freq)` | `predictive_comparison.py` | CES has no latent firm-count state |

⚠️ All four are computed on the estimation sample; none is a genuine out-of-sample score.

---

## 10. Diagnostics

| Mathematical object | Code expression | File:line |
|---|---|---|
| scalar group | `SCALAR_PARAMETERS = ["alpha","kappa",…,"sigma_N"]` | `scripts/12_build_cpi_ppi_report.py:53-72` |
| state group | `STATE_PATH_PARAMETERS = ["Nbar", "Nhat"]` | `scripts/12_build_cpi_ppi_report.py:73` |
| derived group | `DERIVED_PATH_PARAMETERS = ["kappa_t", "theta_t"]` | `scripts/12_build_cpi_ppi_report.py:74` |
| max R̂ over a group | `np.nanmax(np.asarray(az.rhat(idata.posterior[name])))` | `scripts/12_build_cpi_ppi_report.py:319` |
| min bulk ESS over a group | `np.nanmin(np.asarray(az.ess(idata.posterior[name], method="bulk")))` | `scripts/12_build_cpi_ppi_report.py:320` |
| skip constant variables | `if not np.isfinite(values).any() or float(np.nanstd(values)) <= 0.0: continue` | `scripts/12_build_cpi_ppi_report.py:317-318` |
| thresholds | `RHAT_LIMIT = 1.01; ESS_LIMIT = 400.0` | `scripts/12_build_cpi_ppi_report.py:32-33` |
| `†` mark | `value if converged else value + r"\textsuperscript{$\dagger$}"` | `scripts/12_build_cpi_ppi_report.py:388-391` |
| status string | `_conv_status(diagnostics, japanese=…)` | `scripts/12_build_cpi_ppi_report.py:394-400` |

---

## 11. Report traceability

Run selection for **every** report artifact:
```python
runs = load_report_runs(min_iter=…, competition_frequency=…)
```
`scripts/12_build_cpi_ppi_report.py:188`. Loads `results/runs/`, filters on
`estimation_revision == "2026-08-theta-centred"`, `period == "full"`,
`constraint_spec == "unrestricted"`, `n_transform == "log100_centered10"` and the requested
frequency, then keeps the **newest `run_id`** per `(model, data_spec, prior)` key
(`:151-154`). Two build-time guards then run:

```python
        assert_expected_sampler(run_set, model="hsa_full", expected="particle_gibbs", label=label)
        assert_expected_sampler(run_set, model="hsa_const_theta", expected="joint_ffbs", label=label)
```

plus `assert_single_sampler_per_cell`, so a cell cannot be reported under two samplers and a
model cannot silently fall back to a superseded one.

| Report value | Posterior variable | Run key | Summary | Builder | Output file |
|---|---|---|---|---|---|
| `\CoreUnempDelta` | `posterior["delta"]` | `(hsa_steady, unemployment_gap_core, baseline)`, PCHIP | mean, 2.5 %, 97.5 % — `_summary` `:244-254`; formatted `_fmt` `:287-290` as `%+.3f [%+.3f, %+.3f]` | `write_result_macros` `:784-830` | `results/tables/quarterly_interpolated/result_macros.tex` |
| `\CoreUnempDeltaBF` | `posterior["delta"]` + `priors.json` | same | Savage–Dickey, `_bf10` `:269-285`; `_fmt_num` 1 dp, `>999` above 1000 | `write_result_macros` | same |
| `\CoreUnempKappaStart` | `posterior["kappa_t"][:, 0]` | same | `nanmean` of the first column — `_path_summary` `:256-267`; `%+.3f` | `write_result_macros` | same |
| `\CoreUnempKappaEnd` | `posterior["kappa_t"][:, -1]` | same | `nanmean` of the last column | `write_result_macros` | same |
| `\ReportRunCount` | — | all report cells | `len(runs)` | `write_result_macros:794` | same |
| `\ReportWarningCount` | all scalar params | all cells | count of `not converged` (**scalar** rule) | `write_result_macros:791-795` | same |
| `\ReportStateWarningCount` | `Nbar`, `Nhat` | all cells | count of `state_converged is False` | `write_result_macros:792` | same |
| `\ReportJointWarningCount` | all three groups | all cells | count of `not joint_converged` | `write_result_macros:793` | same |
| `\HsaFullSampler` | `attrs["state_sampler"]` | `hsa_full` cells | joined label set | `write_result_macros:796-801` | same |
| θ / γ columns | `posterior["theta"]` / `["theta_0"]` / `["gamma"]` | model × price cells | `_fmt(_summary(...))` + `†` | `build_model_table` `:455-490`, `build_output_gap_model_tables` `:492-526` | `unemployment_by_model.tex`, `output_gap_*_by_model.tex` |
| convergence flags | all three groups | every cell | `_conv_status` | all table builders | every coefficient table |
| `tab:group-convergence` | scalar / `n` / state / derived | baseline unemployment cells | per-group max R̂, min ESS, worst scalar name | `build_group_convergence_diagnostics` | `annual_q4/group_convergence_diagnostics.tex` |
| `tab:fit-comparison` | plug-in score, LPD₁, WAIC, PSIS-LOO, max Pareto k̂ | 4 models × 3 prices, **annual-Q4** | `scripts/predictive_comparison.py` writes the scores; the plug-in score is `plugin_score()` in that file | `scripts/make_fit_comparison_table.py` | `annual_q4/fit_comparison.tex` |
| `tab:main-convergence` | — | all cells | per-model run/warning tallies + sampler | `build_run_manifest` `:679-757` | `convergence_summary.tex` |
| `tab:headline` | δ, BF, κ₀, κ_t path, **δ's own R̂/ESS** | `hsa_steady` × 9 specs, **annual-Q4** | as above | `scripts/make_headline_results_table.py` `build()` | `cpi_ppi_report/annual_q4/headline_results.tex` |
| `tab:model-comp` | slope, δ, BF | 5 models × 3 prices, **annual-Q4** | as above | `make_headline_results_table.py` `build()` | `cpi_ppi_report/annual_q4/model_comparison_unemp.tex` |
| `tab:headline-pchip`, `tab:model-comp-pchip` | same, interpolated design | PCHIP | as above | same, second `build()` call | `cpi_ppi_report/*.tex` |
| `\BiasDirect*` | `ces["kappa"]`, `hsa_dynamic["kappa"]`, `Nhat`, `theta`, `phi_1` | `inv_markup` cells | `_paired_difference` `:528-533`, `_hsa_implied_ovb` `:546-576` | `build_ces_hsa_bias_table` `:577-641` | `bias_macros.tex`, `ces_hsa_kappa_bias.tex` |
| `\CoreUnempKappaDrop` | `posterior["kappa_t"]` | main HSA-steady cell | posterior-mean start minus end | `_state_space_macros` | `result_macros.tex` |

**No translation pass.** The report is English-only. The table builders write English
directly and `_write_latex` raises if a table would contain CJK text, so the former
`cpi_ppi_report_en/` mirror is gone and `scripts/build_english_tables.py`, which had been
reduced to a no-op stub, has been deleted.

**Two observation designs.** `load_report_runs` is called once per design.
`competition_frequency="annual_q4"` produces the **main** tables into
`results/tables/annual_q4/`; `"quarterly_interpolated"` produces the
comparison tables into the base directory. Report sections §4–§6, §9–§10 and
Appendices A and C read the former; §7 and Appendix B read the latter.

**Conditional marginal likelihood (CES vs HSA steady)**:
`results/evidence/tables/conditional_ml.csv`, from `scripts/chib_marginal_likelihood.py`
→ `gibbs/conditional_ml.py`.

| quantity | code | notes |
|---|---|---|
| `log p(π,N^obs \| x,θ)` | `steady_joint_loglik` | Kalman, states integrated out, inflation row included |
| `log p(N^obs \| θ_N)` | `firm_count_loglik` | same filter, inflation row dropped |
| `log p(x \| φ₁,σ_ζ²)` | `activity_loglik` | the activity equation the Gibbs posterior also conditions on |
| `log m(π,N,x \| M)` | `conditional_marginal_likelihood(..., joint_target=True)` | Chib, sampler's own block order |
| `log m(N \| M)` | `firm_count_marginal_likelihood` | Chib over the five firm-count blocks |
| `log m(x)` | `activity_marginal_likelihood` | Chib over `(φ₁, σ_ζ²)`; identical for both models |
| `log p(π \| x,N^obs,M)` | `conditional_comparison` | joint − `m(N)` − `m(x)`; CES has no `m(N)` term |

The two subtracted terms are Occam factors for data being conditioned on, so they must be
refunded rather than charged. `m(x)` is computed once per cell and handed to both models, so
the run itself shows them conditioned on the identical quantity instead of assuming cancellation.

**Guard.** `_checked_logmeanexp` raises `OrdinateNotIdentified` when a Rao-Blackwellised
ordinate factor rests on fewer than `MIN_EFFECTIVE_ORDINATE_DRAWS` effective draws. The AR(2)
block mixes slowly enough that short reduced runs can put the whole average on one draw; that
produced a 660-log-point error on one seed before the guard existed. Reduced runs default to
`--n-keep 30000` and are executed in parallel (they pin only starred values, so they do not
depend on one another).

Earlier defective outputs are quarantined in `results/evidence/tables/` (quarantine removed), with a
`README.md` there listing the defects. Nothing reads that subdirectory.

---

## 12. Quick "where is it?" index

| I want to find… | Look at |
|---|---|
| the inflation equation actually estimated | §3 above, per model |
| the state-space matrices | §4 |
| the Kalman/FFBS recursion | §5 |
| the Particle-Gibbs sweep | §6 |
| how a prior reaches the sampler | `models/common.py:154-197` |
| why there is a `/100` | `docs/estimation_specification.md` §C.5 |
| what makes a cell `OK` vs `OK (coef)` vs `†` | §10, and `docs/estimation_specification.md` §K |
| which sampler produced a given table | `attrs["state_sampler"]`, surfaced in `convergence_summary.tex` and `report_run_manifest.tex` |
| what identifies a parameter | `docs/estimation_specification.md` §N |
| one full MCMC iteration | `docs/estimation_flow.md` |
