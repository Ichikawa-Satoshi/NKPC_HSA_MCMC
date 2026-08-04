# One complete MCMC iteration, model by model

Companion to `docs/estimation_specification.md`. Every step gives: what is held fixed, what is
drawn, the conditional posterior, the equation it comes from, the actual code, what identifies it,
and what is passed on.

**Global MCMC settings** (`configs/models.yaml:1-6`, applied by `wrappers.run_model`):
`n_iter = 12000`, `burn = 4000`, `thin = 5`, `chains = 2`, `seed = 12345`
→ 1,600 retained draws per chain, 3,200 per cell.

Chain seeds are spawned deterministically (`wrappers.py:503-509`):
```python
seed_seq = np.random.SeedSequence(seed)
child_seeds = seed_seq.spawn(chains)
chain_seed = int(child.generate_state(1)[0])
```

**Notation.** `y_t = π_t − E_tπ_{t+1}`, `a_t = π_{t−1} − E_tπ_{t+1}`, `ζ_t = x_t − φ_1x_{t−1}`,
`κ_t^eff = κ_t/100`. Superscript `(m)` = current iteration; `(m−1)` = previous.

---

## 0. The `fixed` block-pinning mechanism (CES and HSA steady only)

`func_nkpc_ces` and `func_nkpc_hsa_decomp_tv_kappa_kalman` accept `opts["fixed"]`, a dict that
**pins named blocks to supplied values and skips their draw**. Every conditional below is
therefore wrapped in a guard:

```python
        if "sigma_zeta2" not in fixed:
            sigma_zeta2 = _sample_invgamma(
```

Recognised block names:

| Model | Pinnable blocks | Source |
|---|---|---|
| CES | `beta`, `lambda_ez`, `phi_1`, `sigma_zeta2`, `sigma_eta2` | `ces/model.py:121-134` |
| HSA steady | the above **plus** `rho`, `sigma_u2`, `n`, `sigma_eps2`, `sigma_N2` | `hsa_steady/model.py:522-546` |

An unrecognised name raises immediately (`ValueError: Unknown fixed block(s)`).

**Why it exists.** It is the reduced-run support for Chib's marginal likelihood
(`gibbs/conditional_ml.py:425`): the reduced Gibbs runs reuse these exact production
conditionals rather than a reimplementation, which is what makes the posterior-ordinate factors
match the sampler by construction.

**It is inert in production.** `run_model` never puts `"fixed"` in `opts`
(`inference/wrappers.py:519-526`), so `fixed == {}` and every guard is true. The other three
models (`hsa_dynamic`, `hsa_const_theta`, `hsa_full`) have no such mechanism.

For readability the step descriptions below omit these guards; assume each draw is wrapped in
`if "<block>" not in fixed:` for CES and HSA steady.

---

## 1. CES

`src/nkpc_hsa/gibbs/ces/model.py`, `func_nkpc_ces` — loop at line 155.
**No latent states.** Four blocks per sweep.

### STEP 1 — (α, κ)

**Held fixed** φ_1, λ_eζ, σ_ζ², σ_η² from iteration m−1.
**Drawn** `β = (α, κ_int)'`.

**Conditional posterior**
```
β | φ_1, λ, σ_η², data  ~  N(b₁, V₁)
V₁ = (V₀⁻¹ + X'X/σ_η²)⁻¹ ,   b₁ = V₁(V₀⁻¹b₀ + X'y*/σ_η²)
```

**Model equation** `y_t = α a_t + κ x_t + λζ_t + η_t`, rearranged to `y_t − λζ_t = α a_t + κ x_t + η_t`.

**Actual code** (`ces/model.py:156-169`)
```python
y = pi_t - pi_expect
zeta = x_t - phi_1 * x_tm1
X = np.column_stack([a_t, x_t / KAPPA_SCALE])
y_adj = y - lambda_ez * zeta
post_cov = np.linalg.inv(X.T @ X / sigma_eta2 + prior_prec)
post_mean = post_cov @ (X.T @ y_adj / sigma_eta2 + prior_prec @ prior_mean)
beta = draw_with_constraints(lambda: _mvnrnd(post_mean, post_cov, rng), ("alpha", "kappa"), ...)
```
`a_t` → α; `x_t/100` → κ (internal). `prior_prec` is diagonal (`:151`).

**Identified by** covariance of net-of-expectations inflation with lagged inflation and with `x_t`.
**Passes on** α, κ (→ `kappa_eff = κ/100`).

### STEP 2 — λ_eζ

**Held fixed** the new α, κ; old σ_η².
**Conditional** `λ | · ~ N(m, v)` from regressing `e_t = y − αa − κ^eff x` on `ζ_t`.
```python
            e_base = y - alpha * a_t - kappa_eff * x_t
            post_var_lambda = 1.0 / (
                lambda_prec0 + float(np.sum(zeta**2)) / sigma_eta2
            )
            post_mean_lambda = post_var_lambda * (
                mu_lambda * lambda_prec0 + float(np.dot(zeta, e_base)) / sigma_eta2
            )
```
(`ces/model.py:177-183`)
**Identified by** the contemporaneous correlation between the inflation shock and the activity
innovation. **Interpretation** the simultaneity correction — it stops `x_t` being treated as
exogenous.

### STEP 3 — φ_1

**Conditional** Normal, combining the AR(1) for `x` **and** the `e = λζ + η` channel
(`ces/model.py:187-202`):
```python
prec_phi = (phi_prec0 + float(np.sum(x_tm1**2)) / sigma_zeta2
            + (lambda_ez**2) * float(np.sum(x_tm1**2)) / sigma_eta2)
mean_num_phi = (mu_phi * phi_prec0 + float(np.dot(x_tm1, x_t)) / sigma_zeta2
                - lambda_ez * float(np.dot(x_tm1, y - alpha*a_t - kappa_eff*x_t - lambda_ez*x_t)) / sigma_eta2)
phi_1 = float(mean_num_phi / prec_phi + rng.standard_normal() / np.sqrt(prec_phi))
```

### STEP 4 — σ_ζ², σ_η²

`ζ` is **recomputed with the new φ_1** first (`:204`), then:
```python
        zeta = x_t - phi_1 * x_tm1
        eta = y - alpha * a_t - kappa_eff * x_t - lambda_ez * zeta

        if "sigma_zeta2" not in fixed:
            sigma_zeta2 = _sample_invgamma(
                a_z + 0.5 * T,
                b_z + 0.5 * float(np.sum(zeta**2)),
                rng,
            )
```
(`ces/model.py:205-213`; `sigma_eta2` follows the same shape at `:214-220`.)
The `if … not in fixed:` guard is the reduced-run mechanism described in §0 below;
in a production run `fixed` is empty and the guard is always true.
**Stored** `sigma_e2 = λ²σ_ζ² + σ_η²` (total NKPC shock variance) and `rho_corr = λσ_ζ/σ_e`
(`:229-233`).

---

## 2. HSA steady

`src/nkpc_hsa/gibbs/hsa_steady/model.py`, `func_nkpc_hsa_decomp_tv_kappa_kalman` — loop at 624.
**Nine steps. States are drawn LAST**, so every parameter block conditions on the previous
iteration's path.

### STEP 1 — (α, κ₀, δ)

**Held fixed** `N̄^(m−1)`, φ_1, λ, σ_η².
**Equation** `y_t = α a_t + (κ₀ + δN̄_t)x_t + λζ_t + η_t`.
```python
        X = np.column_stack(
            [
                a_t,
                x_t / KAPPA_SCALE,
                (x_t * Nbar) / KAPPA_SCALE,
            ]
        )
```
(`hsa_steady/model.py:632-638`; the draw itself is `_sample_beta_gaussian` wrapped in
`draw_with_constraints` at `:670-683`.)
| column | → | equation term |
|---|---|---|
| `a_t` | α | `α a_t` |
| `x_t/100` | κ₀ | `κ₀ x_t` |
| `(x_t*Nbar)/100` | δ | `δ N̄_t x_t` |

**Identified by** for δ, the covariance of the inflation residual with the **interaction**
`x_t·N̄_t` given `a_t` and `x_t`: i.e. whether the inflation–slack relationship is *steeper in
years when the trend firm count is high*.
**Passes on** `κ_t = κ₀ + δN̄`, `κ_t^eff = κ_t/100`.

### STEP 2 — λ_eζ · STEP 3 — φ_1 · STEP 4 — σ_ζ², σ_η²
As CES steps 2–4, with `κ_t^eff x_t` in place of `κ x_t` (`:695-754`).

### STEP 5 — (ρ₁, ρ₂) then σ_u²

**Held fixed** the state path `states^(m−1)`, including the sampled `N̂_{−1} = states[0,1]`.
**Equation** `N̂_t = ρ₁N̂_{t−1} + ρ₂N̂_{t−2} + u_t`.
**Conditional** Normal **truncated to the stationary triangle**, by rejection.
```python
rho1, rho2 = _sample_ar2_coeffs(
    Nhat=Nhat, sigma_state2=sigma_u2, ...,
    max_tries=ar2_max_tries, current=(rho1, rho2), stats=ar2_stats,
    initial_lag=float(states[0, 1]),
)                                                                  # :750-762
resid_u = states[1:, 0] - rho1 * states[:-1, 0] - rho2 * states[:-1, 1]
sigma_u2 = _sample_invgamma(pri["a_u"] + 0.5*resid_u.size,
                            pri["b_u"] + 0.5*float(np.sum(resid_u**2)), rng)
```
Inside `_sample_ar2_coeffs` (`:134-138`) the regression is
```python
y = Nhat[1:]
second_lag = np.concatenate([[float(initial_lag)], Nhat[:-2]])
X = np.column_stack([Nhat[:-1], second_lag])
```
so **T−1** rows, using the sampled `N̂_{−1}` rather than discarding the first transition.
`resid_u` uses `states[:, 1]` for the second lag, which is the same object — consistent.

**Identified by** the autocovariance structure of the *sampled* `N̂` path. ⚠️ Since `N̂` is itself
weakly identified, so are ρ₁, ρ₂; their values differ radically between PCHIP and annual-Q4.

### STEP 6 — n, then σ_ε²

**Equation** `N̄_t = n + N̄_{t−1} + ε_t`, i.e. `ΔN̄_t = n + ε_t`.
```python
dNbar = Nbar[1:] - Nbar[:-1]                                       # :778
post_var_n = 1.0 / (1.0 / pri["sigma_n"]**2 + dNbar.size / sigma_eps2)
post_mean_n = post_var_n * (pri["mu_n"] / pri["sigma_n"]**2 + float(np.sum(dNbar)) / sigma_eps2)
n_drift = float(post_mean_n + np.sqrt(post_var_n) * rng.standard_normal())
resid_eps = Nbar[1:] - n_drift - Nbar[:-1]
sigma_eps2 = _sample_invgamma(pri["a_eps"] + 0.5*resid_eps.size,
                              pri["b_eps"] + 0.5*float(np.sum(resid_eps**2)), rng)
```
**Identified by** the average drift of the sampled trend. ⚠️ This is the worst-mixing scalar in
every HSA cell — the latent level ridge surfacing in the parameter block.

### STEP 7 — σ_N²

```python
resid_N = finite_N_residuals(N_obs, Nhat, Nbar)                    # :807
sigma_N2 = _sample_invgamma(pri["a_N"] + 0.5*resid_N.size,
                            pri["b_N"] + 0.5*float(np.sum(resid_N**2)), rng)
```
`finite_N_residuals` masks on `np.isfinite(N_obs)`, so under annual-Q4 only the **31** Q4 residuals
enter — degrees of freedom `a_N + 31/2`, not `a_N + 124/2`.

### STEP 8 — the joint latent path  `s_{0:T−1} ~ p(· | Θ^(m), y, N^obs)`

**Held fixed** every parameter just drawn.
**Drawn** the entire path in one exact FFBS sweep.

```python
        obs_offset = lambda_ez * zeta                               # :818

        Nbar, Nhat, states = _sample_states_kalman_ffbs(            # :820-846
            N_obs=N_obs,
            pi_t=pi_t,
            ...
            m0=m0,
            P0=P0,
            rng=rng,
        )
```
(kwargs elided with `...`; the source passes one per line.)
which builds `ỹ` and delegates (`:376-398`):
```python
y_tilde = (pi_t - pi_expect - alpha * (pi_tm1 - pi_expect)
           - (kappa0 / KAPPA_SCALE) * x_t - obs_offset)
return sample_joint_competition_states_ffbs(
    N_obs=N_obs, y_tilde=y_tilde,
    h_nhat=np.zeros(T, dtype=float),
    h_nbar=(delta / KAPPA_SCALE) * x_t, ...)
```

State vector, `F`, `c`, `Q`, `H_t`, `R_t`, the forward recursion and the backward draw are in
`docs/estimation_specification.md` §D and §H. For HSA steady `h_nhat = 0`, so `N̂` is informed
**only** by the firm-count equation and the AR(2) transition.

**Identified by** `N^obs_t` pins `N̄_t + N̂_t`; the RW-vs-AR(2) spectral contrast and the `s₀` prior
split it; `δx_t` adds a weak inflation-side signal on `N̄_t`.
**Passes on** `N̄`, `N̂`, `states` to the next iteration's steps 1, 5, 6, 7.

### STEP 9 — store

```python
kappa_t = kappa0 + delta * Nbar                                     # :843 (recomputed AFTER the state draw)
...
kappa0_draws[store_idx] = kappa0 / KAPPA_SCALE                      # :861-862
delta_draws[store_idx]  = delta  / KAPPA_SCALE
kappa_t_draws[store_idx] = kappa_t / KAPPA_SCALE                    # :882
```
Stored only when `it > n_burn and (it - n_burn) % store_every == 0` (`:859`).

---

## 3. HSA dynamic

`src/nkpc_hsa/gibbs/hsa_dynamic/model.py`, `func_nkpc_hsa_decomp_joint_fullSigma` — loop from 1202.
Seven steps. **Different architecture**: a full shock covariance `Σ = Var(e,ζ,u,ε)` replaces the
`e = λζ + η` decomposition, and every coefficient block is drawn from the *conditional* Gaussian
given the other three shocks.

### STEP 0 — current residuals
`_compute_state_residuals` (`:528`) forms `e, ζ, u, ε` at the current parameters and states.

### STEP 1 — (α, κ, θ)
```python
    mean_e, var_e = _conditional_e_all(Sigma, zeta, u, eps)             # :597

    X = np.column_stack(
        [
            a_t,
            x_t / KAPPA_SCALE,
            -Nhat,
        ]
    )                                                                   # :599-605
beta = draw_with_constraints(lambda: _sample_beta_gaussian_weighted(
    y=y - mean_e, X=X, var=var_e, ...), ("alpha", "kappa", "theta"), ...)
```
`mean_e` / `var_e` are the mean and variance of `e_t` **conditional on** `(ζ_t, u_t, ε_t)`; the
regression is run on `y − mean_e` with variance `var_e`. Under the default `e_zeta_only`
restriction only `ζ` is informative. Column `-Nhat` → θ, i.e. the term `−θN̂_t`.

### STEP 2 — φ_1 (`_sample_phi_full`, `:636`), conditional on `(e,u,ε)`.
### STEP 3 — (ρ₁, ρ₂) (`_sample_ar2_coeffs_full`, `:673`), conditional on `(e,ζ,ε)`, truncated to the stationary triangle.
### STEP 4 — n (`_sample_n_full`, `:769`), conditional on `(e,ζ,u)`.

### STEP 5 — Σ
```python
Sigma = _sample_Sigma(...)                                          # :301
```
Dispatches on `covariance_structure`. Production default `e_zeta_only` →
`_sample_sigma_restricted` (`:236`), which draws the 2×2 `(e,ζ)` block from an inverse-Wishart and
`σ_u²`, `σ_ε²` from inverse-gammas, **directly on the constrained space** — it does not zero the
off-diagonals of an unconstrained draw.

### STEP 6 — σ_N²
`finite_N_residuals` as in HSA steady (`:1305-1309`).

### STEP 7 — joint states
`_sample_states_joint_ffbs_fullSigma` (`:817`). Same `F`, `c`, state vector as §D, but with:
- `Q` carrying `cov(u,ε)` off-diagonals (`:913-921`),
- `H = [[1,0,1], [−θ,0,0]]` (`:977-983`) — note `H[1,2] = 0`: with δ = 0 the **trend does not enter
  the inflation row at all**, so `N̄` is informed only by `N^obs`,
- a state/measurement cross-covariance `C_base` (`:933-940`) and the generalised gain
  `K = (P H' + C) S⁻¹` with `S = H P H' + R + HC + C'H'` (`:994-1000`).

Under `e_zeta_only` all cross terms are zero and this reduces to the standard recursion.

---

## 4. HSA const-theta

`src/nkpc_hsa/gibbs/hsa_const_theta/model.py`, `func_nkpc_hsa_const_theta` — loop at 186.
Nine steps, **structurally identical to HSA steady** with θ added to the coefficient block and
`h_nhat = −θ` in the observation row.

### STEP 1 — (α, κ₀, δ, θ)
```python
        zeta = x_t - phi_1 * x_tm1
        y_adj = y - lambda_ez * zeta

        X = np.column_stack(
            [
                a_t,
                x_t / KAPPA_SCALE,
                (x_t * Nbar) / KAPPA_SCALE,
                -Nhat,
            ]
        )                                                           # :189-198
        beta = draw_with_constraints(
            lambda: _sample_beta_gaussian(
                y_adj,
                X,
                sigma2=sigma_eta2,
                prior_mean=beta_prior_mean,
                prior_var=beta_prior_var,
                rng=rng,
            ),
            ("alpha", "kappa_0", "delta", "theta"),
            coefficient_constraints,
            validators=_kappa_t_constraint_validators(Nbar, coefficient_constraints),
            stats=constraint_stats,
        )                                                           # :210-223
        alpha = float(beta[0])
        kappa0 = float(beta[1])
        delta = float(beta[2])
        theta = float(beta[3])
```
| column | → | equation term | economic meaning |
|---|---|---|---|
| `a_t` | α | `α a_t` | backward-looking inertia |
| `x_t/100` | κ₀ | `κ₀ x_t` | slope at average competition |
| `(x_t*Nbar)/100` | δ | `δ N̄_t x_t` | how the slope moves with the trend firm count |
| `-Nhat` | θ | `−θ N̂_t` | cyclical-entry cost-push channel |

**Held fixed** `N̄^(m−1)`, `N̂^(m−1)`, φ_1, λ, σ_η².
**Identified by** δ from the interaction `x_t N̄_t`; θ from the cycle `N̂_t` **given** `x_t` and
`x_tN̄_t`. Because `N̂` is nearly the negative of `N̄` up to the measurement equation, θ and δ are
identified from nearly orthogonal directions but both inherit the state indeterminacy.

### STEP 2 — λ_eζ  (`:233-242`) — note the `+ theta * Nhat` term restoring the entry channel:
```python
e_base = y - alpha * a_t - kappa_t_eff * x_t + theta * Nhat
```
### STEP 3 — φ_1 (`:244-256`) · STEP 4 — σ_ζ², σ_η² (`:258-266`)
```python
eta = y - alpha * a_t - kappa_t_eff * x_t + theta * Nhat - lambda_ez * zeta
```
### STEP 5 — (ρ₁,ρ₂), σ_u² (`:268-286`) · STEP 6 — n, σ_ε² (`:288-300`) · STEP 7 — σ_N² (`:302-306`)
Identical to HSA steady steps 5–7, including `initial_lag=float(states[0, 1])`.

### STEP 8 — joint latent path (the change from the legacy implementation)

```python
y_tilde_state = y - alpha * a_t - (kappa0 / KAPPA_SCALE) * x_t - lambda_ez * zeta
Nbar, Nhat, states = sample_joint_competition_states_ffbs(
    N_obs=N_obs,
    y_tilde=y_tilde_state,
    h_nhat=np.full(T, -theta, dtype=float),
    h_nbar=(delta / KAPPA_SCALE) * x_t,
    n_drift=n_drift, rho1=rho1, rho2=rho2,
    sigma_eta2=sigma_eta2,
    sigma_u2=sigma_u2,
    sigma_eps2=sigma_eps2,
    sigma_N2=sigma_N2,
    m0=m0,
    P0=P0,
    rng=rng,
)                                                                   # :308-326
```
`H_t = [[1,0,1], [−θ, 0, δx_t/100]]`. **One** exact draw of the whole path.

⚠️ **Contrast with the legacy path.** `func_nkpc_hsa_full_static_theta`
(`gibbs/hsa_full/model.py:960`) implements the same model with two alternating blocks. Both kernels
are valid; the alternating one mixes catastrophically because corr(N̄₀,N̂₀) ≈ −0.999 makes the
shared level move with autocorrelation ρ² ≈ 0.998 per sweep. Measured on the core-CPI ×
unemployment cell, identical data/priors/seeds:

| | alternating FFBS | joint FFBS |
|---|---|---|
| `Nbar` path R̂ / ESS | 1.876 / 2.9 | **1.003 / 796** |
| `n` R̂ / ESS | 1.527 / 3.8 | **1.003 / 689** |
| `θ` R̂ / ESS | 1.027 / 62.8 | **1.001 / 2560** |

### STEP 9 — store (`:331-360`)
`theta_t_draws[store_idx] = np.full(T, theta)` — a constant path, stored so the output schema
matches `hsa_full`.

---

## 5. HSA full

**Production sampler: Particle Gibbs** (§5b), for both observation designs. §5a is the
superseded alternating-FFBS implementation, retained for validation and no longer reachable
from `run_model`.

### 5a. Alternating FFBS — SUPERSEDED — `gibbs/hsa_full/model.py`, loop at 702

**STEP 1 — (α, κ₀, δ, θ₀, γ)** (`:710-756`)
```python
columns = [a_t, x_t / KAPPA_SCALE, (x_t * Nbar) / KAPPA_SCALE, -Nhat]
if not static_theta:
    columns.append(-(Nhat * Nbar))
X = np.column_stack(columns)
```
The fifth column `−N̂_tN̄_t` → γ.

**STEPS 2–7** — λ_eζ, φ_1, σ_ζ²/σ_η², (ρ₁,ρ₂)/σ_u², n/σ_ε², σ_N². Same as const-theta except
`θ_t = θ₀ + γN̄_t` replaces the constant θ (`:762`, `:785`).
⚠️ The AR(2) block here uses `Nhat_initial_lag` carried in a scalar variable
(`:802, 850`) rather than `states[0,1]`, because there is no single `states` array.

**STEP 8a — `N̂_{0:T} | N̄_{0:T}`** (`:824-848`), `_sample_ar2_states_ffbs_tv_theta`
```python
Nhat_states = _sample_ar2_states_ffbs_tv_theta(
    y_target=N_obs - Nbar, rho1=rho1, rho2=rho2, sigma_state2=sigma_u2,
    ..., kappa_t=kappa_t_eff, theta_t=theta_t, obs_offset=obs_offset, ...)
Nhat = Nhat_states[0]
Nhat_initial_lag = float(Nhat_states[1, 0])
```
Inside (`:230-244`): `y_pi = α π_{t−1} + (1−α)Eπ + κ_t x_t + λζ − π_t`, `H = [[1,0],[θ_t,0]]`.
Sign check: `π = απ_{t−1} + (1−α)Eπ + κ_tx − θ_tN̂ + λζ + η` ⟹ `θ_tN̂ = y_pi + η`. ✓
Linear-Gaussian because `θ_t` is **known** once `N̄` is conditioned on.

**STEP 8b — `N̄_{0:T} | N̂_{0:T}`** (`:851-871`), `_sample_rw_states_ffbs_tv_theta_kappa`
```python
y2 = (pi_t[t] - pi_expect[t] - alpha * (pi_tm1[t] - pi_expect[t])
      - kappa0 * x_t[t] + theta0 * Nhat[t] - obs_offset[t])
h2 = delta * x_t[t] - gamma * Nhat[t]                               # :365-373
```
(`kappa0`, `delta` are passed pre-divided by KAPPA_SCALE at `:862-863`.)
Sign check: `y − αa = (κ₀+δN̄)x − (θ₀+γN̄)N̂ + λζ + η` ⟹
`y − αa − κ₀x + θ₀N̂ − λζ = (δx − γN̂)N̄ + η`. ✓

Both blocks are exact conditionals, so the alternating scheme is a **valid** Gibbs kernel.

### 5b. Particle Gibbs — PRODUCTION — `gibbs/hsa_full_pg/model.py`, loop at 444

Reached by `run_model("hsa_full")` through the facade `models/hsa_full.py`. Particle count comes
from `configs/models.yaml → defaults.n_particles` (512) and is recorded in run metadata.

**STEPS 1–7 are imported verbatim** from `hsa_full` (`:30-42`) — identical priors, scaling and
conditionals. Only step 8 differs:

**STEP 8 — one conditional-SMC sweep** (`:527-540`)
```python
pg = sample_states_particle_gibbs(
    y=y, a_t=a_t, x_t=x_t, zeta=zeta, N_obs=N_obs,
    alpha=alpha, kappa0_eff=kappa0 / KAPPA_SCALE, delta_eff=delta / KAPPA_SCALE,
    theta0=theta0, gamma=gamma, lambda_ez=lambda_ez,
    rho1=rho1, rho2=rho2, n_drift=n_drift,
    sigma_eta2=sigma_eta2, sigma_u2=sigma_u2, sigma_eps2=sigma_eps2, sigma_N2=sigma_N2,
    Nbar_ref=Nbar, Nhat_ref=Nhat, Nhat_ref_lag=Nhat_initial_lag,
    m0_Nhat=m0_Nhat, ..., n_particles=n_particles, rng=rng)
        Nhat = pg["Nhat"]
        Nbar = pg["Nbar"]
        Nhat_initial_lag = pg["Nhat_lag"]
```
The sweep itself — reference pinning, bootstrap propagation, dual-likelihood weights, log-sum-exp
normalisation, terminal draw, ancestor tracing — is documented in
`docs/estimation_specification.md` §I.

**Why a sweep is needed at all.** `−γN̄_tN̂_t` is bilinear in the state, so `H_t` would have to
depend on `s_t`. No Kalman recursion exists. Conditional SMC handles an arbitrary observation
density and remains a valid MCMC kernel for **any** particle count because the previous path is
retained throughout.

---

## 6. What is passed between iterations

| Model | Carried from iteration m to m+1 |
|---|---|
| CES | α, κ, φ_1, λ, σ_ζ², σ_η² |
| HSA steady | the above + ρ₁, ρ₂, n, σ_u², σ_ε², σ_N², **and `N̄`, `N̂`, `states`** |
| HSA dynamic | α, κ, θ, φ_1, ρ₁, ρ₂, n, Σ, σ_N², **`N̄`, `N̂`, `states`** |
| HSA const-theta | HSA steady's set + θ |
| HSA full | const-theta's set + γ, and `Nhat_initial_lag` as a separate scalar |

The state path is the object that couples the parameter blocks: every coefficient conditional is
computed at the *previous* iteration's path, and the path is redrawn at the *current* iteration's
coefficients. That circularity is what a Gibbs sampler resolves, and it is why the latent-path
diagnostics matter as much as the coefficient diagnostics
(`docs/estimation_specification.md` §K).
