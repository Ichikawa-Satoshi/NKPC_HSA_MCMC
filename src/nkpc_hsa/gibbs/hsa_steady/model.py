from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.linalg import inv

from nkpc_hsa.gibbs.common.competition import finite_N_residuals, initial_competition_path
from nkpc_hsa.gibbs.common.constraints import constraint_stats_summary, draw_with_constraints
from nkpc_hsa.gibbs.common.joint_ffbs import sample_joint_competition_states_ffbs

# Kappa-related parameters are sampled internally on a KAPPA_SCALE-multiplied
# scale because the inflation regressors use x / KAPPA_SCALE. Stored posterior
# draws are converted back to physical units.
KAPPA_SCALE = 100.0

def _getd(d: Optional[dict[str, Any]], key: str, default: Any) -> Any:
    if isinstance(d, dict) and key in d and d[key] is not None:
        return d[key]
    return default


def _assert_all_pos(arr: Any, msg: str) -> None:
    values = np.asarray(arr, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(msg)


def _as_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _is_stationary_ar2(r1: float, r2: float) -> bool:
    return (abs(r2) < 1.0) and ((r1 + r2) < 1.0) and ((r2 - r1) < 1.0)


def _force_pd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


def _mvnrnd(mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, dtype=float), _force_pd(cov))


def _sample_invgamma(a_post: float, b_post: float, rng: np.random.Generator) -> float:
    if a_post <= 0.0 or b_post <= 0.0:
        raise ValueError("Inverse-gamma posterior parameters must be positive.")
    return 1.0 / rng.gamma(shape=a_post, scale=1.0 / b_post)


def _sample_beta_gaussian(
    y: np.ndarray,
    X: np.ndarray,
    sigma2: float,
    prior_mean: np.ndarray,
    prior_var: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    y = _as_1d(y)
    X = np.asarray(X, dtype=float)
    prior_mean = _as_1d(prior_mean)
    prior_var = _as_1d(prior_var)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")
    if X.shape[0] != y.size:
        raise ValueError("X and y lengths do not match.")
    if X.shape[1] != prior_mean.size or prior_mean.size != prior_var.size:
        raise ValueError("Prior dimensions do not match X.")

    _assert_all_pos(prior_var, "Prior variances must be positive.")
    _assert_all_pos([sigma2], "sigma2 must be positive.")

    V0_inv = np.diag(1.0 / prior_var)
    Vn = inv(X.T @ X / sigma2 + V0_inv)
    mn = Vn @ (X.T @ y / sigma2 + V0_inv @ prior_mean)
    return _mvnrnd(mn, Vn, rng)


def _summary(draws: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(draws, dtype=float)
    if arr.shape[0] == 0:
        raise ValueError("No posterior draws were stored. Check n_keep and store_every.")

    qs = np.quantile(arr, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975], axis=0)
    return {
        "draws": arr,
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(np.mean(arr, axis=0)),
        "quantiles": qs,
    }


def _init_states(N_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N_init = initial_competition_path(N_obs)
    T = N_init.size

    Nbar = np.zeros(T, dtype=float)
    k0 = min(2, T)
    Nbar[:k0] = N_init[:k0]

    for t in range(2, T):
        Nbar[t] = 0.7 * Nbar[t - 1] + 0.3 * N_init[t]

    Nhat = N_init - Nbar
    return Nbar, Nhat


def _sample_ar2_coeffs(
    Nhat: np.ndarray,
    sigma_state2: float,
    mu_rho1: float,
    sigma_rho1: float,
    mu_rho2: float,
    sigma_rho2: float,
    enforce_stationary: bool,
    rng: np.random.Generator,
    max_tries: int = 2000,
    current: tuple[float, float] | None = None,
    stats: dict[str, int] | None = None,
    initial_lag: float | None = None,
) -> tuple[float, float]:
    Nhat = _as_1d(Nhat)

    if initial_lag is None:
        if Nhat.size < 3:
            raise ValueError("Need T >= 3 to sample AR(2) coefficients.")
        y = Nhat[2:]
        X = np.column_stack([Nhat[1:-1], Nhat[:-2]])
    else:
        if Nhat.size < 2:
            raise ValueError("Need T >= 2 when the sampled initial lag is supplied.")
        y = Nhat[1:]
        second_lag = np.concatenate([[float(initial_lag)], Nhat[:-2]])
        X = np.column_stack([Nhat[:-1], second_lag])

    prior_prec = np.diag([1.0 / sigma_rho1**2, 1.0 / sigma_rho2**2])

    post_cov = inv(X.T @ X / sigma_state2 + prior_prec)
    post_mean = post_cov @ (
        X.T @ y / sigma_state2
        + prior_prec @ np.array([mu_rho1, mu_rho2], dtype=float)
    )

    if not enforce_stationary:
        draw = _mvnrnd(post_mean, post_cov, rng)
        return float(draw[0]), float(draw[1])

    if stats is not None:
        stats["draw_calls"] = stats.get("draw_calls", 0) + 1

    for attempt in range(1, max_tries + 1):
        draw = _mvnrnd(post_mean, post_cov, rng)
        r1, r2 = float(draw[0]), float(draw[1])
        if _is_stationary_ar2(r1, r2):
            if stats is not None:
                stats["proposals"] = stats.get("proposals", 0) + attempt
                stats["rejections"] = stats.get("rejections", 0) + attempt - 1
            return r1, r2

    if stats is not None:
        stats["proposals"] = stats.get("proposals", 0) + max_tries
        stats["rejections"] = stats.get("rejections", 0) + max_tries
        stats["fallbacks"] = stats.get("fallbacks", 0) + 1

    if current is not None and _is_stationary_ar2(float(current[0]), float(current[1])):
        return float(current[0]), float(current[1])
    if _is_stationary_ar2(float(post_mean[0]), float(post_mean[1])):
        return float(post_mean[0]), float(post_mean[1])
    if _is_stationary_ar2(mu_rho1, mu_rho2):
        return float(mu_rho1), float(mu_rho2)
    return 0.0, 0.0


def _kappa_t_constraint_validators(
    Nbar: np.ndarray,
    coefficient_constraints: dict[str, Any] | None,
    *,
    offset: int = 1,
) -> list:
    """``offset`` is the index of kappa_0 in the drawn beta vector.

    It is 1 in the baseline block ``(alpha, kappa_0, delta)`` and 0 under the
    purely forward-looking restriction, where alpha is not drawn.
    """
    bounds = dict((coefficient_constraints or {}).get("bounds", {}) or {})
    pair = bounds.get("kappa_t", bounds.get("kappa"))
    if pair is None:
        return []
    lower, upper = pair
    Nbar_arr = _as_1d(Nbar)

    def _valid(beta: np.ndarray) -> bool:
        kappa_t = float(beta[offset]) + float(beta[offset + 1]) * Nbar_arr
        if lower is not None and np.any(kappa_t < float(lower)):
            return False
        if upper is not None and np.any(kappa_t > float(upper)):
            return False
        return True

    return [_valid]


def _ar2_stats_summary(stats: dict[str, int]) -> dict[str, float | int]:
    proposals = int(stats.get("proposals", 0))
    rejections = int(stats.get("rejections", 0))
    return {
        "draw_calls": int(stats.get("draw_calls", 0)),
        "proposals": proposals,
        "rejections": rejections,
        "fallbacks": int(stats.get("fallbacks", 0)),
        "proposal_rejection_rate": float(rejections / proposals) if proposals else 0.0,
    }


def _common_priors(priors: dict[str, Any]) -> dict[str, float]:
    return {
        "mu_alpha": _getd(priors, "mu_alpha", 0.5),
        "sigma_alpha": _getd(priors, "sigma_alpha", 0.2),

        "mu_kappa0": _getd(priors, "mu_kappa_0", _getd(priors, "mu_kappa", 10.0)),
        "sigma_kappa0": _getd(priors, "sigma_kappa_0", _getd(priors, "sigma_kappa", 20.0)),

        "mu_delta": _getd(priors, "mu_delta", 10.0),
        "sigma_delta": _getd(priors, "sigma_delta", 20.0),

        "mu_phi": _getd(priors, "mu_phi_1", 0.7),
        "sigma_phi": _getd(priors, "sigma_phi_1", 0.2),

        # Optional correlation representation:
        #   e_t = lambda_ez * zeta_t + eta_t
        # Set orth=True in the main function to impose lambda_ez = 0.
        "mu_lambda": _getd(priors, "mu_lambda", 0.0),
        "sigma_lambda": _getd(priors, "sigma_lambda", 0.5),

        "mu_rho1": _getd(priors, "mu_rho1", 0.5),
        "sigma_rho1": _getd(priors, "sigma_rho1", 0.2),
        "mu_rho2": _getd(priors, "mu_rho2", -0.5),
        "sigma_rho2": _getd(priors, "sigma_rho2", 0.2),

        "mu_n": _getd(priors, "mu_n", 0.0),
        "sigma_n": _getd(priors, "sigma_n", 0.1),

        # eta variance in inflation equation
        "a_e": _getd(priors, "a_e", _getd(priors, "a_v", 2.0)),
        "b_e": _getd(priors, "b_e", _getd(priors, "b_v", 2.0)),

        # u variance in AR(2) Nhat equation
        "a_u": _getd(priors, "a_u", 2.0),
        "b_u": _getd(priors, "b_u", 2.0),

        # epsilon variance in random-walk Nbar equation
        "a_eps": _getd(priors, "a_eps", 2.0),
        "b_eps": _getd(priors, "b_eps", 2.0),

        # zeta variance in x equation
        "a_z": _getd(priors, "a_z", 0.001),
        "b_z": _getd(priors, "b_z", 0.001),

        # measurement error variance in:
        #   N_obs_t = Nhat_t + Nbar_t + measurement_error_t
        "a_N": _getd(priors, "a_N", _getd(priors, "a_m", 2.0)),
        "b_N": _getd(priors, "b_N", _getd(priors, "b_m", 2.0)),

        # Initial state prior for:
        #   s_0 = [Nhat_0, Nhat_{-1}, Nbar_0]'
        "m0_Nhat": _getd(priors, "m0_Nhat", 0.0),
        "m0_Nhat_lag": _getd(priors, "m0_Nhat_lag", 0.0),
        "m0_Nbar": _getd(priors, "m0_Nbar", 0.0),

        "P0_Nhat": _getd(priors, "P0_Nhat", 10.0),
        "P0_Nhat_lag": _getd(priors, "P0_Nhat_lag", 10.0),
        "P0_Nbar": _getd(priors, "P0_Nbar", 10.0),
    }


def _sample_phi_joint(
    *,
    x_t: np.ndarray,
    x_tm1: np.ndarray,
    y_tilde: np.ndarray,
    lambda_ez: float,
    sigma_zeta2: float,
    sigma_eta2: float,
    mu_phi: float,
    sigma_phi: float,
    rng: np.random.Generator,
) -> float:
    """
    Sample phi_1 from the conditional posterior.

    x_t equation:
        x_t = phi_1*x_{t-1} + zeta_t

    Optional correlated shock:
        e_t = lambda_ez*zeta_t + eta_t
    """
    x_t = _as_1d(x_t)
    x_tm1 = _as_1d(x_tm1)
    y_tilde = _as_1d(y_tilde)

    prec = (
        1.0 / sigma_phi**2
        + float(np.sum(x_tm1**2)) / sigma_zeta2
        + (lambda_ez**2) * float(np.sum(x_tm1**2)) / sigma_eta2
    )

    mean_num = (
        mu_phi / sigma_phi**2
        + float(np.dot(x_tm1, x_t)) / sigma_zeta2
        - lambda_ez * float(np.dot(x_tm1, y_tilde - lambda_ez * x_t)) / sigma_eta2
    )

    return float(mean_num / prec + rng.standard_normal() / np.sqrt(prec))


def _sample_states_kalman_ffbs(
    *,
    N_obs: np.ndarray,
    pi_t: np.ndarray,
    pi_tm1: np.ndarray,
    pi_expect: np.ndarray,
    x_t: np.ndarray,
    alpha: float,
    kappa0: float,
    delta: float,
    n_drift: float,
    rho1: float,
    rho2: float,
    sigma_eta2: float,
    sigma_u2: float,
    sigma_eps2: float,
    sigma_N2: float,
    obs_offset: np.ndarray,
    m0: np.ndarray,
    P0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kalman filter + FFBS for the joint state vector:

        s_t = [Nhat_t, Nhat_{t-1}, Nbar_t]'

    State equation:

        Nhat_t = rho1*Nhat_{t-1} + rho2*Nhat_{t-2} + u_t
        Nbar_t = n + Nbar_{t-1} + epsilon_t

    Measurement equations:

        N_obs_t = Nhat_t + Nbar_t + measurement_error_t

        pi_t - Epi_t - alpha*(pi_{t-1}-Epi_t)
              - kappa0*x_t - obs_offset_t
            = delta*x_t*Nbar_t + eta_t

    where obs_offset_t = lambda_ez*zeta_t.
    If orth=True, lambda_ez=0 and obs_offset_t=0.

    Thin wrapper over the shared exact joint FFBS in
    ``nkpc_hsa.gibbs.common.joint_ffbs``; hsa_steady is the ``h_nhat = 0``
    special case of that routine. The delegation is numerically exact -- it
    reproduces the previous in-line implementation bit for bit under the same
    seed (pinned by ``tests/test_joint_ffbs.py``), so existing hsa_steady
    posteriors are unaffected.
    """
    N_obs = _as_1d(N_obs)
    pi_t = _as_1d(pi_t)
    pi_tm1 = _as_1d(pi_tm1)
    pi_expect = _as_1d(pi_expect)
    x_t = _as_1d(x_t)
    obs_offset = _as_1d(obs_offset)

    T = N_obs.size

    if not (pi_t.size == pi_tm1.size == pi_expect.size == x_t.size == obs_offset.size == T):
        raise ValueError("All input series must have the same length.")

    y_tilde = (
        pi_t
        - pi_expect
        - alpha * (pi_tm1 - pi_expect)
        - (kappa0 / KAPPA_SCALE) * x_t
        - obs_offset
    )

    return sample_joint_competition_states_ffbs(
        N_obs=N_obs,
        y_tilde=y_tilde,
        h_nhat=np.zeros(T, dtype=float),
        h_nbar=(delta / KAPPA_SCALE) * x_t,
        n_drift=n_drift,
        rho1=rho1,
        rho2=rho2,
        sigma_eta2=sigma_eta2,
        sigma_u2=sigma_u2,
        sigma_eps2=sigma_eps2,
        sigma_N2=sigma_N2,
        m0=m0,
        P0=P0,
        rng=rng,
    )


def func_nkpc_hsa_decomp_tv_kappa_kalman(
    pi_data,
    pi_prev_data,
    Epi_data,
    x_data,
    x_prev_data,
    N_data,
    n_burn: int,
    n_keep: int,
    priors: Optional[dict[str, Any]] = None,
    opts: Optional[dict[str, Any]] = None,
    *,
    orth: bool = False,
) -> dict[str, Any]:
    """
    Gibbs sampler with Kalman/FFBS state draw for:

        pi_t = alpha*pi_{t-1}
               + (1-alpha)*E_t*pi_{t+1}
               + kappa_t*x_t
               + e_t

        x_t = phi_1*x_{t-1} + zeta_t

        N_obs_t = Nhat_t + Nbar_t + measurement_error_t

        Nhat_t = rho_1*Nhat_{t-1} + rho_2*Nhat_{t-2} + u_t

        Nbar_t = n + Nbar_{t-1} + epsilon_t

        kappa_t = kappa_0 + delta*Nbar_t

    Optional correlation representation:

        e_t = lambda_ez*zeta_t + eta_t

    Set orth=True to impose lambda_ez = 0.
    """
    pi_t = _as_1d(pi_data)
    pi_tm1 = _as_1d(pi_prev_data)
    pi_expect = _as_1d(Epi_data)
    x_t = _as_1d(x_data)
    x_tm1 = _as_1d(x_prev_data)
    N_obs = _as_1d(N_data)

    T = pi_t.size

    if not (pi_tm1.size == pi_expect.size == x_t.size == x_tm1.size == N_obs.size == T):
        raise ValueError("All input series must have the same length.")

    if T < 3:
        raise ValueError("Need T >= 3 for the AR(2) gap equation.")

    if n_burn < 0:
        raise ValueError("n_burn must be nonnegative.")

    if n_keep <= 0:
        raise ValueError("n_keep must be positive.")

    pri = _common_priors(priors or {})

    _assert_all_pos(
        [
            pri["sigma_alpha"],
            pri["sigma_kappa0"],
            pri["sigma_delta"],
            pri["sigma_phi"],
            pri["sigma_lambda"],
            pri["sigma_rho1"],
            pri["sigma_rho2"],
            pri["sigma_n"],
            pri["a_e"],
            pri["b_e"],
            pri["a_u"],
            pri["b_u"],
            pri["a_eps"],
            pri["b_eps"],
            pri["a_z"],
            pri["b_z"],
            pri["a_N"],
            pri["b_N"],
            pri["P0_Nhat"],
            pri["P0_Nhat_lag"],
            pri["P0_Nbar"],
        ],
        "Prior scales and inverse-gamma hyperparameters must be positive.",
    )

    opts = opts or {}
    rng = np.random.default_rng(_getd(opts, "seed", None))

    alpha = float(_getd(opts, "alpha0", pri["mu_alpha"]))
    kappa0 = float(_getd(opts, "kappa0", pri["mu_kappa0"]))
    delta = float(_getd(opts, "delta0", pri["mu_delta"]))
    phi_1 = float(_getd(opts, "phi10", pri["mu_phi"]))

    lambda_ez = 0.0 if orth else float(_getd(opts, "lambda0", pri["mu_lambda"]))

    rho1 = float(_getd(opts, "rho10", pri["mu_rho1"]))
    rho2 = float(_getd(opts, "rho20", pri["mu_rho2"]))
    n_drift = float(_getd(opts, "n0", pri["mu_n"]))

    sigma_eta2 = float(_getd(opts, "sigma_e20", _getd(opts, "sigma_eta20", 1.0)))
    sigma_zeta2 = float(_getd(opts, "sigma_zeta20", 1.0))
    sigma_u2 = float(_getd(opts, "sigma_u20", 1.0))
    sigma_eps2 = float(_getd(opts, "sigma_eps20", 1.0))
    sigma_N2 = float(_getd(opts, "sigma_N20", _getd(opts, "sigma_m20", 1.0)))

    _assert_all_pos(
        [sigma_eta2, sigma_zeta2, sigma_u2, sigma_eps2, sigma_N2],
        "Initial variances must be positive.",
    )

    # Purely forward-looking restriction: alpha == 0, i.e. no lagged-inflation term.
    # The theoretical Phillips curve has no backward-looking term; it is added in the
    # baseline specification only as reduced-form inertia. Because every inflation series
    # here is a four-quarter change sampled quarterly, pi_t and pi_{t-1} share three of
    # four quarters, so a large alpha is partly a property of that overlap rather than of
    # price setting. This restriction estimates the specification without it.
    no_inertia = bool(_getd(opts, "no_inertia", False))
    enforce_stationary = bool(_getd(opts, "enforce_stationary", True))
    ar2_max_tries = int(max(1, _getd(opts, "ar2_max_tries", 2000)))
    store_every = int(max(1, _getd(opts, "store_every", 1)))
    verbose = bool(_getd(opts, "verbose", False))
    coefficient_constraints = _getd(opts, "coefficient_constraints", {})
    constraint_stats: dict[str, int] = {}
    ar2_stats: dict[str, int] = {}

    # Reduced-run support for Chib's marginal likelihood. ``opts["fixed"]`` pins
    # named blocks to supplied values and skips their draw, so the reduced runs
    # reuse these exact production conditionals instead of a reimplementation.
    # Recognised block names: beta, lambda_ez, phi_1, sigma_zeta2, sigma_eta2,
    # rho, sigma_u2, n, sigma_eps2, sigma_N2.
    fixed = dict(_getd(opts, "fixed", {}) or {})
    unknown_fixed = set(fixed) - {
        "beta", "lambda_ez", "phi_1", "sigma_zeta2", "sigma_eta2",
        "rho", "sigma_u2", "n", "sigma_eps2", "sigma_N2",
    }
    if unknown_fixed:
        raise ValueError(f"Unknown fixed block(s): {sorted(unknown_fixed)}")
    if "beta" in fixed:
        alpha, kappa0, delta = (float(v) for v in fixed["beta"])
    if "lambda_ez" in fixed:
        lambda_ez = float(fixed["lambda_ez"])
    if "phi_1" in fixed:
        phi_1 = float(fixed["phi_1"])
    if "rho" in fixed:
        rho1, rho2 = (float(v) for v in fixed["rho"])
    if "sigma_zeta2" in fixed:
        sigma_zeta2 = float(fixed["sigma_zeta2"])
    if "sigma_eta2" in fixed:
        sigma_eta2 = float(fixed["sigma_eta2"])
    if "sigma_u2" in fixed:
        sigma_u2 = float(fixed["sigma_u2"])
    if "sigma_eps2" in fixed:
        sigma_eps2 = float(fixed["sigma_eps2"])
    if "sigma_N2" in fixed:
        sigma_N2 = float(fixed["sigma_N2"])
    if "n" in fixed:
        n_drift = float(fixed["n"])

    n_store = int(n_keep // store_every)

    if n_store <= 0:
        raise ValueError("No draws would be stored. Use n_keep >= store_every.")

    Nbar, Nhat = _init_states(N_obs)
    a_t = pi_tm1 - pi_expect
    y = pi_t - pi_expect

    lambda_prec0 = 0.0 if orth else 1.0 / pri["sigma_lambda"]**2

    # Initial state prior for Kalman filter:
    #   s_0 = [Nhat_0, Nhat_{-1}, Nbar_0]'
    m0 = np.array(
        [
            float(_getd(opts, "m0_Nhat", pri["m0_Nhat"])),
            float(_getd(opts, "m0_Nhat_lag", pri["m0_Nhat_lag"])),
            float(_getd(opts, "m0_Nbar", pri["m0_Nbar"])),
        ],
        dtype=float,
    )

    P0 = np.diag(
        [
            float(_getd(opts, "P0_Nhat", pri["P0_Nhat"])),
            float(_getd(opts, "P0_Nhat_lag", pri["P0_Nhat_lag"])),
            float(_getd(opts, "P0_Nbar", pri["P0_Nbar"])),
        ]
    )

    states = np.zeros((T, 3), dtype=float)
    states[:, 0] = Nhat
    states[:, 2] = Nbar
    states[0, 1] = m0[1]
    if T > 1:
        states[1:, 1] = Nhat[:-1]

    alpha_draws = np.zeros(n_store)
    kappa0_draws = np.zeros(n_store)
    delta_draws = np.zeros(n_store)
    phi_draws = np.zeros(n_store)
    lambda_draws = np.zeros(n_store)

    rho1_draws = np.zeros(n_store)
    rho2_draws = np.zeros(n_store)
    n_draws = np.zeros(n_store)

    sigma_e_draws = np.zeros(n_store)
    sigma_eta_draws = np.zeros(n_store)
    sigma_zeta_draws = np.zeros(n_store)
    sigma_u_draws = np.zeros(n_store)
    sigma_eps_draws = np.zeros(n_store)
    sigma_N_draws = np.zeros(n_store)

    rho_ez_draws = np.zeros(n_store)

    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))
    kappa_t_draws = np.zeros((n_store, T))
    # Nhat_{-1}, the sampled initial AR(2) lag. Not part of the saved production
    # schema (opt-in via opts["return_state_lag"]) because it would change the
    # shape of every stored posterior; Chib's AR(2) ordinate needs it, since with
    # sigma_u ~ 0.045 substituting the prior mean for it mis-weights the first
    # regression row by hundreds of log points.
    nhat_lag_draws = np.zeros(n_store)

    total_iter = n_burn + n_keep
    store_idx = 0

    for it in range(1, total_iter + 1):
        # ------------------------------------------------------------
        # 1. Draw alpha, kappa0, delta from inflation regression.
        # ------------------------------------------------------------
        zeta = x_t - phi_1 * x_tm1

        y_adj = y - lambda_ez * zeta

        columns = [] if no_inertia else [a_t]
        columns += [x_t / KAPPA_SCALE, (x_t * Nbar) / KAPPA_SCALE]
        X = np.column_stack(columns)

        prior_means = [] if no_inertia else [pri["mu_alpha"]]
        prior_means += [pri["mu_kappa0"], pri["mu_delta"]]
        prior_vars = [] if no_inertia else [pri["sigma_alpha"]**2]
        prior_vars += [pri["sigma_kappa0"]**2, pri["sigma_delta"]**2]
        beta_prior_mean = np.array(prior_means, dtype=float)
        beta_prior_var = np.array(prior_vars, dtype=float)
        beta_names = ("kappa_0", "delta") if no_inertia else ("alpha", "kappa_0", "delta")

        if "beta" not in fixed:
            beta = draw_with_constraints(
                lambda: _sample_beta_gaussian(
                    y=y_adj,
                    X=X,
                    sigma2=sigma_eta2,
                    prior_mean=beta_prior_mean,
                    prior_var=beta_prior_var,
                    rng=rng,
                ),
                beta_names,
                coefficient_constraints,
                validators=_kappa_t_constraint_validators(
                    Nbar, coefficient_constraints, offset=0 if no_inertia else 1
                ),
                stats=constraint_stats,
            )

            if no_inertia:
                alpha = 0.0
                kappa0 = float(beta[0])
                delta = float(beta[1])
            else:
                alpha = float(beta[0])
                kappa0 = float(beta[1])
                delta = float(beta[2])

        kappa_t = kappa0 + delta * Nbar
        kappa_t_eff = kappa_t / KAPPA_SCALE

        # ------------------------------------------------------------
        # 2. Draw lambda_ez if not imposing orthogonality.
        # ------------------------------------------------------------
        if orth:
            lambda_ez = 0.0
        elif "lambda_ez" not in fixed:
            e_base = y - alpha * a_t - kappa_t_eff * x_t

            post_var_lambda = 1.0 / (
                lambda_prec0
                + float(np.sum(zeta**2)) / sigma_eta2
            )

            post_mean_lambda = post_var_lambda * (
                pri["mu_lambda"] * lambda_prec0
                + float(np.dot(zeta, e_base)) / sigma_eta2
            )

            lambda_ez = float(
                post_mean_lambda
                + np.sqrt(post_var_lambda) * rng.standard_normal()
            )

        # ------------------------------------------------------------
        # 3. Draw phi_1.
        # ------------------------------------------------------------
        y_tilde_phi = y - alpha * a_t - kappa_t_eff * x_t

        if "phi_1" not in fixed:
            phi_1 = _sample_phi_joint(
                x_t=x_t,
                x_tm1=x_tm1,
                y_tilde=y_tilde_phi,
                lambda_ez=lambda_ez,
                sigma_zeta2=sigma_zeta2,
                sigma_eta2=sigma_eta2,
                mu_phi=pri["mu_phi"],
                sigma_phi=pri["sigma_phi"],
                rng=rng,
            )

        # ------------------------------------------------------------
        # 4. Draw sigma_zeta2 and sigma_eta2.
        # ------------------------------------------------------------
        zeta = x_t - phi_1 * x_tm1

        eta = (
            y
            - alpha * a_t
            - kappa_t_eff * x_t
            - lambda_ez * zeta
        )

        if "sigma_zeta2" not in fixed:
            sigma_zeta2 = _sample_invgamma(
                pri["a_z"] + 0.5 * T,
                pri["b_z"] + 0.5 * float(np.sum(zeta**2)),
                rng,
            )

        if "sigma_eta2" not in fixed:
            sigma_eta2 = _sample_invgamma(
                pri["a_e"] + 0.5 * T,
                pri["b_e"] + 0.5 * float(np.sum(eta**2)),
                rng,
            )

        # ------------------------------------------------------------
        # 5. Draw rho1, rho2 and sigma_u2.
        # ------------------------------------------------------------
        if "rho" not in fixed:
            rho1, rho2 = _sample_ar2_coeffs(
                Nhat=Nhat,
                sigma_state2=sigma_u2,
                mu_rho1=pri["mu_rho1"],
                sigma_rho1=pri["sigma_rho1"],
                mu_rho2=pri["mu_rho2"],
                sigma_rho2=pri["sigma_rho2"],
                enforce_stationary=enforce_stationary,
                rng=rng,
                max_tries=ar2_max_tries,
                current=(rho1, rho2),
                stats=ar2_stats,
                initial_lag=float(states[0, 1]),
            )

        resid_u = states[1:, 0] - rho1 * states[:-1, 0] - rho2 * states[:-1, 1]

        if "sigma_u2" not in fixed:
            sigma_u2 = _sample_invgamma(
                pri["a_u"] + 0.5 * resid_u.size,
                pri["b_u"] + 0.5 * float(np.sum(resid_u**2)),
                rng,
            )

        # ------------------------------------------------------------
        # 6. Draw n and sigma_eps2.
        # ------------------------------------------------------------
        dNbar = Nbar[1:] - Nbar[:-1]

        post_var_n = 1.0 / (
            1.0 / pri["sigma_n"]**2
            + dNbar.size / sigma_eps2
        )

        post_mean_n = post_var_n * (
            pri["mu_n"] / pri["sigma_n"]**2
            + float(np.sum(dNbar)) / sigma_eps2
        )

        if "n" not in fixed:
            n_drift = float(
                post_mean_n
                + np.sqrt(post_var_n) * rng.standard_normal()
            )

        resid_eps = Nbar[1:] - n_drift - Nbar[:-1]

        if "sigma_eps2" not in fixed:
            sigma_eps2 = _sample_invgamma(
                pri["a_eps"] + 0.5 * resid_eps.size,
                pri["b_eps"] + 0.5 * float(np.sum(resid_eps**2)),
                rng,
            )

        # ------------------------------------------------------------
        # 7. Draw measurement variance sigma_N2.
        # ------------------------------------------------------------
        resid_N = finite_N_residuals(N_obs, Nhat, Nbar)

        if "sigma_N2" not in fixed:
            sigma_N2 = _sample_invgamma(
                pri["a_N"] + 0.5 * resid_N.size,
                pri["b_N"] + 0.5 * float(np.sum(resid_N**2)),
                rng,
            )

        # ------------------------------------------------------------
        # 8. Draw latent states jointly by Kalman filter + FFBS.
        # ------------------------------------------------------------
        obs_offset = lambda_ez * zeta

        Nbar, Nhat, states = _sample_states_kalman_ffbs(
            N_obs=N_obs,
            pi_t=pi_t,
            pi_tm1=pi_tm1,
            pi_expect=pi_expect,
            x_t=x_t,
            alpha=alpha,
            kappa0=kappa0,
            delta=delta,
            n_drift=n_drift,
            rho1=rho1,
            rho2=rho2,
            sigma_eta2=sigma_eta2,
            sigma_u2=sigma_u2,
            sigma_eps2=sigma_eps2,
            sigma_N2=sigma_N2,
            obs_offset=obs_offset,
            m0=m0,
            P0=P0,
            rng=rng,
        )

        kappa_t = kappa0 + delta * Nbar

        # ------------------------------------------------------------
        # 9. Store posterior draws.
        # ------------------------------------------------------------
        sigma_e = float(np.sqrt(lambda_ez**2 * sigma_zeta2 + sigma_eta2))

        rho_ez = (
            0.0
            if orth
            else float(
                (lambda_ez * np.sqrt(sigma_zeta2))
                / max(sigma_e, 1e-12)
            )
        )

        if it > n_burn and (it - n_burn) % store_every == 0:
            alpha_draws[store_idx] = alpha
            kappa0_draws[store_idx] = kappa0 / KAPPA_SCALE
            delta_draws[store_idx] = delta / KAPPA_SCALE
            phi_draws[store_idx] = phi_1
            lambda_draws[store_idx] = lambda_ez

            rho1_draws[store_idx] = rho1
            rho2_draws[store_idx] = rho2
            n_draws[store_idx] = n_drift

            sigma_e_draws[store_idx] = sigma_e
            sigma_eta_draws[store_idx] = np.sqrt(sigma_eta2)
            sigma_zeta_draws[store_idx] = np.sqrt(sigma_zeta2)
            sigma_u_draws[store_idx] = np.sqrt(sigma_u2)
            sigma_eps_draws[store_idx] = np.sqrt(sigma_eps2)
            sigma_N_draws[store_idx] = np.sqrt(sigma_N2)

            rho_ez_draws[store_idx] = rho_ez

            Nbar_draws[store_idx] = Nbar
            Nhat_draws[store_idx] = Nhat
            nhat_lag_draws[store_idx] = float(states[0, 1])
            kappa_t_draws[store_idx] = kappa_t / KAPPA_SCALE

            store_idx += 1

        if verbose and it % 5000 == 0:
            print(
                f"Iter {it}/{total_iter}: "
                f"alpha={alpha:.3f}, "
                f"kappa0={kappa0:.3f}, "
                f"delta={delta:.3f}, "
                f"rho1={rho1:.3f}, "
                f"rho2={rho2:.3f}, "
                f"n={n_drift:.3f}, "
                f"sigma_N={np.sqrt(sigma_N2):.3f}"
            )

    return {
        "alpha": _summary(alpha_draws),
        "kappa_0": _summary(kappa0_draws),
        "delta": _summary(delta_draws),
        "phi_1": _summary(phi_draws),
        "lambda_ez": _summary(lambda_draws),
        "rho": _summary(rho_ez_draws),
        "rho1": _summary(rho1_draws),
        "rho2": _summary(rho2_draws),
        "n": _summary(n_draws),
        "sigma_e": _summary(sigma_e_draws),
        "sigma_eta": _summary(sigma_eta_draws),
        "sigma_zeta": _summary(sigma_zeta_draws),
        "sigma_u": _summary(sigma_u_draws),
        "sigma_eps": _summary(sigma_eps_draws),
        "sigma_N": _summary(sigma_N_draws),
        "state_draws": {
            "Nbar": Nbar_draws,
            "Nhat": Nhat_draws,
            "kappa_t": kappa_t_draws,
            **(
                {"Nhat_lag": nhat_lag_draws}
                if bool(_getd(opts, "return_state_lag", False))
                else {}
            ),
        },
        "priors": priors or {},
        "opts": opts,
        "model": {
            "N_measurement_error": True,
            "N_measurement_equation": "N_obs_t = Nhat_t + Nbar_t + measurement_error_t",
            "state_sampler": "joint_ffbs",
            "theta_sampled": False,
            "inflation_equation": (
                "y_t = kappa_t*x_t + lambda_ez*zeta_t + eta_t  (alpha restricted to 0)"
                if no_inertia
                else "y_t = alpha*a_t + kappa_t*x_t + lambda_ez*zeta_t + eta_t"
            ),
            "no_inertia": no_inertia,
            "state_vector": "[Nhat_t, Nhat_{t-1}, Nbar_t]'",
            "kappa_scale": KAPPA_SCALE,
            "kappa_internal": "stored kappa_0, delta, and kappa_t multiplied by KAPPA_SCALE",
            "stored_units": "physical",
            "coefficient_constraints": coefficient_constraints,
            "coefficient_constraint_stats": constraint_stats_summary(constraint_stats),
            "ar2_stationarity": {
                "enforce_stationary": enforce_stationary,
                "max_tries": ar2_max_tries,
                **_ar2_stats_summary(ar2_stats),
            },
        },
    }


def func_nkpc_hsa_decomp_tv_kappa_noerror(*args, **kwargs) -> dict[str, Any]:
    """Backward-compatible alias for the current steady HSA sampler."""
    return func_nkpc_hsa_decomp_tv_kappa_kalman(*args, **kwargs)


def func_nkpc_hsa_decomp_tv_theta_kappa(*args, **kwargs) -> dict[str, Any]:
    raise NotImplementedError(
        "func_nkpc_hsa_decomp_tv_theta_kappa is not available in the current "
        "func_gibbs.hsa_steady implementation."
    )

__all__ = [
    "func_nkpc_hsa_decomp_tv_kappa_noerror",
    "func_nkpc_hsa_decomp_tv_kappa_kalman",
    "func_nkpc_hsa_decomp_tv_theta_kappa",
]
