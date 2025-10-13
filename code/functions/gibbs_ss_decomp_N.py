import numpy as np
from numpy.linalg import inv, det, eig
from dataclasses import dataclass
from typing import Dict, Any
from tqdm import trange

# ------------------------------
# Utility: placeholder for Chib (1995) marginal likelihood
# ------------------------------
@dataclass
class ChibResult:
    log_ml: float

# ------------------------------
# FFBS for AR(2) state (cycle)
# ------------------------------

def sample_ar2_states_ffbs(y_target: np.ndarray, rho1: float, rho2: float, sigma_eps2: float,
                            pi_t: np.ndarray, alpha: float, pi_tm1: np.ndarray, E_pi_tp1: np.ndarray,
                            kappa: float, x_t: np.ndarray, theta: float, sigma_v2: float) -> np.ndarray:
    T = len(y_target)
    if T < 3:
        return y_target.copy()

    # Transition and covariance
    F = np.array([[rho1, rho2],
                  [1.0,  0.0]])
    Q = np.array([[sigma_eps2, 0.0],
                  [0.0,        0.0]])

    # Storage
    m = np.zeros((2, T))        # filtered means
    P = np.zeros((2, 2, T))     # filtered covariances
    m_pred = np.zeros((2, T))
    P_pred = np.zeros((2, 2, T))

    # Diffuse-ish prior for initial state
    m[:, 0] = np.array([y_target[0], 0.0])
    P[:, :, 0] = np.eye(2) * 10.0

    for t in range(1, T):
        if t >= 2:
            m_pred[:, t] = F @ m[:, t-1]
            P_pred[:, :, t] = F @ P[:, :, t-1] @ F.T + Q
        else:
            m_pred[:, t] = m[:, t-1]
            P_pred[:, :, t] = P[:, :, t-1]

        # Target observation: Nhat_t ≈ y_target[t]
        H_target = np.array([[1.0, 0.0]])
        R_target = np.array([[sigma_eps2 * 0.1]])

        # NKPC observation: theta * Nhat_t = alpha*pi_{t-1} + (1-alpha)*Epi_{t+1} + kappa*x_t - pi_t + v_t
        nkpc_obs = alpha * pi_tm1[t] + (1.0 - alpha) * E_pi_tp1[t] + kappa * x_t[t] - pi_t[t]
        H_nkpc = np.array([[theta, 0.0]])
        R_nkpc = np.array([[sigma_v2]])

        # Combine observations
        H_comb = np.vstack([H_target, H_nkpc])     # (2 x 2)
        y_comb = np.array([y_target[t], nkpc_obs]) # (2,)
        R_comb = np.diagflat([R_target.item(), R_nkpc.item()])

        S = H_comb @ P_pred[:, :, t] @ H_comb.T + R_comb
        K = P_pred[:, :, t] @ H_comb.T @ inv(S)
        innov = y_comb - (H_comb @ m_pred[:, t])
        m[:, t] = m_pred[:, t] + K @ innov
        P[:, :, t] = P_pred[:, :, t] - K @ H_comb @ P_pred[:, :, t]

    # Backward sampling
    Nhat_states = np.zeros((2, T))
    # Sample last state
    P_T = P[:, :, T-1]
    # ensure PD
    vals, vecs = eig(P_T)
    P_T = vecs @ np.diag(np.maximum(vals.real, 1e-10)) @ vecs.T
    Nhat_states[:, T-1] = np.random.multivariate_normal(mean=m[:, T-1], cov=P_T)

    for t in range(T-2, -1, -1):
        if t >= 1:
            Ppred_next = P_pred[:, :, t+1]
            A = P[:, :, t] @ F.T @ inv(Ppred_next)
            m_smooth = m[:, t] + A @ (Nhat_states[:, t+1] - m_pred[:, t+1])
            P_smooth = P[:, :, t] - A @ (Ppred_next - P[:, :, t]) @ A.T
            vals, vecs = eig(P_smooth)
            P_smooth = vecs @ np.diag(np.maximum(vals.real, 1e-10)) @ vecs.T
            Nhat_states[:, t] = np.random.multivariate_normal(mean=m_smooth, cov=P_smooth)
        else:
            Nhat_states[:, t] = Nhat_states[:, t+1]

    return Nhat_states[0, :].copy()


# ------------------------------
# FFBS for Random Walk with drift (trend)
# ------------------------------

def sample_rw_states_ffbs(y_target: np.ndarray, n: float, sigma_eta2: float) -> np.ndarray:
    T = len(y_target)
    if T < 2:
        return y_target.copy()

    m = np.zeros(T)
    P = np.zeros(T)

    # Initial diffuse prior
    m[0] = y_target[0]
    P[0] = 10.0

    for t in range(1, T):
        # Predict
        m_pred = n + m[t-1]
        P_pred = P[t-1] + sigma_eta2
        # Observation variance (small)
        R_obs = sigma_eta2 * 0.1
        K = P_pred / (P_pred + R_obs)
        m[t] = m_pred + K * (y_target[t] - m_pred)
        P[t] = P_pred * (1.0 - K)

    # Backward sampling
    Nbar_new = np.zeros(T)
    Nbar_new[T-1] = m[T-1] + np.sqrt(max(P[T-1], 1e-10)) * np.random.randn()
    for t in range(T-2, -1, -1):
        A = P[t] / (P[t] + sigma_eta2)
        m_smooth = m[t] + A * (Nbar_new[t+1] - n - m[t])
        P_smooth = P[t] * (1.0 - A)
        Nbar_new[t] = m_smooth + np.sqrt(max(P_smooth, 1e-10)) * np.random.randn()

    return Nbar_new


# ------------------------------
# Main estimation function
# ------------------------------
def estimate_hsa(pi_data: np.ndarray, pi_prev_data: np.ndarray, Epi_data: np.ndarray,
                 x_data: np.ndarray, N_data: np.ndarray,
                 n_burn: int, n_keep: int, seed: int | None = None,
                 priors: dict | None = None) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)
    # Coerce to NumPy arrays to support pandas Series/DataFrames safely
    pi_t = np.ravel(np.asarray(pi_data, dtype=float))
    pi_tm1 = np.ravel(np.asarray(pi_prev_data, dtype=float))
    E_pi_tp1 = np.ravel(np.asarray(Epi_data, dtype=float))
    x_t = np.ravel(np.asarray(x_data, dtype=float))
    N_data = np.ravel(np.asarray(N_data, dtype=float))
    T = len(pi_t)

    # --- default priors ---
    default_priors = {
        "mu_alpha": 0.5, "sigma_alpha": 0.1,
        "mu_kappa": 0.0, "sigma_kappa": 0.1,
        "mu_theta": 0.0, "sigma_theta": 0.1,
        "mu_rho1": 0.5, "sigma_rho1": 0.1,
        "mu_rho2": -0.5, "sigma_rho2": 0.1,
        "mu_n": 0.0, "sigma_n": 0.05,
        "a_v": 0.001, "b_v": 0.001,
        "a_eps": 0.001, "b_eps": 0.001,
        "a_eta": 0.001, "b_eta": 0.001,
    }
    if priors is None:
        priors = default_priors
    else:
        priors = {**default_priors, **priors}

    # --- priors ---
    mu_alpha, sigma_alpha = priors["mu_alpha"], priors["sigma_alpha"]
    mu_kappa, sigma_kappa = priors["mu_kappa"], priors["sigma_kappa"]
    mu_theta, sigma_theta = priors["mu_theta"], priors["sigma_theta"]
    mu_rho1, sigma_rho1 = priors["mu_rho1"], priors["sigma_rho1"]
    mu_rho2, sigma_rho2 = priors["mu_rho2"], priors["sigma_rho2"]
    mu_n, sigma_n = priors["mu_n"], priors["sigma_n"]

    a_v, b_v = priors["a_v"], priors["b_v"]
    a_eps, b_eps = priors["a_eps"], priors["b_eps"]
    a_eta, b_eta = priors["a_eta"], priors["b_eta"]

    # Initial params
    alpha = 0.6
    kappa = 0.3
    theta = 0.5
    rho1 = 0.5
    rho2 = -0.5
    n = 0.01
    sigma_v2 = 1.0
    sigma_eps2 = 0.5
    sigma_eta2 = 0.1

    # Initial states via simple smoothing of N_data
    Nbar = np.zeros(T)
    Nbar[0] = N_data[0]
    if T > 1:
        Nbar[1] = N_data[1]
    for t in range(2, T):
        Nbar[t] = 0.7 * Nbar[t-1] + 0.3 * N_data[t]
    Nhat = N_data - Nbar

    # Storage
    alpha_draws = np.zeros(n_keep)
    kappa_draws = np.zeros(n_keep)
    theta_draws = np.zeros(n_keep)
    rho1_draws = np.zeros(n_keep)
    rho2_draws = np.zeros(n_keep)
    n_draws = np.zeros(n_keep)
    sigma_v2_draws = np.zeros(n_keep)
    sigma_eps2_draws = np.zeros(n_keep)
    sigma_eta2_draws = np.zeros(n_keep)

    # Store some state draws (every ~10th)
    n_store = max(1, n_keep // 10)
    Nbar_draws = np.zeros((n_store, T))
    Nhat_draws = np.zeros((n_store, T))

    print(f"Starting Gibbs sampling: burn-in={n_burn}, keep={n_keep}")
    for iter_ in trange(1, n_burn + n_keep + 1, desc="Gibbs sampling"):
        # ---- Sample alpha ----
        y_alpha = pi_t - E_pi_tp1 - kappa * x_t + theta * Nhat
        X_alpha = pi_tm1 - E_pi_tp1
        XtX = X_alpha @ X_alpha
        if abs(XtX) > 1e-12:
            prior_prec = 1.0 / (sigma_alpha ** 2)
            data_prec = XtX / sigma_v2
            post_prec = prior_prec + data_prec
            post_var = 1.0 / post_prec
            post_mean = post_var * (prior_prec * mu_alpha + (X_alpha @ y_alpha) / sigma_v2)
            alpha = post_mean + np.sqrt(post_var) * np.random.randn()

        # ---- Sample kappa ----
        y_kappa = pi_t - alpha * pi_tm1 - (1.0 - alpha) * E_pi_tp1 + theta * Nhat
        X_kappa = x_t
        XtX = X_kappa @ X_kappa
        if abs(XtX) > 1e-12:
            prior_prec_k = 1.0 / (sigma_kappa ** 2)
            data_prec_k = XtX / sigma_v2
            post_prec_k = prior_prec_k + data_prec_k
            post_var_k = 1.0 / post_prec_k
            post_mean_k = post_var_k * (prior_prec_k * mu_kappa + (X_kappa @ y_kappa) / sigma_v2)
            kappa = post_mean_k + np.sqrt(post_var_k) * np.random.randn()

        # ---- Sample theta ----
        y_theta = pi_t - alpha * pi_tm1 - (1.0 - alpha) * E_pi_tp1 - kappa * x_t
        X_theta = -Nhat
        XtX = X_theta @ X_theta
        if abs(XtX) > 1e-12:
            prior_prec_t = 1.0 / (sigma_theta ** 2)
            data_prec_t = XtX / sigma_v2
            post_prec_t = prior_prec_t + data_prec_t
            post_var_t = 1.0 / post_prec_t
            post_mean_t = post_var_t * (prior_prec_t * mu_theta + (X_theta @ y_theta) / sigma_v2)
            theta = post_mean_t + np.sqrt(post_var_t) * np.random.randn()

        # ---- Sample rho1, rho2 (AR(2)) ----
        if T >= 3:
            y_rho = Nhat[2:]
            X_rho = np.column_stack([Nhat[1:-1], Nhat[:-2]])
            XtX = X_rho.T @ X_rho
            if X_rho.shape[0] > 0 and abs(det(XtX)) > 1e-12:
                prior_prec_rho = np.diag([1.0 / (sigma_rho1 ** 2), 1.0 / (sigma_rho2 ** 2)])
                data_prec_rho = XtX / sigma_eps2
                post_prec_rho = prior_prec_rho + data_prec_rho
                post_cov_rho = inv(post_prec_rho)
                prior_mean_rho = np.array([mu_rho1, mu_rho2])
                post_mean_rho = post_cov_rho @ (prior_prec_rho @ prior_mean_rho + (X_rho.T @ y_rho) / sigma_eps2)

                # Draw with stationarity check
                max_tries = 2000
                ok = False
                for _ in range(max_tries):
                    rho_draw = np.random.multivariate_normal(mean=post_mean_rho, cov=post_cov_rho)
                    r1, r2 = rho_draw
                    # Simple sufficient (not necessary) conditions used in MATLAB code
                    if (abs(r2) < 1.0) and ((r1 + r2) < 1.0) and ((r2 - r1) < 1.0):
                        rho1, rho2 = float(r1), float(r2)
                        ok = True
                        break
                if not ok:
                    rho1, rho2 = float(post_mean_rho[0]), float(post_mean_rho[1])

        # ---- Sample n (drift) ----
        if T >= 2:
            y_n = Nbar[1:] - Nbar[:-1]
            T_n = len(y_n)
            prior_prec_n = 1.0 / (sigma_n ** 2)
            data_prec_n = T_n / sigma_eta2
            post_prec_n = prior_prec_n + data_prec_n
            post_var_n = 1.0 / post_prec_n
            post_mean_n = post_var_n * (prior_prec_n * mu_n + y_n.sum() / sigma_eta2)
            n = post_mean_n + np.sqrt(post_var_n) * np.random.randn()

        # ---- Sample variances (Inverse-Gamma via 1/Gamma) ----
        nkpc_resid = pi_t - alpha * pi_tm1 - (1.0 - alpha) * E_pi_tp1 - kappa * x_t + theta * Nhat
        a_post_v = a_v + T / 2.0
        b_post_v = b_v + 0.5 * np.sum(nkpc_resid ** 2)
        sigma_v2 = 1.0 / np.random.gamma(shape=a_post_v, scale=1.0 / b_post_v)

        if T >= 3:
            ar_resid = Nhat[2:] - rho1 * Nhat[1:-1] - rho2 * Nhat[:-2]
            a_post_eps = a_eps + len(ar_resid) / 2.0
            b_post_eps = b_eps + 0.5 * np.sum(ar_resid ** 2)
            sigma_eps2 = 1.0 / np.random.gamma(shape=a_post_eps, scale=1.0 / b_post_eps)

        if T >= 2:
            rw_resid = Nbar[1:] - n - Nbar[:-1]
            a_post_eta = a_eta + len(rw_resid) / 2.0
            b_post_eta = b_eta + 0.5 * np.sum(rw_resid ** 2)
            sigma_eta2 = 1.0 / np.random.gamma(shape=a_post_eta, scale=1.0 / b_post_eta)

        # ---- FFBS for states ----
        Nhat = sample_ar2_states_ffbs(N_data - Nbar, rho1, rho2, sigma_eps2,
                                      pi_t, alpha, pi_tm1, E_pi_tp1, kappa, x_t, theta, sigma_v2)
        Nbar = sample_rw_states_ffbs(N_data - Nhat, n, sigma_eta2)

        # ---- Store draws ----
        if iter_ > n_burn:
            idx = iter_ - n_burn - 1
            alpha_draws[idx] = alpha
            kappa_draws[idx] = kappa
            theta_draws[idx] = theta
            rho1_draws[idx] = rho1
            rho2_draws[idx] = rho2
            n_draws[idx] = n
            sigma_v2_draws[idx] = sigma_v2
            sigma_eps2_draws[idx] = sigma_eps2
            sigma_eta2_draws[idx] = sigma_eta2

            if n_store > 0 and (idx % 10 == 0) and (idx // 10 < n_store):
                store_idx = idx // 10
                Nbar_draws[store_idx, :] = Nbar
                Nhat_draws[store_idx, :] = Nhat

    def summarize(draws: np.ndarray) -> Dict[str, Any]:
        return {
            "draws": draws,
            "mean": float(np.mean(draws)),
            "std": float(np.std(draws, ddof=0)),
            "quantiles": np.quantile(draws, [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975])
        }

    results: Dict[str, Any] = {}
    # Parameters
    results["alpha"] = summarize(alpha_draws)
    results["kappa"] = summarize(kappa_draws)
    results["theta"] = summarize(theta_draws)
    results["rho1"] = summarize(rho1_draws)
    results["rho2"] = summarize(rho2_draws)
    results["n"] = summarize(n_draws)
    results["sigma_v2"] = summarize(sigma_v2_draws)
    results["sigma_eps2"] = summarize(sigma_eps2_draws)
    results["sigma_eta2"] = summarize(sigma_eta2_draws)

    # States
    if n_store > 0:
        Nbar_mean = np.mean(Nbar_draws[:min(n_store, n_keep // 10 if n_keep // 10 > 0 else n_store), :], axis=0)
        Nhat_mean = np.mean(Nhat_draws[:min(n_store, n_keep // 10 if n_keep // 10 > 0 else n_store), :], axis=0)
    else:
        Nbar_mean = Nbar
        Nhat_mean = Nhat
    results["states"] = {
        "Nbar_mean": Nbar_mean,
        "Nhat_mean": Nhat_mean,
        "N_mean": Nbar_mean + Nhat_mean
    }
    # Diagnostics
    results["diagnostics"] = {"T": int(T), "n_params": 9}
    return results
