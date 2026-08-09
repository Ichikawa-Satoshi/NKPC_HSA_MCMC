"""Validation for the MA(3) error-structure robustness package.

The package is additive: nothing in ``nkpc_hsa.gibbs`` is modified, so the
central property to pin is that the MA machinery *collapses to production* when
``psi = 0``. Four groups of tests, in order of what they protect:

1. Banded algebra. ``Omega_0(psi)`` operations must agree with a dense Toeplitz
   reference to machine precision, and reduce to the identity at ``psi = 0``.
2. Nesting. ``func_nkpc_ces_ma3(ma_order=0)`` must reproduce
   ``func_nkpc_ces`` bit for bit -- same RNG stream, same algebra. The
   augmented FFBS and the hsa_steady sampler cannot match draw-for-draw (a
   higher-dimensional multivariate normal consumes a different number of random
   numbers), so those are pinned distributionally instead.
3. Identification. Known ``psi`` must be recovered from synthetic data.
4. Guards. Non-invertible ``psi`` must be refused rather than silently used.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal

from nkpc_hsa.error_robustness import ma_error as M
from nkpc_hsa.error_robustness.ces_ma3 import func_nkpc_ces_ma3
from nkpc_hsa.error_robustness.joint_ffbs_ma3 import sample_joint_states_ffbs_ma
from nkpc_hsa.gibbs.common.joint_ffbs import sample_joint_competition_states_ffbs
from nkpc_hsa.gibbs.ces.model import func_nkpc_ces

PSI = np.array([0.42, 0.32, 0.59])
CES_PRIORS = dict(
    mu_alpha=0.5, sigma_alpha=0.2, mu_kappa=0.1, sigma_kappa=0.2,
    mu_phi_1=0.7, sigma_phi_1=0.2, mu_lambda=0.0, sigma_lambda=0.5,
    a_e=2.0, b_e=2.0, a_z=0.001, b_z=0.001,
)


def _dense_omega(psi: np.ndarray, n: int) -> np.ndarray:
    gamma = M.autocovariance(psi, 1.0)
    return toeplitz(np.r_[gamma, np.zeros(max(0, n - gamma.size))][:n])


def _ces_data(seed: int = 0, T: int = 124, psi: np.ndarray | None = None):
    """CES-shaped synthetic data; ``psi`` set means the disturbance is MA."""
    rng = np.random.default_rng(seed)
    x = np.zeros(T + 1)
    for t in range(1, T + 1):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    x_t, x_tm1 = x[1:], x[:-1]
    pi_expect = 2.0 + 0.3 * rng.standard_normal(T)
    pi_prev = 2.0 + rng.standard_normal(T)
    if psi is None:
        disturbance = 0.8 * rng.standard_normal(T)
    else:
        v = rng.standard_normal(T + psi.size) * 0.8
        q = psi.size
        disturbance = np.array(
            [v[t + q] + sum(psi[j] * v[t + q - 1 - j] for j in range(q)) for t in range(T)]
        )
    pi_t = pi_expect + 0.6 * (pi_prev - pi_expect) + 0.15 * x_t + disturbance
    return pi_t, pi_prev, pi_expect, x_t, x_tm1


# ------------------------------------------------------------------
# 1. Banded algebra
# ------------------------------------------------------------------

def test_banded_operations_match_dense_reference():
    n, sigma2 = 40, 1.7
    rng = np.random.default_rng(0)
    r = rng.standard_normal(n) * 2.0
    X = rng.standard_normal((n, 2))

    weighting = M.MAWeighting(PSI, n)
    omega = _dense_omega(PSI, n)

    assert weighting.log_det == pytest.approx(np.linalg.slogdet(omega)[1], abs=1e-10)
    assert weighting.quadratic_form(r) == pytest.approx(r @ np.linalg.solve(omega, r), abs=1e-9)
    assert weighting.log_likelihood(r, sigma2) == pytest.approx(
        multivariate_normal(np.zeros(n), sigma2 * omega).logpdf(r), abs=1e-9
    )

    whitened = weighting.whiten(r)
    assert whitened @ whitened == pytest.approx(r @ np.linalg.solve(omega, r), abs=1e-9)

    XtWX, XtWy = weighting.gls_moments(r, X)
    assert XtWX == pytest.approx(X.T @ np.linalg.solve(omega, X), abs=1e-9)
    assert XtWy == pytest.approx(X.T @ np.linalg.solve(omega, r), abs=1e-9)


def test_psi_zero_gives_identity_weighting():
    n, sigma2 = 30, 1.3
    rng = np.random.default_rng(1)
    r = rng.standard_normal(n)
    weighting = M.MAWeighting(np.zeros(3), n)

    assert weighting.log_det == pytest.approx(0.0, abs=1e-12)
    assert weighting.solve(np.eye(n)) == pytest.approx(np.eye(n), abs=1e-12)
    assert weighting.log_likelihood(r, sigma2) == pytest.approx(
        multivariate_normal(np.zeros(n), sigma2 * np.eye(n)).logpdf(r), abs=1e-9
    )


def test_state_augmentation_reproduces_ma_autocovariance():
    sigma_v2 = 1.7
    F_v, _Q_v, h_v, P0_v = M.state_augmentation(PSI, sigma_v2)
    implied = [
        float(h_v @ np.linalg.matrix_power(F_v, k) @ P0_v @ h_v) for k in range(PSI.size + 1)
    ]
    assert implied == pytest.approx(M.autocovariance(PSI, sigma_v2), abs=1e-12)


# ------------------------------------------------------------------
# 2. Nesting: psi = 0 must reproduce production
# ------------------------------------------------------------------

def test_ces_ma_order_zero_reproduces_production():
    """Bit-for-bit against ``gibbs.ces.model`` -- the sampler production dispatches to.

    Not ``gibbs.gibbs_ces``: that module is deprecated and, crucially, does not
    apply ``KAPPA_SCALE``. Building the MA(3) CES sampler from it made every
    kappa a factor of 100 out relative to what ``run_model`` and Chib pass and
    read, which is invisible in a self-contained recovery test and fatal in the
    reduced runs.
    """
    data = _ces_data(seed=0)
    opts = dict(seed=42)
    produced = func_nkpc_ces(*data, 1000, 4000, priors=CES_PRIORS, opts=opts)
    nested = func_nkpc_ces_ma3(*data, 1000, 4000, priors=CES_PRIORS, opts=dict(opts, ma_order=0))

    for key in ("alpha", "kappa", "phi_1", "lambda_ez", "sigma_zeta2", "sigma_e2", "rho"):
        assert nested[key]["draws"] == pytest.approx(produced[key]["draws"], abs=1e-12), key


def test_augmented_ffbs_matches_production_distributionally():
    """No MA block (psi = []) leaves the (Nhat, Nbar) smoothing posterior unchanged.

    Draw-for-draw equality is impossible -- the augmented state is one dimension
    wider, so ``multivariate_normal`` consumes a different number of standard
    normals -- so this compares sampling moments against Monte Carlo error.
    """
    rng = np.random.default_rng(3)
    T = 40
    rho1, rho2, n_drift = 1.2, -0.35, 0.02
    sigma_u2, sigma_eps2, sigma_N2, sigma_v2 = 0.02, 0.01, 0.005, 0.30

    Nhat_path = np.zeros(T + 2)
    Nbar_path = np.zeros(T + 1)
    for t in range(2, T + 2):
        Nhat_path[t] = rho1 * Nhat_path[t - 1] + rho2 * Nhat_path[t - 2] + np.sqrt(sigma_u2) * rng.standard_normal()
    for t in range(1, T + 1):
        Nbar_path[t] = n_drift + Nbar_path[t - 1] + np.sqrt(sigma_eps2) * rng.standard_normal()
    Nhat_true, Nbar_true = Nhat_path[2:], Nbar_path[1:]

    N_obs = Nhat_true + Nbar_true + np.sqrt(sigma_N2) * rng.standard_normal(T)
    N_obs[np.arange(T) % 4 != 3] = np.nan  # annual-Q4 style missingness
    h_nbar = 0.05 * rng.standard_normal(T)
    h_nhat = np.zeros(T)
    y_tilde = h_nhat * Nhat_true + h_nbar * Nbar_true + np.sqrt(sigma_v2) * rng.standard_normal(T)

    m0 = np.array([0.0, 0.0, float(N_obs[3])])
    P0 = np.diag([0.5, 0.5, 2.0])
    shared = dict(
        N_obs=N_obs, y_tilde=y_tilde, h_nhat=h_nhat, h_nbar=h_nbar, n_drift=n_drift,
        rho1=rho1, rho2=rho2, sigma_u2=sigma_u2, sigma_eps2=sigma_eps2,
        sigma_N2=sigma_N2, m0=m0, P0=P0,
    )

    reps = 600
    prod_draws = np.zeros((reps, T, 2))
    aug_draws = np.zeros((reps, T, 2))
    rng_prod = np.random.default_rng(101)
    rng_aug = np.random.default_rng(202)
    for i in range(reps):
        nbar, nhat, _ = sample_joint_competition_states_ffbs(
            sigma_eta2=sigma_v2, rng=rng_prod, **shared
        )
        prod_draws[i, :, 0], prod_draws[i, :, 1] = nbar, nhat
        nbar, nhat, _, _ = sample_joint_states_ffbs_ma(
            psi=np.zeros(0), sigma_v2=sigma_v2, rng=rng_aug, **shared
        )
        aug_draws[i, :, 0], aug_draws[i, :, 1] = nbar, nhat

    for j in range(2):
        mc_se = prod_draws[:, :, j].std(axis=0) / np.sqrt(reps)
        gap = np.abs(prod_draws[:, :, j].mean(axis=0) - aug_draws[:, :, j].mean(axis=0))
        assert np.all(gap < 5.0 * mc_se), f"component {j}: max gap {gap.max():.5f}"


# ------------------------------------------------------------------
# 3. Identification
# ------------------------------------------------------------------

def test_ces_ma3_recovers_known_psi():
    data = _ces_data(seed=5, psi=PSI)
    result = func_nkpc_ces_ma3(
        *data, 2000, 6000, priors=CES_PRIORS, opts=dict(seed=9, ma_order=3)
    )
    draws = result["psi"]["draws"]
    for j in range(PSI.size):
        lo, hi = np.quantile(draws[:, j], [0.025, 0.975])
        assert lo < PSI[j] < hi, f"psi_{j + 1}={PSI[j]} outside [{lo:.3f}, {hi:.3f}]"

    accept = result["error_structure"]["psi_acceptance_rate"]
    assert 0.1 < accept < 0.6, f"random-walk acceptance {accept:.3f} is off target"


def test_ma3_removes_the_lagged_dependent_variable_bias():
    """The motivating claim: with a recursive NKPC, i.i.d. overstates alpha.

    Overlapping year-over-year inflation puts ``eta_{t-1..t-3}`` in both the
    disturbance and the ``pi_{t-1}`` regressor. Ignoring it biases the inertia
    coefficient up and attenuates the slope; modelling it should not.
    """
    true_alpha, true_kappa, sigma_v = 0.60, 0.15, 0.8
    T = 124
    rng = np.random.default_rng(1234)

    x = np.zeros(T + 1)
    for t in range(1, T + 1):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    x_t, x_tm1 = x[1:], x[:-1]
    pi_expect = 2.0 + 0.3 * rng.standard_normal(T)
    v = rng.standard_normal(T + 3) * sigma_v
    xi = np.array([v[t + 3] + PSI[0] * v[t + 2] + PSI[1] * v[t + 1] + PSI[2] * v[t] for t in range(T)])

    pi = np.zeros(T + 1)
    pi[0] = 2.0
    for t in range(T):
        pi[t + 1] = true_alpha * pi[t] + (1 - true_alpha) * pi_expect[t] + true_kappa * x_t[t] + xi[t]
    pi_t, pi_tm1 = pi[1:], pi[:-1]

    args = (pi_t, pi_tm1, pi_expect, x_t, x_tm1, 1500, 5000)
    iid = func_nkpc_ces(*args, priors=CES_PRIORS, opts=dict(seed=77))
    ma3 = func_nkpc_ces_ma3(*args, priors=CES_PRIORS, opts=dict(seed=77, ma_order=3))

    iid_bias = abs(iid["alpha"]["draws"].mean() - true_alpha)
    ma3_bias = abs(ma3["alpha"]["draws"].mean() - true_alpha)
    assert iid_bias > 0.10, f"expected a large i.i.d. alpha bias, got {iid_bias:.3f}"
    assert ma3_bias < iid_bias / 2.0, f"MA(3) bias {ma3_bias:.3f} vs i.i.d. {iid_bias:.3f}"


# ------------------------------------------------------------------
# 4. Guards
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "psi, expected",
    [
        (np.zeros(3), True),          # the i.i.d. baseline is nested
        (PSI, True),
        (np.array([1.0, 1.0, 1.0]), False),   # equal-weight overlap: roots -1, +/-i
        (np.array([2.0, 0.0, 0.0]), False),
        (np.array([np.nan, 0.0, 0.0]), False),
    ],
)
def test_invertibility_classification(psi, expected):
    assert M.is_invertible(psi) is expected


def test_non_invertible_psi_is_refused():
    with pytest.raises(ValueError, match="not invertible"):
        M.MAWeighting(np.array([1.0, 1.0, 1.0]), 32)


def test_psi_prior_rejects_outside_invertible_region():
    prior = M.PsiPrior()
    assert prior.log_pdf(np.array([1.0, 1.0, 1.0])) == -np.inf
    assert np.isfinite(prior.log_pdf(PSI))


def test_conditional_psi_ordinate_is_seed_stable():
    """The ordinate feeds a Bayes factor, so its Monte Carlo error has to be small."""
    T = 124
    rng = np.random.default_rng(7)
    v = rng.standard_normal(T + 3) * np.sqrt(1.4)
    xi = np.array([v[t + 3] + 0.45 * v[t + 2] + 0.30 * v[t + 1] + 0.55 * v[t] for t in range(T)])
    prior = M.PsiPrior()
    star = np.array([0.40, 0.27, 0.62])

    values = [
        M.log_conditional_psi_ordinate(star, xi, 1.3, prior=prior, n_draws=20_000, seed=s)
        for s in (1, 2, 3)
    ]
    assert np.std(values) < 0.01, f"ordinate seed spread {np.std(values):.4f} is too large"


# ------------------------------------------------------------------
# 5. The MA samplers must nest production, which starts with matching defaults
# ------------------------------------------------------------------

def _opts_defaults(func) -> dict[str, str]:
    """Extract ``_getd(opts, "key", default)`` initialisations from a sampler's source."""
    import inspect
    import re

    out: dict[str, str] = {}
    for line in inspect.getsource(func).splitlines():
        # ces uses ``getd``, the HSA samplers ``_getd``.
        match = re.match(
            r'\s*\w+ = (?:float|bool|int|str)?\(?_?getd\(opts, "(\w+)", (.+?)\)\)?$', line.strip()
        )
        if match:
            out[match.group(1)] = match.group(2)
    return out


# Options the MA samplers add, plus the innovation-variance aliases. On the
# aliases the two sides differ only in which name wins when *both* are supplied:
# production reads sigma_e20 first, the MA samplers read sigma_v20 first, since
# that is what the parameter now is. The resolved default is 1.0 either way and
# no caller in scripts/ or inference/ passes either key, so the branch is
# unreachable -- unlike the Sigma0 default, which was reachable and wrong.
_ALLOWED_TO_DIFFER = {"ma_order", "n_psi_steps", "psi_init_scale", "psi0",
                      "sigma_e20", "sigma_v20", "sigma_eta20"}


@pytest.mark.parametrize(
    "ma_module, ma_name, prod_module, prod_name, extra",
    [
        (
            "nkpc_hsa.error_robustness.hsa_dynamic_ma3", "func_nkpc_hsa_dynamic_ma3",
            "nkpc_hsa.gibbs.hsa_dynamic.model", "func_nkpc_hsa_decomp_joint_fullSigma",
            _ALLOWED_TO_DIFFER,
        ),
        (
            "nkpc_hsa.error_robustness.hsa_steady_ma3", "func_nkpc_hsa_steady_ma3",
            "nkpc_hsa.gibbs.hsa_steady.model", "func_nkpc_hsa_decomp_tv_kappa_kalman",
            _ALLOWED_TO_DIFFER,
        ),
        (
            "nkpc_hsa.error_robustness.hsa_full_ma3", "func_nkpc_hsa_full_ma3",
            "nkpc_hsa.gibbs.hsa_full_pg.model", "func_nkpc_hsa_full_pg",
            _ALLOWED_TO_DIFFER,
        ),
        (
            "nkpc_hsa.error_robustness.ces_ma3", "func_nkpc_ces_ma3",
            "nkpc_hsa.gibbs.ces.model", "func_nkpc_ces",
            _ALLOWED_TO_DIFFER,
        ),
    ],
)
def test_initial_values_match_production(ma_module, ma_name, prod_module, prod_name, extra):
    """A different starting value is indistinguishable from a sampler bug here.

    ``rho_1``/``rho_2`` have an integrated autocorrelation time near 90, so the
    2000-sweep burn-in is only about 20 effective draws for that block. A
    mismatched initial ``sigma_u2``/``sigma_eps2`` therefore survives burn-in and
    shifts the posterior means by half a standard deviation -- which is exactly
    what happened while this package was being written, and cost a full round of
    exact-smoother checks to localise. Pin the defaults instead.
    """
    import importlib

    mine = _opts_defaults(getattr(importlib.import_module(ma_module), ma_name))
    prod = _opts_defaults(getattr(importlib.import_module(prod_module), prod_name))
    shared = (set(mine) | set(prod)) - extra
    mismatched = {k: (mine.get(k), prod.get(k)) for k in shared if mine.get(k) != prod.get(k)}
    assert not mismatched, f"initial-value defaults drifted from production: {mismatched}"


def test_particle_gibbs_ma_reduces_to_production_bit_for_bit():
    """psi = [] must leave the conditional-SMC sweep byte-identical.

    The MA version solves for ``v_t`` instead of proposing it, but with no MA
    lags the solved innovation *is* the disturbance and the weight is production's
    inflation term, so nothing changes -- including the RNG consumption.
    """
    from nkpc_hsa.error_robustness.particle_gibbs_ma3 import sample_states_particle_gibbs_ma
    from nkpc_hsa.gibbs.hsa_full_pg.model import sample_states_particle_gibbs

    rng = np.random.default_rng(0)
    T = 40
    shared = dict(
        y=rng.standard_normal(T), a_t=rng.standard_normal(T), x_t=rng.standard_normal(T),
        zeta=rng.standard_normal(T) * 0.3,
        N_obs=np.where(np.arange(T) % 4 == 3, np.cumsum(rng.standard_normal(T)) * 0.1, np.nan),
        alpha=0.8, kappa0_eff=0.05, delta_eff=0.01, theta0=0.3, gamma=0.02, lambda_ez=0.1,
        rho1=0.6, rho2=-0.2, n_drift=0.01,
        sigma_u2=0.02, sigma_eps2=0.01, sigma_N2=0.005,
        Nbar_ref=np.cumsum(rng.standard_normal(T)) * 0.05,
        Nhat_ref=rng.standard_normal(T) * 0.1, Nhat_ref_lag=0.0,
        m0_Nhat=0.0, P0_Nhat=0.5, m0_Nhat_lag=0.0, P0_Nhat_lag=0.5,
        m0_Nbar=0.0, P0_Nbar=2.0, n_particles=128,
    )
    produced = sample_states_particle_gibbs(
        sigma_eta2=0.3, rng=np.random.default_rng(7), **shared
    )
    nested = sample_states_particle_gibbs_ma(
        psi=np.zeros(0), sigma_v2=0.3, v_presample_ref=np.zeros(0),
        rng=np.random.default_rng(7), **shared,
    )
    assert np.array_equal(produced["Nhat"], nested["Nhat"])
    assert np.array_equal(produced["Nbar"], nested["Nbar"])
    assert produced["ess_mean"] == nested["ess_mean"]
    assert produced["moved_frac"] == nested["moved_frac"]


def test_particle_gibbs_ma_innovation_reproduces_the_disturbance():
    """psi(L) applied to the drawn innovation must return the inflation residual."""
    from nkpc_hsa.error_robustness.particle_gibbs_ma3 import sample_states_particle_gibbs_ma

    rng = np.random.default_rng(0)
    T = 40
    shared = dict(
        y=rng.standard_normal(T), a_t=rng.standard_normal(T), x_t=rng.standard_normal(T),
        zeta=rng.standard_normal(T) * 0.3,
        N_obs=np.where(np.arange(T) % 4 == 3, np.cumsum(rng.standard_normal(T)) * 0.1, np.nan),
        alpha=0.8, kappa0_eff=0.05, delta_eff=0.01, theta0=0.3, gamma=0.02, lambda_ez=0.1,
        rho1=0.6, rho2=-0.2, n_drift=0.01,
        sigma_u2=0.02, sigma_eps2=0.01, sigma_N2=0.005,
        Nbar_ref=np.cumsum(rng.standard_normal(T)) * 0.05,
        Nhat_ref=rng.standard_normal(T) * 0.1, Nhat_ref_lag=0.0,
        m0_Nhat=0.0, P0_Nhat=0.5, m0_Nhat_lag=0.0, P0_Nhat_lag=0.5,
        m0_Nbar=0.0, P0_Nbar=2.0, n_particles=128,
    )
    drawn = sample_states_particle_gibbs_ma(
        psi=PSI, sigma_v2=0.3, v_presample_ref=np.zeros(PSI.size),
        rng=np.random.default_rng(7), **shared,
    )
    mu = (
        shared["alpha"] * shared["a_t"]
        + shared["kappa0_eff"] * shared["x_t"]
        + shared["delta_eff"] * shared["x_t"] * drawn["Nbar"]
        - shared["theta0"] * drawn["Nhat"]
        - shared["gamma"] * drawn["Nbar"] * drawn["Nhat"]
        + shared["lambda_ez"] * shared["zeta"]
    )
    rebuilt = M.ma_filter(drawn["v"], PSI) + 0.0
    history = list(drawn["v_presample"])
    exact = []
    for t in range(T):
        exact.append(drawn["v"][t] + float(np.dot(PSI, history)))
        history = [drawn["v"][t]] + history[: PSI.size - 1]
    assert np.allclose(np.array(exact), shared["y"] - mu, atol=1e-10)
    assert rebuilt.shape == (T,)


def test_joint_loglik_ma_matches_production_kalman_at_psi_zero():
    from nkpc_hsa.error_robustness.joint_ffbs_ma3 import joint_loglik_ma
    from nkpc_hsa.gibbs.conditional_ml import kalman_loglik

    rng = np.random.default_rng(5)
    T = 80
    shared = dict(
        N_obs=np.where(np.arange(T) % 4 == 3, np.cumsum(rng.standard_normal(T)) * 0.1, np.nan),
        y_tilde=rng.standard_normal(T) * 0.4,
        h_nhat=np.zeros(T), h_nbar=0.03 * rng.standard_normal(T),
        n_drift=0.02, rho1=1.1, rho2=-0.3,
        sigma_u2=0.02, sigma_eps2=0.01, sigma_N2=0.005,
        m0=np.zeros(3), P0=np.eye(3) * 10.0,
    )
    assert joint_loglik_ma(psi=np.zeros(0), sigma_v2=0.3, **shared) == pytest.approx(
        kalman_loglik(sigma_eta2=0.3, **shared), abs=1e-8
    )


def test_joint_loglik_ma_matches_the_banded_density_without_states():
    """With the state block switched off the filter must return the banded Gaussian."""
    from nkpc_hsa.error_robustness.joint_ffbs_ma3 import joint_loglik_ma

    rng = np.random.default_rng(3)
    T = 60
    y = rng.standard_normal(T) * 0.5
    sigma_v2 = 0.3
    filtered = joint_loglik_ma(
        N_obs=np.full(T, np.nan), y_tilde=y, h_nhat=np.zeros(T), h_nbar=np.zeros(T),
        n_drift=0.0, rho1=0.0, rho2=0.0, psi=PSI, sigma_v2=sigma_v2,
        sigma_u2=1e-12, sigma_eps2=1e-12, sigma_N2=1.0,
        m0=np.zeros(3), P0=np.zeros((3, 3)),
    )
    direct = M.MAWeighting(PSI, T).log_likelihood(y, sigma_v2)
    assert filtered == pytest.approx(direct, abs=1e-6)


def test_heteroskedastic_weighting_matches_dense_reference():
    """hsa_dynamic's innovation variance is not constant across t."""
    rng = np.random.default_rng(0)
    n = 25
    var = 0.5 + rng.random(n)
    weighting = M.MAWeighting(PSI, n, innovation_var=var, presample_var=float(var[0]))

    coef = np.r_[1.0, PSI]
    q = PSI.size
    design = np.zeros((n, n + q))
    for s in range(n):
        for j in range(q + 1):
            design[s, s + q - j] = coef[j]
    reference = design @ np.diag(np.r_[np.full(q, var[0]), var]) @ design.T

    r = rng.standard_normal(n)
    assert weighting.log_det == pytest.approx(np.linalg.slogdet(reference)[1], abs=1e-9)
    assert weighting.quadratic_form(r) == pytest.approx(
        r @ np.linalg.solve(reference, r), abs=1e-9
    )


def test_inverse_ma_filter_round_trips():
    rng = np.random.default_rng(0)
    T = 60
    v = rng.standard_normal(T + PSI.size)
    q = PSI.size
    xi = np.array(
        [v[t + q] + sum(PSI[j] * v[t + q - 1 - j] for j in range(q)) for t in range(T)]
    )
    presample = v[q - 1::-1][:q]
    assert M.inverse_ma_filter(xi, PSI, presample) == pytest.approx(v[q:], abs=1e-10)
