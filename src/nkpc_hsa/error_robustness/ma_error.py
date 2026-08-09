"""MA(q) machinery for the NKPC inflation disturbance.

The disturbance is

    xi_t = psi(L) v_t,   psi(L) = 1 + psi_1 L + ... + psi_q L^q,   v_t ~ iid N(0, sigma_v^2)

with ``psi`` in the invertible region. Everything below works with the
*unit-innovation* covariance

    Omega_0(psi)[s, t] = gamma_{|s-t|},   gamma_k = sum_j psi_j psi_{j+k},  psi_0 = 1

so that ``Cov(xi) = sigma_v^2 * Omega_0(psi)``. ``Omega_0`` is Toeplitz and
banded with bandwidth ``q``, so its Cholesky factor is banded too and every
operation here is O(T q^2) rather than O(T^3). That is what makes it cheap
enough to sit inside a Gibbs sweep.

This is the exact stationary likelihood, not a conditional one: the banded
Toeplitz form already integrates out the pre-sample innovations, so there is no
initialisation approximation and no need to discard the first q observations.

Sampling psi
------------
``psi`` cannot be drawn by a conjugate step. Conditional on the coefficients the
disturbance path is data, and conditional on ``psi`` the innovation path follows
by deterministic inverse recursion -- so a Gibbs pair over (psi, v) is reducible
and would never move. ``sample_psi`` therefore runs random-walk Metropolis on
the exact banded likelihood, with invertibility enforced by rejection.

Because a Metropolis block complicates Chib's marginal likelihood, callers
should place ``psi`` **last** in the block order. Chib's final ordinate factor
is then evaluated with every other block at its starred value, which
``log_conditional_psi_ordinate`` computes by numerically normalising the
conditional over the invertible region -- the same device
``gibbs.conditional_ml.log_stationary_mass`` already uses for the AR(2) block.
No Chib-Jeliazkov correction is needed under that ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded

__all__ = [
    "MA_ORDER",
    "MAWeighting",
    "PsiPrior",
    "AdaptiveRandomWalk",
    "autocovariance",
    "inverse_ma_filter",
    "is_invertible",
    "ma_filter",
    "sample_psi",
    "state_augmentation",
    "log_invertible_mass",
    "log_conditional_psi_ordinate",
]

# Fixed by the four-quarter overlap of year-over-year inflation: pi_t sums four
# quarterly price changes, so an iid quarterly shock enters the estimated
# equation as an MA(3). The order is implied by the data construction, not
# selected -- see the package docstring for the evidence.
MA_ORDER = 3

# Roots of psi(z) must sit strictly outside the unit circle. A bare |z| > 1 test
# admits near-boundary draws whose spectral density touches zero at some
# frequency, which makes Omega_0 numerically singular. Keep a margin.
MIN_ROOT_MODULUS = 1.001


# ============================================================
# Invertibility and autocovariances
# ============================================================

def _as_psi(psi: Any) -> np.ndarray:
    arr = np.asarray(psi, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    return arr


def is_invertible(psi: Any, *, min_root_modulus: float = MIN_ROOT_MODULUS) -> bool:
    """True when every root of ``1 + psi_1 z + ... + psi_q z^q`` is outside the unit circle.

    ``psi = 0`` is invertible (the polynomial is the constant 1 and has no
    roots), which is what makes the i.i.d. baseline a nested special case.
    """
    arr = _as_psi(psi)
    if not np.all(np.isfinite(arr)):
        return False
    if arr.size == 0 or np.all(arr == 0.0):
        return True
    # np.roots wants descending powers and trims leading zeros itself.
    roots = np.roots(np.r_[1.0, arr][::-1])
    if roots.size == 0:
        return True
    return bool(np.all(np.abs(roots) > min_root_modulus))


def autocovariance(psi: Any, sigma2: float = 1.0) -> np.ndarray:
    """Return ``[gamma_0, ..., gamma_q]`` of ``psi(L) v_t`` with ``Var(v) = sigma2``."""
    p = np.r_[1.0, _as_psi(psi)]
    q = p.size
    return np.array([sigma2 * float(np.dot(p[: q - k], p[k:])) for k in range(q)])


def ma_filter(x: Any, psi: Any, *, presample: float = 0.0) -> np.ndarray:
    """Apply ``psi(L)`` to a series, padding pre-sample values with ``presample``.

    Only used for diagnostics. The sampler never needs it: the specification
    keeps the cross-equation loading contemporaneous, so no observed series is
    ever run through the MA filter.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.r_[1.0, _as_psi(psi)]
    out = np.full(x.size, 0.0, dtype=float)
    for k, coef in enumerate(p):
        if coef == 0.0:
            continue
        if k == 0:
            out += coef * x
        else:
            out[k:] += coef * x[:-k]
            out[:k] += coef * presample
    return out


def inverse_ma_filter(xi: Any, psi: Any, presample: Any) -> np.ndarray:
    """Recover the innovation path from the disturbance: ``v = psi(L)^{-1} xi``.

    Runs the recursion ``v_t = xi_t - sum_j psi_j v_{t-j}`` forward from supplied
    pre-sample values. ``presample`` is ``[v_{-1}, v_{-2}, ..., v_{-q}]``.

    ``hsa_dynamic`` needs this because its coefficient blocks condition on the
    innovation rather than on the disturbance, and every coefficient draw moves
    the disturbance. The pre-sample values are not invented: the augmented FFBS
    state at ``t = 0`` carries ``v_{-1..-q}`` as sampled quantities, so passing
    ``states[0, 4:]`` makes the recovery exact rather than an approximation that
    conditions the first ``q`` periods away.

    Only meaningful for invertible ``psi``; the recursion is explosive otherwise,
    which is one more reason the sampler rejects non-invertible draws.
    """
    xi = np.asarray(xi, dtype=float).reshape(-1)
    psi_arr = _as_psi(psi)
    q = psi_arr.size
    if q == 0:
        return xi.copy()

    pre = np.asarray(presample, dtype=float).reshape(-1)
    if pre.size != q:
        raise ValueError(f"presample must have length q={q}, got {pre.size}.")

    history = list(pre)  # history[j] is v_{t-1-j} as the loop advances
    out = np.empty(xi.size, dtype=float)
    for t in range(xi.size):
        out[t] = xi[t] - float(np.dot(psi_arr, history))
        history = [out[t]] + history[: q - 1]
    return out


# ============================================================
# Banded Cholesky of Omega_0(psi)
# ============================================================

class MAWeighting:
    """Banded Cholesky of the disturbance covariance for a fixed sample length.

    Two modes.

    *Homoskedastic* (``innovation_var=None``, the default). The covariance is
    ``Omega_0(psi)``, Toeplitz with ``gamma_k = sum_j psi_j psi_{j+k}``, and the
    caller supplies the scale separately as ``sigma2``. This is what ``ces_ma3``
    and ``hsa_steady_ma3`` use.

    *Heteroskedastic* (``innovation_var`` a length-``n_obs`` array). The
    innovation variance varies by period, so the covariance is
    ``L_psi diag(innovation_var) L_psi'`` -- still banded with bandwidth ``q``,
    but no longer Toeplitz:

        Cov(xi_s, xi_t) = sum_{j=k}^{q} psi_j psi_{j-k} * innovation_var[s-j],
        k = s - t >= 0

    ``hsa_dynamic`` needs this: the innovation there is the first element of a
    correlated 4-vector, and its conditional variance at ``t = 0`` differs from
    ``t >= 1`` because only ``zeta_0`` is available to condition on, not
    ``u_0`` and ``eps_0``. Absolute variances are already folded in, so pass
    ``sigma2 = 1.0`` in this mode.

    Build once per ``(psi, T)`` and reuse across every block of a Gibbs sweep
    that needs GLS weighting. All methods are O(T q^2).
    """

    def __init__(
        self,
        psi: Any,
        n_obs: int,
        innovation_var: Any | None = None,
        presample_var: float | None = None,
    ) -> None:
        psi_arr = _as_psi(psi)
        n_obs = int(n_obs)
        if n_obs <= 0:
            raise ValueError("n_obs must be positive.")
        if not is_invertible(psi_arr):
            raise ValueError(f"psi={psi_arr} is not invertible; refusing to build Omega_0.")

        coef = np.r_[1.0, psi_arr]
        q = min(coef.size - 1, n_obs - 1)
        ab = np.zeros((q + 1, n_obs), dtype=float)

        if innovation_var is None:
            gamma = autocovariance(psi_arr, 1.0)
            # Lower banded storage: ab[i, j] = Cov[i + j, j] = gamma_i.
            for k in range(q + 1):
                ab[k, : n_obs - k] = gamma[k]
            self.innovation_var = None
        else:
            var = np.asarray(innovation_var, dtype=float).reshape(-1)
            if var.size != n_obs:
                raise ValueError("innovation_var must have length n_obs.")
            if np.any(~np.isfinite(var)) or np.any(var <= 0.0):
                raise ValueError("innovation_var must be finite and positive.")
            # Pre-sample innovations are conditioned on nothing, so their
            # variance is the *unconditional* one, which the caller knows and
            # the in-sample conditional variances do not equal. hsa_dynamic
            # passes Sigma[0, 0]; without it the first q rows would be built
            # from a conditional variance that does not apply to them.
            pre_var = float(var[0]) if presample_var is None else float(presample_var)
            if not (pre_var > 0.0 and np.isfinite(pre_var)):
                raise ValueError("presample_var must be finite and positive.")
            padded = np.r_[np.full(coef.size - 1, pre_var), var]
            for k in range(q + 1):
                for s in range(k, n_obs):
                    ab[k, s - k] = float(
                        sum(
                            coef[j] * coef[j - k] * padded[s - j + coef.size - 1]
                            for j in range(k, coef.size)
                        )
                    )
            self.innovation_var = var

        self.psi = psi_arr
        self.n_obs = n_obs
        self._chol = cholesky_banded(ab, lower=True)
        self._logdet = 2.0 * float(np.sum(np.log(self._chol[0])))

    def __repr__(self) -> str:
        return f"MAWeighting(psi={np.round(self.psi, 4).tolist()}, n_obs={self.n_obs})"

    @property
    def bandwidth(self) -> int:
        return self._chol.shape[0] - 1

    @property
    def log_det(self) -> float:
        """``log |Omega_0(psi)|``."""
        return self._logdet

    def solve(self, b: Any) -> np.ndarray:
        """Return ``Omega_0^{-1} b``; ``b`` may be a vector or a column-stacked matrix."""
        arr = np.asarray(b, dtype=float)
        if arr.shape[0] != self.n_obs:
            raise ValueError(f"leading dimension {arr.shape[0]} does not match n_obs={self.n_obs}")
        return cho_solve_banded((self._chol, True), arr)

    def whiten(self, b: Any) -> np.ndarray:
        """Return ``L^{-1} b`` where ``Omega_0 = L L'`` -- the GLS-whitened series."""
        arr = np.asarray(b, dtype=float)
        if arr.shape[0] != self.n_obs:
            raise ValueError(f"leading dimension {arr.shape[0]} does not match n_obs={self.n_obs}")
        # cholesky_banded's lower output is already the (l, u) = (q, 0) banded
        # storage that solve_banded expects for L.
        return solve_banded((self.bandwidth, 0), self._chol, arr)

    def quadratic_form(self, r: Any) -> float:
        """Return ``r' Omega_0^{-1} r``."""
        arr = np.asarray(r, dtype=float).reshape(-1)
        return float(arr @ self.solve(arr))

    def log_likelihood(self, r: Any, sigma2: float) -> float:
        """Exact Gaussian log density of ``r`` under ``N(0, sigma2 * Omega_0(psi))``."""
        if not (sigma2 > 0.0 and np.isfinite(sigma2)):
            raise ValueError("sigma2 must be finite and positive.")
        n = self.n_obs
        return -0.5 * (
            n * np.log(2.0 * np.pi)
            + n * np.log(sigma2)
            + self.log_det
            + self.quadratic_form(r) / sigma2
        )

    def gls_moments(self, y: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(X' Omega_0^{-1} X, X' Omega_0^{-1} y)`` for a GLS coefficient block."""
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        WX = self.solve(X_arr)
        return X_arr.T @ WX, X_arr.T @ self.solve(y_arr)


# ============================================================
# Prior and Metropolis block for psi
# ============================================================

@dataclass(frozen=True)
class PsiPrior:
    """Independent Gaussian prior on ``psi``, truncated to the invertible region.

    The default is deliberately loose and centred at zero: centring on the
    equal-weight overlap value ``(1, 1, 1)`` would build in a restriction the
    data reject outright (see the package docstring).
    """

    mean: np.ndarray = field(default_factory=lambda: np.zeros(MA_ORDER))
    sd: np.ndarray = field(default_factory=lambda: np.full(MA_ORDER, 0.5))

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        sd = np.asarray(self.sd, dtype=float).reshape(-1)
        if mean.size != sd.size:
            raise ValueError("psi prior mean and sd must have the same length.")
        if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
            raise ValueError("psi prior sd must be finite and positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sd", sd)

    @property
    def order(self) -> int:
        return int(self.mean.size)

    def log_pdf(self, psi: Any) -> float:
        """Un-normalised log prior: ``-inf`` outside the invertible region."""
        arr = _as_psi(psi)
        if not is_invertible(arr):
            return -np.inf
        z = (arr - self.mean) / self.sd
        return float(-0.5 * np.sum(z**2) - np.sum(np.log(self.sd)) - 0.5 * arr.size * np.log(2.0 * np.pi))

    @classmethod
    def from_config(cls, priors: Optional[dict[str, Any]], order: int = MA_ORDER) -> "PsiPrior":
        """Read ``mu_psi`` / ``sigma_psi`` from a priors dict, falling back to the default."""
        priors = priors or {}
        mean = priors.get("mu_psi")
        sd = priors.get("sigma_psi")
        mean_arr = np.zeros(order) if mean is None else np.asarray(mean, dtype=float).reshape(-1)
        sd_arr = np.full(order, 0.5) if sd is None else np.asarray(sd, dtype=float).reshape(-1)
        if mean_arr.size == 1:
            mean_arr = np.full(order, float(mean_arr[0]))
        if sd_arr.size == 1:
            sd_arr = np.full(order, float(sd_arr[0]))
        return cls(mean=mean_arr, sd=sd_arr)


class AdaptiveRandomWalk:
    """Random-walk proposal whose scale and shape adapt during burn-in only.

    Adaptation must stop before the retained draws begin: a chain with a
    perpetually adapting kernel is not Markov, and Chib's reduced runs assume a
    fixed kernel. ``freeze`` is called by the sampler at the end of burn-in.
    """

    def __init__(
        self,
        dim: int,
        *,
        init_scale: float = 0.1,
        target_accept: float = 0.30,
        adapt_after: int = 50,
    ) -> None:
        self.dim = int(dim)
        self.log_scale = float(np.log(init_scale))
        self.target_accept = float(target_accept)
        self.adapt_after = int(adapt_after)
        self.cov = np.eye(self.dim)
        self._chol = np.eye(self.dim)
        self._n = 0
        self._mean = np.zeros(self.dim)
        self._m2 = np.zeros((self.dim, self.dim))
        self.frozen = False
        self.n_accept = 0
        self.n_propose = 0

    def propose(self, current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        step = self._chol @ rng.standard_normal(self.dim)
        return current + np.exp(self.log_scale) * step

    def register(self, psi: np.ndarray, accepted: bool) -> None:
        self.n_propose += 1
        self.n_accept += int(accepted)
        if self.frozen:
            return

        # Robbins-Monro on the log scale, targeting the acceptance rate.
        gain = 1.0 / max(self._n + 1, 10) ** 0.6
        self.log_scale += gain * (float(accepted) - self.target_accept)
        self.log_scale = float(np.clip(self.log_scale, -12.0, 4.0))

        # Running empirical covariance for the proposal shape.
        self._n += 1
        delta = psi - self._mean
        self._mean += delta / self._n
        self._m2 += np.outer(delta, psi - self._mean)
        if self._n > max(self.adapt_after, 5 * self.dim):
            emp = self._m2 / (self._n - 1)
            shrunk = 0.95 * emp + 0.05 * np.eye(self.dim) * max(np.trace(emp) / self.dim, 1e-8)
            try:
                self._chol = np.linalg.cholesky(shrunk)
            except np.linalg.LinAlgError:
                pass

    def freeze(self) -> None:
        self.frozen = True

    @property
    def acceptance_rate(self) -> float:
        return self.n_accept / self.n_propose if self.n_propose else float("nan")


def sample_psi(
    psi: np.ndarray,
    resid: np.ndarray,
    sigma2: float,
    *,
    prior: PsiPrior,
    proposal: AdaptiveRandomWalk,
    rng: np.random.Generator,
    n_steps: int = 1,
    weighting: Optional[MAWeighting] = None,
) -> tuple[np.ndarray, MAWeighting]:
    """One or more random-walk Metropolis steps for ``psi | resid, sigma2``.

    ``resid`` is the inflation disturbance net of everything that does not
    involve ``psi`` -- i.e. ``xi``. Returns the (possibly unchanged) ``psi`` and
    the ``MAWeighting`` matching it, so the caller can reuse the factorisation
    without rebuilding it.
    """
    current = _as_psi(psi).copy()
    resid = np.asarray(resid, dtype=float).reshape(-1)
    n_obs = resid.size

    if weighting is None or not np.array_equal(weighting.psi, current):
        weighting = MAWeighting(current, n_obs)
    log_post = weighting.log_likelihood(resid, sigma2) + prior.log_pdf(current)

    for _ in range(max(1, int(n_steps))):
        candidate = proposal.propose(current, rng)
        accepted = False
        if is_invertible(candidate):
            try:
                cand_weighting = MAWeighting(candidate, n_obs)
            except (ValueError, np.linalg.LinAlgError):
                cand_weighting = None
            if cand_weighting is not None:
                cand_post = cand_weighting.log_likelihood(resid, sigma2) + prior.log_pdf(candidate)
                if np.isfinite(cand_post) and np.log(rng.random()) < cand_post - log_post:
                    current, weighting, log_post = candidate, cand_weighting, cand_post
                    accepted = True
        proposal.register(current, accepted)

    return current, weighting


# ============================================================
# State-space augmentation
# ============================================================

def state_augmentation(psi: Any, sigma_v2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(F_v, Q_v, h_v, P0_v)`` for the MA block appended to a state vector.

    The block carries ``(v_t, v_{t-1}, ..., v_{t-q})`` so that the inflation
    observation row can load ``xi_t = [1, psi_1, ..., psi_q] . v_block``. The
    inflation row then has **no** measurement noise of its own -- the whole
    disturbance lives in the state.

    ``P0_v = sigma_v2 * I`` is the exact stationary initial covariance, because
    the ``v`` are i.i.d.; unlike the AR blocks there is no initialisation
    approximation here.
    """
    psi_arr = _as_psi(psi)
    q = psi_arr.size
    dim = q + 1
    if not (sigma_v2 > 0.0 and np.isfinite(sigma_v2)):
        raise ValueError("sigma_v2 must be finite and positive.")

    F_v = np.zeros((dim, dim), dtype=float)
    if q > 0:
        F_v[1:, :-1] = np.eye(q)
    Q_v = np.zeros((dim, dim), dtype=float)
    Q_v[0, 0] = float(sigma_v2)
    h_v = np.r_[1.0, psi_arr]
    P0_v = float(sigma_v2) * np.eye(dim)
    return F_v, Q_v, h_v, P0_v


# ============================================================
# Chib support
# ============================================================

_INVERTIBLE_MASS_SEED = 20260807


def log_invertible_mass(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    n_draws: int = 200_000,
    seed: int = _INVERTIBLE_MASS_SEED,
) -> float:
    """Monte Carlo ``log Pr(psi invertible)`` under ``N(mean, cov)``.

    Mirrors ``gibbs.conditional_ml.log_stationary_mass``: the same fixed seed is
    used for the prior term and for every ordinate factor so the Monte Carlo
    errors cancel rather than accumulate.
    """
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.asarray(mean, dtype=float), np.asarray(cov, dtype=float), size=n_draws)
    share = float(np.mean([is_invertible(row) for row in draws]))
    if share <= 0.0:
        raise ValueError("No invertible draws: the psi conditional is degenerate.")
    return float(np.log(share))


def log_conditional_psi_ordinate(
    psi_star: np.ndarray,
    resid: np.ndarray,
    sigma2: float,
    *,
    prior: PsiPrior,
    n_draws: int = 20_000,
    df: float = 5.0,
    seed: int = _INVERTIBLE_MASS_SEED,
) -> float:
    """``log p(psi* | resid, sigma2)`` by importance sampling.

    Chib's final ordinate factor is evaluated with every other block at its
    starred value, so the normalising constant is computed once per marginal
    likelihood rather than averaged over a reduced run. That is what makes
    direct normalisation affordable and Chib-Jeliazkov unnecessary.

    A product grid was tried first and rejected: in three dimensions it needed
    more kernel evaluations than importance sampling for an order of magnitude
    less accuracy (0.13 log units between a 26- and a 40-point rule, which is
    large enough to move a Bayes factor). Here the conditional is fitted by
    Laplace approximation and used as the centre of a multivariate-t proposal;
    the heavy tails keep the weights bounded where the Gaussian fit is too
    narrow, and the invertibility indicator is folded into the integrand.

    The seed is fixed and shared with :func:`log_invertible_mass` so the Monte
    Carlo errors of the prior and ordinate terms cancel rather than accumulate.
    """
    from scipy.optimize import minimize

    psi_star = _as_psi(psi_star)
    resid = np.asarray(resid, dtype=float).reshape(-1)
    order = psi_star.size
    n_obs = resid.size

    def log_kernel(psi: Any) -> float:
        arr = _as_psi(psi)
        if not is_invertible(arr):
            return -np.inf
        try:
            weighting = MAWeighting(arr, n_obs)
        except (ValueError, np.linalg.LinAlgError):
            return -np.inf
        value = weighting.log_likelihood(resid, sigma2) + prior.log_pdf(arr)
        return value if np.isfinite(value) else -np.inf

    # --- Laplace fit, started from psi_star ---
    neg = lambda p: -log_kernel(p) if np.isfinite(log_kernel(p)) else 1e12
    opt = minimize(neg, psi_star, method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-7, fatol=1e-7))
    mode = _as_psi(opt.x) if is_invertible(opt.x) else psi_star

    step = 1e-4
    hess = np.zeros((order, order))
    f0 = log_kernel(mode)
    for i in range(order):
        for j in range(i, order):
            ei = np.zeros(order); ei[i] = step
            ej = np.zeros(order); ej[j] = step
            fpp = log_kernel(mode + ei + ej)
            fpm = log_kernel(mode + ei - ej)
            fmp = log_kernel(mode - ei + ej)
            fmm = log_kernel(mode - ei - ej)
            if not all(np.isfinite([fpp, fpm, fmp, fmm])):
                hess[i, j] = hess[j, i] = -1.0 / prior.sd[i] ** 2 if i == j else 0.0
                continue
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * step * step)
    try:
        cov = np.linalg.inv(-hess)
        cov = (cov + cov.T) / 2.0
        vals, vecs = np.linalg.eigh(cov)
        if np.any(vals <= 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov = np.diag(prior.sd**2)
    chol = np.linalg.cholesky(cov * (df - 2.0) / df if df > 2.0 else cov)

    # --- multivariate-t importance sampling ---
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_draws, order))
    chi = rng.chisquare(df, size=n_draws)
    draws = mode + (z @ chol.T) / np.sqrt(chi / df)[:, None]

    from scipy.special import gammaln

    def log_t_pdf(points: np.ndarray) -> np.ndarray:
        diff = points - mode
        sol = np.linalg.solve(chol, diff.T).T
        maha = np.sum(sol**2, axis=1)
        const = (
            gammaln((df + order) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * order * np.log(df * np.pi)
            - float(np.sum(np.log(np.diag(chol))))
        )
        return const - 0.5 * (df + order) * np.log1p(maha / df)

    log_q = log_t_pdf(draws)
    log_p = np.array([log_kernel(row) for row in draws])
    log_w = log_p - log_q
    finite = log_w[np.isfinite(log_w)]
    if finite.size < 100:
        raise ValueError("psi conditional: too few finite importance weights.")
    peak = float(np.max(finite))
    log_norm = peak + np.log(np.sum(np.exp(finite - peak)) / n_draws)

    value = log_kernel(psi_star)
    if not np.isfinite(value):
        raise ValueError("psi_star is outside the invertible region.")
    return float(value - log_norm)
