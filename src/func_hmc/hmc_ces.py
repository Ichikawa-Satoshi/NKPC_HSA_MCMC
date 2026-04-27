import numpyro
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan

def model_ces(pi, pi_prev, pi_expect, x, x_prev, l, prior_family):
    priors = prior_family.distributions
    # NKPC params
    alpha = numpyro.sample("alpha", priors["alpha"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    # AR(1) param for x
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])
    # Variance parameters
    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])
    # Correlation parameter
    rho = numpyro.sample("rho", dist.Uniform(-0.99, 0.99))
    sigma_e_zeta = rho * sigma_e * sigma_zeta
    numpyro.deterministic("sigma_e_zeta", sigma_e_zeta)
    # Covariance matrix
    cov_matrix = jnp.array([[sigma_e**2, sigma_e_zeta],
                            [sigma_e_zeta, sigma_zeta**2]])
    def transition(carry, t):
        # X equation
        x_pred = phi_1 * x_prev[t]
        zeta_t = x[t] - x_pred
        # Pi equation
        pi_pred = alpha * pi_prev[t] + (1 - alpha) * pi_expect[t] + kappa * x[t]
        e_t = pi[t] - pi_pred
        # Joint distribution of residuals
        residuals = jnp.array([e_t, zeta_t])
        numpyro.sample(f"residuals_{t}", dist.MultivariateNormal(jnp.zeros(2), cov_matrix), obs=residuals)
        return carry, None
    timesteps = jnp.arange(0, l)
    scan(transition, None, timesteps)

def model_ces_orth(pi, pi_prev, pi_expect, x, x_prev, l, prior_family):
    priors = prior_family.distributions
    # NKPC params
    alpha = numpyro.sample("alpha", priors["alpha"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    # AR(1) param for x
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])
    # Variance parameters
    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])

    x_pred = phi_1 * x_prev
    pi_pred = alpha * pi_prev + (1 - alpha) * pi_expect + kappa * x
    numpyro.sample("obs", dist.Normal(pi_pred, sigma_e), obs=pi)
    numpyro.sample("x_obs", dist.Normal(x_pred, sigma_zeta), obs=x)