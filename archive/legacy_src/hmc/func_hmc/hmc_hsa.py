import numpyro
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.control_flow import scan
# Dynamic Effects Only (with correlation between e_t and zeta_t)
def model_hsa_dynamic(pi, pi_prev, pi_expect, x, x_prev, N, l, prior_family):
    priors = prior_family.distributions
    alpha = numpyro.sample("alpha", priors["alpha"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])

    r = numpyro.sample("r", dist.Uniform(-1.0, 1.0))
    p = numpyro.sample("p", dist.Uniform(0.1, 0.9))
    rho_1 = 2.0 * r * jnp.cos(jnp.pi * p)
    rho_2 = -r**2
    numpyro.deterministic("rho_1", rho_1)
    numpyro.deterministic("rho_2", rho_2)
    n = numpyro.sample("n", priors["n"])
    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])
    sigma_u = numpyro.sample("sigma_u", priors["sigma_u"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    rho_e_zeta = numpyro.sample("rho_e_zeta", dist.Uniform(-0.99, 0.99))
    sigma_e_zeta = rho_e_zeta * sigma_e * sigma_zeta
    numpyro.deterministic("sigma_e_zeta", sigma_e_zeta)
    cov_2x2 = jnp.array([
        [sigma_e**2, sigma_e_zeta],
        [sigma_e_zeta, sigma_zeta**2],
    ])
    bar_N_0 = numpyro.sample("bar_N_0", dist.Normal(N[0], 1.0))
    hat_N_0 = numpyro.sample("hat_N_0", dist.Normal(0.0, 0.05))
    hat_N_1 = numpyro.sample("hat_N_1", dist.Normal(0.0, 0.05))
    def transition(carry, t):
        Nbar_prev, Nhat_prev_1, Nhat_prev_2 = carry
        Nbar_t = numpyro.sample(f"Nbar_{t}", dist.Normal(n + Nbar_prev, sigma_eps))
        Nhat_pred = rho_1 * Nhat_prev_1 + rho_2 * Nhat_prev_2
        Nhat_obs = N[t] - Nbar_t
        u_t = Nhat_obs - Nhat_pred
        Nhat_t = numpyro.deterministic(f"Nhat_{t}", Nhat_obs)
        x_pred = phi_1 * x_prev[t]
        zeta_t = x[t] - x_pred
        pi_pred = alpha * pi_prev[t] + (1.0 - alpha) * pi_expect[t] + kappa * x[t] - theta * Nhat_t
        e_t = pi[t] - pi_pred
        ez_resid = jnp.array([e_t, zeta_t])
        numpyro.sample(
            f"ez_residuals_{t}",
            dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=cov_2x2),
            obs=ez_resid,
        )

        numpyro.sample(f"u_residual_{t}", dist.Normal(0.0, sigma_u), obs=u_t)

        return (Nbar_t, Nhat_t, Nhat_prev_1), None

    timesteps = jnp.arange(1, l)
    init_carry = (bar_N_0, hat_N_0, hat_N_1)
    scan(transition, init_carry, timesteps)

# Dynamic Effects Only (orthogonalized)
def model_hsa_dynamic_orth(pi, pi_prev, pi_expect, x, x_prev, N, l, prior_family):
    priors = prior_family.distributions
    alpha = numpyro.sample("alpha", priors["alpha"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])

    r = numpyro.sample("r", dist.Uniform(-1.0, 1.0))
    p = numpyro.sample("p", dist.Uniform(0.1, 0.9))
    rho_1 = 2.0 * r * jnp.cos(jnp.pi * p)
    rho_2 = -r**2
    numpyro.deterministic("rho_1", rho_1)
    numpyro.deterministic("rho_2", rho_2)

    n = numpyro.sample("n", priors["n"])
    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])
    sigma_u = numpyro.sample("sigma_u", priors["sigma_u"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])

    bar_N_0 = numpyro.sample("bar_N_0", dist.Normal(N[0], 1.0))
    hat_N_0 = numpyro.sample("hat_N_0", dist.Normal(0.0, 0.05))
    hat_N_1 = numpyro.sample("hat_N_1", dist.Normal(0.0, 0.05))
    def transition(carry, t):
        Nbar_prev, Nhat_prev_1, Nhat_prev_2 = carry
        Nbar_t = numpyro.sample(f"Nbar_{t}", dist.Normal(n + Nbar_prev, sigma_eps))
        Nhat_pred = rho_1 * Nhat_prev_1 + rho_2 * Nhat_prev_2
        Nhat_obs = N[t] - Nbar_t
        u_t = Nhat_obs - Nhat_pred
        Nhat_t = numpyro.deterministic(f"Nhat_{t}", Nhat_obs)
        x_pred = phi_1 * x_prev[t]
        pi_pred = alpha * pi_prev[t] + (1.0 - alpha) * pi_expect[t] + kappa * x[t] - theta * Nhat_t
        numpyro.sample(f"pi_obs_{t}", dist.Normal(pi_pred, sigma_e), obs=pi[t])
        numpyro.sample(f"x_obs_{t}", dist.Normal(x_pred, sigma_zeta), obs=x[t])
        numpyro.sample(f"u_residual_{t}", dist.Normal(0.0, sigma_u), obs=u_t)
        return (Nbar_t, Nhat_t, Nhat_prev_1), None
    timesteps = jnp.arange(1, l)
    init_carry = (bar_N_0, hat_N_0, hat_N_1)
    scan(transition, init_carry, timesteps)


# Steady-State Only
def model_hsa_steady(pi, pi_prev, pi_expect, x, x_prev, N, l, prior_family):
    priors = prior_family.distributions

    alpha = numpyro.sample("alpha", priors["alpha"])
    delta = numpyro.sample("delta", priors["delta"])
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])

    r = numpyro.sample("r", dist.Uniform(-1.0, 1.0))
    p = numpyro.sample("p", dist.Uniform(0.1, 0.9))
    rho_1 = 2.0 * r * jnp.cos(jnp.pi * p)
    rho_2 = -r**2
    numpyro.deterministic("rho_1", rho_1)
    numpyro.deterministic("rho_2", rho_2)

    n = numpyro.sample("n", priors["n"])

    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])
    sigma_u = numpyro.sample("sigma_u", priors["sigma_u"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])

    # correlation only between e_t and zeta_t
    rho_e_zeta = numpyro.sample("rho_e_zeta", dist.Uniform(-0.99, 0.99))
    sigma_e_zeta = rho_e_zeta * sigma_e * sigma_zeta
    numpyro.deterministic("sigma_e_zeta", sigma_e_zeta)

    cov_2x2 = jnp.array([
        [sigma_e**2,   sigma_e_zeta],
        [sigma_e_zeta, sigma_zeta**2],
    ])

    bar_N_0 = numpyro.sample("bar_N_0", dist.Normal(N[0], 1.0))
    hat_N_0 = numpyro.sample("hat_N_0", dist.Normal(0.0, 0.05))
    hat_N_1 = numpyro.sample("hat_N_1", dist.Normal(0.0, 0.05))

    kappa_0 = numpyro.sample("kappa_0", priors["kappa_0"])

    def transition(carry, t):
        Nbar_prev, Nhat_prev_1, Nhat_prev_2 = carry

        # trend
        Nbar_t = numpyro.sample(f"Nbar", dist.Normal(n + Nbar_prev, sigma_eps))

        # cycle
        Nhat_pred = rho_1 * Nhat_prev_1 + rho_2 * Nhat_prev_2
        Nhat_obs = N[t] - Nbar_t
        u_t = Nhat_obs - Nhat_pred
        Nhat_t = numpyro.deterministic(f"Nhat", Nhat_obs)

        # both steady-state and dynamic effects
        kappa_t = kappa_0 + delta * Nbar_t    
        numpyro.deterministic(f"kappa_{t}", kappa_t)

        # x equation
        x_pred = phi_1 * x_prev[t]
        zeta_t = x[t] - x_pred

        # inflation equation
        pi_pred = (
            alpha * pi_prev[t]
            + (1.0 - alpha) * pi_expect[t]
            + kappa_t * x[t]
        )
        e_t = pi[t] - pi_pred

        # correlated block: (e_t, zeta_t)
        ez_resid = jnp.array([e_t, zeta_t])
        numpyro.sample(
            f"ez_residuals_{t}",
            dist.MultivariateNormal(
                jnp.zeros(2),
                covariance_matrix=cov_2x2,
            ),
            obs=ez_resid,
        )

        # independent block: u_t
        numpyro.sample(
            f"u_residual_{t}",
            dist.Normal(0.0, sigma_u),
            obs=u_t,
        )

        return (Nbar_t, Nhat_t, Nhat_prev_1), None

    timesteps = jnp.arange(1, l)
    init_carry = (bar_N_0, hat_N_0, hat_N_1)
    scan(transition, init_carry, timesteps)

def model_hsa_steady_orth(pi, pi_prev, pi_expect, x, x_prev, N, l, prior_family):
    priors = prior_family.distributions

    alpha = numpyro.sample("alpha", priors["alpha"])
    delta = numpyro.sample("delta", priors["delta"])
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])

    r = numpyro.sample("r", dist.Uniform(-1.0, 1.0))
    p = numpyro.sample("p", dist.Uniform(0.1, 0.9))
    rho_1 = 2.0 * r * jnp.cos(jnp.pi * p)
    rho_2 = -r**2
    numpyro.deterministic("rho_1", rho_1)
    numpyro.deterministic("rho_2", rho_2)

    n = numpyro.sample("n", priors["n"])

    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])
    sigma_u = numpyro.sample("sigma_u", priors["sigma_u"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])

    bar_N_0 = numpyro.sample("bar_N_0", dist.Normal(N[0], 1.0))
    hat_N_0 = numpyro.sample("hat_N_0", dist.Normal(0.0, 0.05))
    hat_N_1 = numpyro.sample("hat_N_1", dist.Normal(0.0, 0.05))

    kappa_0 = numpyro.sample("kappa_0", priors["kappa_0"])

    def transition(carry, t):
        Nbar_prev, Nhat_prev_1, Nhat_prev_2 = carry

        Nbar_t = numpyro.sample("Nbar", dist.Normal(n + Nbar_prev, sigma_eps))

        Nhat_pred = rho_1 * Nhat_prev_1 + rho_2 * Nhat_prev_2
        Nhat_t = N[t] - Nbar_t
        u_t = Nhat_t - Nhat_pred
        numpyro.deterministic("Nhat", Nhat_t)

        kappa_t = kappa_0 + delta * Nbar_t
        numpyro.deterministic("kappa_t", kappa_t)

        x_pred = phi_1 * x_prev[t]
        pi_pred = (
            alpha * pi_prev[t]
            + (1.0 - alpha) * pi_expect[t]
            + kappa_t * x[t]
        )

        numpyro.sample("pi_obs", dist.Normal(pi_pred, sigma_e), obs=pi[t])
        numpyro.sample("x_obs", dist.Normal(x_pred, sigma_zeta), obs=x[t])
        numpyro.sample("u_residual", dist.Normal(0.0, sigma_u), obs=u_t)

        return (Nbar_t, Nhat_t, Nhat_prev_1), None

    timesteps = jnp.arange(1, l)
    init_carry = (bar_N_0, hat_N_0, hat_N_1)
    scan(transition, init_carry, timesteps)



def model_hsa_full(pi, pi_prev, pi_expect, x, x_prev, N, l, prior_family):
    priors = prior_family.distributions

    alpha = numpyro.sample("alpha", priors["alpha"])
    delta = numpyro.sample("delta", priors["delta"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])

    r = numpyro.sample("r", dist.Uniform(-1.0, 1.0))
    p = numpyro.sample("p", dist.Uniform(0.1, 0.9))
    rho_1 = 2.0 * r * jnp.cos(jnp.pi * p)
    rho_2 = -r**2
    numpyro.deterministic("rho_1", rho_1)
    numpyro.deterministic("rho_2", rho_2)

    n = numpyro.sample("n", priors["n"])

    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])
    sigma_u = numpyro.sample("sigma_u", priors["sigma_u"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])

    # correlation only between e_t and zeta_t
    rho_e_zeta = numpyro.sample("rho_e_zeta", dist.Uniform(-0.99, 0.99))
    sigma_e_zeta = rho_e_zeta * sigma_e * sigma_zeta
    numpyro.deterministic("sigma_e_zeta", sigma_e_zeta)

    cov_2x2 = jnp.array([
        [sigma_e**2,   sigma_e_zeta],
        [sigma_e_zeta, sigma_zeta**2],
    ])

    bar_N_0 = numpyro.sample("bar_N_0", dist.Normal(N[0], 1.0))
    hat_N_0 = numpyro.sample("hat_N_0", dist.Normal(0.0, 0.05))
    hat_N_1 = numpyro.sample("hat_N_1", dist.Normal(0.0, 0.05))

    kappa_0 = numpyro.sample("kappa_0", priors["kappa_0"])
    theta_0 = numpyro.sample("theta_0", priors["theta_0"])

    def transition(carry, t):
        Nbar_prev, Nhat_prev_1, Nhat_prev_2 = carry

        # trend
        Nbar_t = numpyro.sample(f"Nbar", dist.Normal(n + Nbar_prev, sigma_eps))

        # cycle
        Nhat_pred = rho_1 * Nhat_prev_1 + rho_2 * Nhat_prev_2
        Nhat_obs = N[t] - Nbar_t
        u_t = Nhat_obs - Nhat_pred
        Nhat_t = numpyro.deterministic(f"Nhat", Nhat_obs)

        # both steady-state and dynamic effects
        kappa_t = kappa_0 + delta * Nbar_t
        theta_t = theta_0 + gamma * Nbar_t
        numpyro.deterministic(f"kappa_{t}", kappa_t)
        numpyro.deterministic(f"theta_{t}", theta_t)

        # x equation
        x_pred = phi_1 * x_prev[t]
        zeta_t = x[t] - x_pred

        # inflation equation
        pi_pred = (
            alpha * pi_prev[t]
            + (1.0 - alpha) * pi_expect[t]
            + kappa_t * x[t]
            - theta_t * Nhat_t
        )
        e_t = pi[t] - pi_pred

        # correlated block: (e_t, zeta_t)
        ez_resid = jnp.array([e_t, zeta_t])
        numpyro.sample(
            f"ez_residuals_{t}",
            dist.MultivariateNormal(
                jnp.zeros(2),
                covariance_matrix=cov_2x2,
            ),
            obs=ez_resid,
        )

        # independent block: u_t
        numpyro.sample(
            f"u_residual_{t}",
            dist.Normal(0.0, sigma_u),
            obs=u_t,
        )

        return (Nbar_t, Nhat_t, Nhat_prev_1), None

    timesteps = jnp.arange(1, l)
    init_carry = (bar_N_0, hat_N_0, hat_N_1)
    scan(transition, init_carry, timesteps)



def model_hsa_full_orth(pi, pi_prev, pi_expect, x, x_prev, N, l, prior_family):
    priors = prior_family.distributions

    alpha = numpyro.sample("alpha", priors["alpha"])
    delta = numpyro.sample("delta", priors["delta"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    phi_1 = numpyro.sample("phi_1", priors["phi_1"])

    r = numpyro.sample("r", dist.Uniform(-1.0, 1.0))
    p = numpyro.sample("p", dist.Uniform(0.1, 0.9))
    rho_1 = 2.0 * r * jnp.cos(jnp.pi * p)
    rho_2 = -r**2
    numpyro.deterministic("rho_1", rho_1)
    numpyro.deterministic("rho_2", rho_2)

    n = numpyro.sample("n", priors["n"])

    sigma_e = numpyro.sample("sigma_e", priors["sigma_e"])
    sigma_zeta = numpyro.sample("sigma_zeta", priors["sigma_zeta"])
    sigma_u = numpyro.sample("sigma_u", priors["sigma_u"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])

    bar_N_0 = numpyro.sample("bar_N_0", dist.Normal(N[0], 1.0))
    hat_N_0 = numpyro.sample("hat_N_0", dist.Normal(0.0, 0.05))
    hat_N_1 = numpyro.sample("hat_N_1", dist.Normal(0.0, 0.05))

    kappa_0 = numpyro.sample("kappa_0", priors["kappa_0"])
    theta_0 = numpyro.sample("theta_0", priors["theta_0"])

    def transition(carry, t):
        Nbar_prev, Nhat_prev_1, Nhat_prev_2 = carry

        # trend
        Nbar_t = numpyro.sample(f"Nbar", dist.Normal(n + Nbar_prev, sigma_eps))

        # cycle
        Nhat_pred = rho_1 * Nhat_prev_1 + rho_2 * Nhat_prev_2
        Nhat_obs = N[t] - Nbar_t
        u_t = Nhat_obs - Nhat_pred
        Nhat_t = numpyro.deterministic(f"Nhat", Nhat_obs)
        numpyro.sample("u_residual", dist.Normal(0.0, sigma_u), obs=u_t)

        # both steady-state and dynamic effects
        kappa_t = kappa_0 + delta * Nbar_t
        theta_t = theta_0 + gamma * Nbar_t
        numpyro.deterministic(f"kappa_{t}", kappa_t)
        numpyro.deterministic(f"theta_{t}", theta_t)

        # x equation
        x_pred = phi_1 * x_prev[t]
        # inflation equation
        pi_pred = (
            alpha * pi_prev[t]
            + (1.0 - alpha) * pi_expect[t]
            + kappa_t * x[t]
            - theta_t * Nhat_t
        )
        numpyro.sample(f"pi_obs_{t}", dist.Normal(pi_pred, sigma_e), obs=pi[t])
        numpyro.sample(f"x_obs_{t}", dist.Normal(x_pred, sigma_zeta), obs=x[t])
        return (Nbar_t, Nhat_t, Nhat_prev_1), None

    timesteps = jnp.arange(1, l)
    init_carry = (bar_N_0, hat_N_0, hat_N_1)
    scan(transition, init_carry, timesteps)