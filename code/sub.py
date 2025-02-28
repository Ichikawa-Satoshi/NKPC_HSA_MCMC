import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.control_flow import scan
from tqdm import tqdm
import arviz as az
import matplotlib.pyplot as plt
from IPython.display import display, Math
import warnings
warnings.simplefilter('ignore')


def set_prior_distributions():
    priors = {
        "beta" : dist.Uniform(0, 1),
        "kappa": dist.Gamma(concentration=2, rate=10),
        "theta": dist.Gamma(concentration=2, rate=10),
        "gamma": dist.Gamma(concentration=2, rate=10),
        "rho"  : dist.TruncatedNormal(-0.5, 0.2, high=0),
        "delta": dist.Gamma(concentration=2, rate=10),
        "sigma_eps": dist.HalfCauchy(scale=0.5),
        "sigma_eta": dist.HalfCauchy(scale=0.5),
        "sigma_kappa": dist.HalfCauchy(scale=0.5)
    }
    return priors
#--------------------------------------------------------------------------------
# z_{t-1}
def model_1(pi, pi_expect, Y, l):
    # Parameters
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    rho_kappa = numpyro.sample("rho_kappa", priors["kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    theta = numpyro.sample("theta", priors["theta"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    sigma_eps = numpyro.sample("sigma_e", priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])

    # inital for time varient
    z_init = 1
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    timesteps = jnp.arange(l)
    t = 0
    
    # state space model
    def transition(carry, _):
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.Normal(rho_kappa * kappa_prev, sigma_kappa))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        return [z_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_2(pi, pi_prev, pi_expect, Y, l):
    # Parameters
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    kappa = numpyro.sample("kappa", priors["kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    theta = numpyro.sample("theta", priors["theta"])
    # state space model
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    
    z_init = 1
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)
#--------------------------------------------------------------------------------
# Y_{t-1}
def model_3(pi, pi_expect, Y, Y_prev, l):
    # Parameters
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    rho = numpyro.sample("rho", priors["rho"])
    # error terms
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    # initial z
    z_init = 1
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)


def model_4(pi, pi_prev , pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    rho = numpyro.sample("rho", priors["rho"])

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    
    # initial z
    z_init = 1
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)
#--------------------------------------------------------------------------------
# z_{t-1} + Y_{t-1}
def model_5(pi, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho = numpyro.sample("rho", priors["rho"])

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    
    # initial z
    z_init = 1
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

def model_6(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta = numpyro.sample("theta", priors["theta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho = numpyro.sample("rho", priors["rho"])

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    
    # initial z
    z_init = 1
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

#--------------------------------------------------------------------------------
# z_{t-1} + Y_{t-1} + π_{t-1}
def model_7(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho = numpyro.sample("rho", priors["rho"])
    delta = numpyro.sample("delta", priors["delta"])

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    
    # initial z
    z_init = 1
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + delta * pi_prev[t] + rho * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

def model_8(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta = numpyro.sample("theta", priors["theta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    delta = numpyro.sample("delta", priors["delta"])
    rho = numpyro.sample("rho", priors["rho"])

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    
    # initial z
    z_init = 1
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + delta * pi_prev[t] + rho * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)
