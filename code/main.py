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
        # NKPC params
        "beta"       : dist.Uniform(0, 1),
        "kappa"      : dist.Gamma(concentration=2, rate=10),
        "theta"      : dist.Gamma(concentration=2, rate=10),
        # State equation (z) params
        "gamma"      : dist.Gamma(concentration=2, rate=10),
        "delta"      : dist.Gamma(concentration=2, rate=10),   
        "rho"        : dist.Normal(0, 0.2),
        # State equation (kappa) params
        "rho_kappa"  : dist.Normal(0, 0.2),
        "rho_z"      : dist.Normal(0, 0.2),
        # initial
        "z_init"     : dist.Normal(0, 0.2),
        "kappa_init" : dist.Gamma(concentration=2, rate=10),
        # Sigma
        "sigma_eps"  : dist.HalfCauchy(scale=1),
        "sigma_kappa": dist.HalfCauchy(scale=1),
        "sigma_eta"  : dist.HalfCauchy(scale=1)
    }
    return priors


# normal NKPC
def model_0_0(pi, pi_expect, Y):
    priors = set_prior_distributions()
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    pi_pred = beta * pi_expect + kappa * Y
    numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi)

def model_0_1(pi, pi_prev, pi_expect, Y):
    priors = set_prior_distributions()
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    # Sigma
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    # model
    pi_pred = alpha * pi_prev + beta * pi_expect + kappa * Y
    numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi)

#--------------------------------------------------------------------------------
# z_{t-1}
def model_1(pi, pi_expect, Y, l):
    priors = set_prior_distributions()
    # NKPC params
    beta  = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    # State equation (z) params
    gamma = numpyro.sample("gamma", priors["gamma"])
    theta = numpyro.sample("theta", priors["theta"])
    # Sigma
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    sigma_eps = numpyro.sample("sigma_e", priors["sigma_eps"])
    # inital
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

def model_2(pi, pi_prev, pi_expect, Y, l):
    priors = set_prior_distributions()
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    kappa = numpyro.sample("kappa", priors["kappa"])
    # State equation (z) params
    gamma = numpyro.sample("gamma", priors["gamma"])
    theta = numpyro.sample("theta", priors["theta"])
    # Sigma
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])  
    # inital z
    z_init = numpyro.sample("z_init", priors["z_init"])
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
    priors = set_prior_distributions()
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    # State equation (z) params
    rho = numpyro.sample("rho", priors["rho"])
    # Sigma
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    # initial
    z_init = numpyro.sample("z_init", priors["z_init"])
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
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    # State equation (z) params
    rho = numpyro.sample("rho", priors["rho"])
    # Sigma
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    # initial 
    z_init = numpyro.sample("z_init", priors["z_init"])
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
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    # State equation (z) params
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho = numpyro.sample("rho", priors["rho"])
    # Sigma
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    # initial
    z_init = numpyro.sample("z_init", priors["z_init"])
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
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta = numpyro.sample("theta", priors["theta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    # State equation (z) params
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho = numpyro.sample("rho", priors["rho"])
    # Sigma
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    # initial
    z_init = numpyro.sample("z_init", priors["z_init"])
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
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    theta = numpyro.sample("theta", priors["theta"])
    # State equation (z) params
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho = numpyro.sample("rho", priors["rho"])
    delta = numpyro.sample("delta", priors["delta"])
    # Sigma
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    # initial 
    z_init = numpyro.sample("z_init", priors["z_init"])
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
    # NKPC params
    beta = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta = numpyro.sample("theta", priors["theta"])
    kappa = numpyro.sample("kappa", priors["kappa"])
    # State equation (z) params
    gamma = numpyro.sample("gamma", priors["gamma"])
    delta = numpyro.sample("delta", priors["delta"])
    rho = numpyro.sample("rho", priors["rho"])
    # Sigma
    sigma_eps = numpyro.sample("sigma_eps", priors["sigma_eps"])
    sigma_eta = numpyro.sample("sigma_eta", priors["sigma_eta"])
    # initial 
    z_init = numpyro.sample("z_init", priors["z_init"])
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

#--------------------------------------------------------------------------------
# time varient (AR1) kappa
#--------------------------------------------------------------------------------
# z_{t-1}
def model_t_1(pi, pi_expect, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_t_2(pi, pi_prev, pi_expect, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)
#--------------------------------------------------------------------------------
# Y_{t-1}
def model_t_3(pi, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho     = numpyro.sample("rho",     priors["rho"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(rho * Y_prev[t], sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_t_4(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho     = numpyro.sample("rho",     priors["rho"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(rho + Y_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)
#--------------------------------------------------------------------------------
# Z_{t-1}, Y_{t-1}
def model_t_5(pi, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho     = numpyro.sample("rho",     priors["rho"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t], sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_t_6(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho     = numpyro.sample("rho",     priors["rho"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho + Y_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

#--------------------------------------------------------------------------------
# Z_{t-1}, Y_{t-1}, π_{t-1}
def model_t_7(pi, pi_prev, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho     = numpyro.sample("rho",     priors["rho"])
    delta    = numpyro.sample("delta",     priors["delta"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t] + delta * pi_prev[t], sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_t_8(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho     = numpyro.sample("rho",     priors["rho"])
    delta    = numpyro.sample("delta",  priors["delta"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t] + delta * pi_prev[t], sigma_eta))
        pi_pred =alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)



#--------------------------------------------------------------------------------
# time varient z_{t-1}
#--------------------------------------------------------------------------------
# z_{t-1}
def model_z_1(pi, pi_expect, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    # state equation params(kappa)
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    # state equation params(z)
    z_init = numpyro.sample("z_init", priors["z_init"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        t = carry[2]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_z_2(pi, pi_prev, pi_expect, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    # state equation params(kappa)
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    # state equation params(z)
    z_init = numpyro.sample("z_init", priors["z_init"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])

    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        t = carry[2]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)
#--------------------------------------------------------------------------------
# Y_{t-1}
def model_z_3(pi, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho     = numpyro.sample("rho",     priors["rho"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        t = carry[2]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t], sigma_eta))
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_z_4(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho     = numpyro.sample("rho",     priors["rho"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho + Y_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)
#--------------------------------------------------------------------------------
# Z_{t-1}, Y_{t-1}
def model_z_5(pi, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    rho     = numpyro.sample("rho",     priors["rho"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t], sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_z_6(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho     = numpyro.sample("rho",     priors["rho"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho + Y_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

#--------------------------------------------------------------------------------
# Z_{t-1}, Y_{t-1}, π_{t-1}
def model_z_7(pi, pi_prev, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    rho     = numpyro.sample("rho",     priors["rho"])
    delta    = numpyro.sample("delta",     priors["delta"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t] + delta * pi_prev[t], sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_z_8(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho     = numpyro.sample("rho",     priors["rho"])
    delta    = numpyro.sample("delta",  priors["delta"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t] + delta * pi_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)


#--------------------------------------------------------------------------------
# time varient (AR1) kappa + z
#--------------------------------------------------------------------------------
# z_{t-1}
def model_tz_1(pi, pi_expect, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    # state equation params(kappa)
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    # state equation params(z)
    z_init = numpyro.sample("z_init", priors["z_init"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_tz_2(pi, pi_prev, pi_expect, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    # state equation params(kappa)
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    # state equation params(z)
    z_init = numpyro.sample("z_init", priors["z_init"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])

    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev, sigma_eta))
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)
#--------------------------------------------------------------------------------
# Y_{t-1}
def model_tz_3(pi, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho     = numpyro.sample("rho",     priors["rho"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho * Y_prev[t], sigma_eta))
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_tz_4(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    rho     = numpyro.sample("rho",     priors["rho"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(rho + Y_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)
#--------------------------------------------------------------------------------
# Z_{t-1}, Y_{t-1}
def model_tz_5(pi, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    rho     = numpyro.sample("rho",     priors["rho"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t], sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_tz_6(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho     = numpyro.sample("rho",     priors["rho"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho + Y_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

#--------------------------------------------------------------------------------
# Z_{t-1}, Y_{t-1}, π_{t-1}
def model_tz_7(pi, pi_prev, pi_expect, Y_prev, Y, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    rho     = numpyro.sample("rho",     priors["rho"])
    delta    = numpyro.sample("delta",     priors["delta"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t] + delta * pi_prev[t], sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)

def model_tz_8(pi, pi_prev, pi_expect, Y, Y_prev, l):
    priors = set_prior_distributions()
    beta   = numpyro.sample("beta", priors["beta"])
    alpha = 1 - beta
    numpyro.deterministic("alpha",alpha)
    theta  = numpyro.sample("theta", priors["theta"])
    kappa_init = numpyro.sample("kappa_init", priors["kappa"])
    z_init = numpyro.sample("z_init", priors["z_init"])
    # state equation params
    rho_kappa = numpyro.sample("rho_kappa", priors["rho_kappa"])
    rho_z = numpyro.sample("rho_z", priors["rho_z"])
    gamma = numpyro.sample("gamma", priors["gamma"])
    rho     = numpyro.sample("rho",     priors["rho"])
    delta    = numpyro.sample("delta",  priors["delta"])
    # sigma
    sigma_eta   = numpyro.sample("sigma_eta",   priors["sigma_eta"])
    sigma_eps   = numpyro.sample("sigma_e",     priors["sigma_eps"])
    sigma_kappa = numpyro.sample("sigma_kappa", priors["sigma_kappa"])
    timesteps = jnp.arange(l)
    t = 0
    # state space model
    def transition(carry, _):
        # t-1
        z_prev = carry[0]
        kappa_prev = carry[1]
        t = carry[2]
        kappa = numpyro.sample("kappa", numpyro.distributions.TruncatedNormal(rho_kappa * kappa_prev + rho_z * z_prev, sigma_kappa, low=0))
        z = numpyro.sample("z", numpyro.distributions.Normal(gamma * z_prev + rho * Y_prev[t] + delta * pi_prev[t], sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        kappa_carry = kappa
        return [z_carry, kappa_carry, t_carry], None
    scan(transition, [z_init, kappa_init, t], timesteps)