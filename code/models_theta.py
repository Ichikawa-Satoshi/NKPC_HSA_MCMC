import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.control_flow import scan
from tqdm import tqdm
import arviz as az
import matplotlib.pyplot as plt
from IPython.display import display, Math
import warnings
warnings.simplefilter('ignore')

# normal NKPC
def model_0_0(pi, pi_expect, Y):
    # Parameters
    beta = numpyro.sample("beta", numpyro.distributions.Normal(1, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    # state space model
    sigma_eps = numpyro.sample("sigma_e", numpyro.distributions.HalfCauchy(scale=1))
    pi_pred = beta * pi_expect + kappa * Y
    numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi)


def model_0_1(pi, pi_prev, pi_expect, Y):
    # Parameters
    beta = numpyro.sample("beta", numpyro.distributions.Normal(1, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    alpha = numpyro.sample("alpha", numpyro.distributions.Normal(1, 1))
    # state space model
    sigma_eps = numpyro.sample("sigma_e", numpyro.distributions.HalfCauchy(scale=1))
    pi_pred = alpha * pi_prev + beta * pi_expect + kappa * Y
    numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi)


# / AR1 z
def model_1(pi, pi_expect, Y, l):
    # Parameters
    beta = numpyro.sample("beta", numpyro.distributions.Normal(1, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    rho = numpyro.sample("rho", numpyro.distributions.Normal(0,1))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    # state space model
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eps = numpyro.sample("sigma_e", numpyro.distributions.HalfCauchy(scale=1))
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample(f"z", numpyro.distributions.Normal(rho * z_prev, sigma_eta))
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

# prev pi / AR1 z
def model_2(pi, pi_prev, pi_expect, Y, l):
    # Parameters
    alpha = numpyro.sample("alpha", numpyro.distributions.Normal(1))
    beta = numpyro.sample("beta", numpyro.distributions.Normal(1))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    rho = numpyro.sample("rho", numpyro.distributions.Normal(0,1))
    
    # Latent variable parameters
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eps = numpyro.sample("sigma_e", numpyro.distributions.HalfCauchy(scale=1))

    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample(f"z", numpyro.distributions.Normal(rho * z_prev, sigma_eta))
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        z_carry = z
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)


# / AR1 z + Y
def model_3(pi, pi_expect, Y, Y_prev, l):
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    rho1 = numpyro.sample("rho1", numpyro.distributions.Normal(0, 1))
    rho2 = numpyro.sample("rho2", numpyro.distributions.Normal(0, 1))

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho1 * z_prev + rho2 * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

# prev pi / AR1 z + prev Y
def model_4(pi, pi_prev, pi_expect, Y, Y_prev, l):
    alpha = numpyro.sample("alpha", numpyro.distributions.Normal(0, 1))
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    rho1 = numpyro.sample("rho1", numpyro.distributions.Normal(0, 1))
    rho2 = numpyro.sample("rho2", numpyro.distributions.Normal(0, 1))

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho1 * z_prev + rho2 * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

#  / AR1 z + prev Y + prev_pi
def model_5(pi, pi_prev, pi_expect, Y, Y_prev, l):
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    rho1 = numpyro.sample("rho1", numpyro.distributions.Normal(0, 1))
    rho2 = numpyro.sample("rho2", numpyro.distributions.Normal(0, 1))
    rho3 = numpyro.sample("rho3", numpyro.distributions.HalfNormal(0.5))

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho1 * z_prev + rho2 * pi_prev[t] - rho3 * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

# prev_pi / AR1 z + prev Y + prev_pi
def model_6(pi, pi_prev, pi_expect, Y, Y_prev, l):
    alpha = numpyro.sample("alpha", numpyro.distributions.Normal(0, 1))
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    rho1 = numpyro.sample("rho1", numpyro.distributions.Normal(0, 1))
    rho2 = numpyro.sample("rho2", numpyro.distributions.Normal(0, 1))
    rho3 = numpyro.sample("rho3", numpyro.distributions.HalfNormal(0.5))

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho1 * z_prev + rho2 * pi_prev[t] + rho3 * Y_prev[t], sigma_eta))
        z_carry = z
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

#  / AR1 z + Y
def model_7(pi, pi_expect, Y, l):
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    rho1 = numpyro.sample("rho1", numpyro.distributions.Normal(0, 1))
    rho2 = numpyro.sample("rho2", numpyro.distributions.Normal(0,1))

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho1 * z_prev + rho2 * Y[t], sigma_eta))
        z_carry = z
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)

# prev pi / AR1 z + Y
def model_8(pi, pi_prev, pi_expect, Y, l):
    alpha = numpyro.sample("alpha", numpyro.distributions.Normal(0, 1))
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    rho1 = numpyro.sample("rho1", numpyro.distributions.Normal(0, 1))
    rho2 = numpyro.sample("rho2", numpyro.distributions.Normal(0, 1))
    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho1 * z_prev + rho2 * Y[t], sigma_eta))
        z_carry = z
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)


def model_9(pi, pi_expect, Y, l):
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    rho = numpyro.sample("rho", numpyro.distributions.Normal(0,1))

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho * Y[t], sigma_eta))
        z_carry = z
        pi_pred = beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)


def model_10(pi, pi_prev , pi_expect, Y, l):
    beta = numpyro.sample("beta", numpyro.distributions.Normal(0, 1))
    alpha = numpyro.sample("alpha", numpyro.distributions.Normal(0, 1))
    kappa = numpyro.sample("kappa", numpyro.distributions.HalfNormal(0.5))
    theta = numpyro.sample("theta", numpyro.distributions.HalfNormal(0.5))
    rho = numpyro.sample("rho", numpyro.distributions.Normal(0, 1))

    # error terms
    sigma_eps = numpyro.sample("sigma_eps", numpyro.distributions.HalfCauchy(scale=1))
    sigma_eta = numpyro.sample("sigma_eta", numpyro.distributions.HalfCauchy(scale=1))
    
    # initial z
    z_init = numpyro.sample("z_init", numpyro.distributions.Normal(0,1))
    
    # state space model
    timesteps = jnp.arange(l)
    t = 0
    def transition(carry, _):
        z_prev = carry[0]
        t = carry[1]
        z = numpyro.sample("z", numpyro.distributions.Normal(rho * Y[t], sigma_eta))
        z_carry = z
        pi_pred = alpha * pi_prev[t] + beta * pi_expect[t] + kappa * Y[t] - theta * z
        numpyro.sample(f"pi_obs", numpyro.distributions.Normal(pi_pred, sigma_eps), obs=pi[t])
        t_carry = t + 1
        return [z_carry, t_carry], None
    scan(transition, [z_init, t], timesteps)