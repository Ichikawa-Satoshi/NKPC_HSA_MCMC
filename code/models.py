import pandas as pd
import numpy as np
import re
import jax
from IPython.display import Markdown, display
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm
import arviz as az
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import numpyro
from jax import random
import numpyro.distributions as dist
import matplotlib.dates as mdates
from datetime import datetime
from numpyro.contrib.control_flow import scan

import warnings
warnings.simplefilter('ignore')
import seaborn as sns



def set_prior_distributions():
    priors = {
        # NKPC params
        "alpha"      : dist.Uniform(0,1), 
        "kappa"      : dist.Uniform(0,1), 
        "theta"      : dist.Uniform(0,1), 
        "eta"        : dist.Uniform(0,1), 

        "phi"        : dist.Normal(0, 1),
        "psi"        : dist.Normal(0, 1),
        "sigma"      : dist.Normal(0, 1),
        "delta"      : dist.Normal(0, 1), 
        "beta"       : dist.Normal(0, 1),
        
        "n"          : dist.Normal(0, 1),  # Nbar trend
        "g"          : dist.Normal(0, 1),  # Y trend
        # initial
        "z_init"     : dist.Normal(0, 0.5), # z init
        "kappa_init" : dist.Normal(0, 0.5), # kappa init
        # Sigma
        "sigma_u"    : dist.LogNormal(0,1),  
        "sigma_eps"  : dist.LogNormal(0,1),  
        "sigma_omega": dist.LogNormal(0,1),  
        "sigma_chi"  : dist.LogNormal(0,1),  
        "sigma_v"    : dist.LogNormal(0,1),
        "sigma_mu"   : dist.LogNormal(0,1),

        "theta_z"    : dist.Normal(0, 0.5), # z coeff
        "theta_pi"   : dist.Normal(0, 0.5), # pi coeff
        "theta_N"    : dist.Normal(0, 0.5), # N coeff
        "theta_Y"    : dist.Normal(0, 0.5), # Y coeff
    }
    return priors









    



