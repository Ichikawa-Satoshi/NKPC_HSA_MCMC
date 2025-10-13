# NKPC + Endogenous Cost-Push
By Ippei Fujiwara, James Morley, Satoshi Ichikawa

# Model
### NKPC under CES
$s(z_t) = \gamma_{CES}(z_t)^{1−\theta}$
$\theta$ denotes the constant price elasticity, and $\zeta(z) = \theta$ and $\rho(z) = 1$:
$$
\pi_{t}=\alpha\pi_{t-1}+(1-\alpha)\mathbb{E}_{t}\pi_{t-1}+\underset{\kappa}{\underbrace{\frac{\theta-1}{\chi}}}x_{t}+v_{t}
$$
$\chi$: scales the size of the cost
$\theta$: the constant price elasticity,
### NKPC under HSA
$$
\pi_{t}=\alpha\pi_{t-1}+(1-\alpha)\mathbb{E}_{t}\pi_{t+1}+\underset{\kappa}{\underbrace{\frac{\zeta(z)-1}{\chi}}}x_{t}-\underset{\theta}{\underbrace{\frac{1-\rho(z)}{\rho(z)\chi}}}\hat{N}_{t}+v_{t},
$$
$\zeta(z) = 1-\frac{s′(z)z}{s(z)}$ : the price elasticity function $\zeta(z) > 1$
$\rho(z)=[1-\frac{dln(\frac{\zeta(z)}{\zeta(z)-1})}{dln(z)}]^{-1}$   : the pass-through rate function, $1 >\rho(z) > 0$

## Data
#### One year inflation rate (1947:Q1 - 2024:Q4, Quarterly)
FRED (https://fred.stlouisfed.org/series/CPIAUCSL)
Consumer Price Index for All Urban Consumers: All Items in U.S. City Average 

#### One Inflation Expectation (1982Q1-2024, Quarterly)
Fed of Cleveland (https://www.clevelandfed.org/indicators-and-data/inflation-expectations)
![alt text](figure/image.png)
#### Markup (1947:Q1 - 2017:Q4, Annual)
The Cyclical Behavior of the Price-Cost Markup

Christopher J. Nekarda (Board of Governors of the Federal Reserve System)

Valerie A. Ramey (University of California, San Diego and NBER)

paper: https://econweb.ucsd.edu/~vramey/research/markupcyc.pdf

data: https://econweb.ucsd.edu/~vramey/research.html

#### De-trended Markup
BN filter's cycle component of markup

#### Unemployment gap (Quarterly)
Unemployment gap = Non-Accelerating Inflation Rate of Unemployment - Unemployment rate
- Non-accelerating inflation rate of unemployment: https://fred.stlouisfed.org/series/NROU
From  U.S. Congressional Budget Office

- Unemployment rate: https://fred.stlouisfed.org/series/UNRATE
From U.S. Bureau of Labor Statistics  

#### Output and Output gap (1947:Q1 - 2024:Q4, Quarterly) 
- Output: GDPC1 (https://fred.stlouisfed.org/series/GDPC1)

- Output gap : Beveridge-Nelson Filter's cycle component of GDPC1 (https://bnfiltering.com)

### The number of Firm  (inverse of HHI)
- HHI from G Grullon et al. 2019
Are US Industries Becoming More Concentrated?
Review of Finance, https://doi.org/10.1093/rof/rfz007
- extract data from image by chat-gpt
- annual→quarterly with linear interpolation 
![alt text](figure/image-3.png)

# Sampling method
Two types of sampling
- Gibbs Sampler

    Evaluate models with
    - Savage-Dickey Density ratio for ($\theta = 0 \text{ or } \theta \neq 0$)
    The Savage–Dickey density ratio is defined as
    $$
    BF_{01} = \frac{p(\theta = \theta_0 \mid \text{prior})}{p(\theta = \theta_0 \mid \text{posterior})},
    $$
    where $BF_{01}$ quantifies the evidence in favor of $H_0$ relative to $H_1$.
    Values of $BF_{01} < 1$ indicate evidence for $H_1$, while values of $BF_{01} > 1$ indicate evidence for $H_0$.
    $H_0$: $\theta = \theta_0$, $H_1$: $\theta \neq 0$

    - Marginal Likelihood with Chib's method (https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/NormConstants/ChibJASA1995.pdf)

# Estimation (Gibbs Sampling)
### CES (Benchmark)
$$
\pi_{t}=\alpha\pi_{t-1}+(1-\alpha)\mathbb{E}_{t}\pi_{t-1}+\kappa x_{t}+v_{t}
$$
- Priors: 
    - $\alpha \sim N(0.5, 0.1)$  
    - $\kappa \sim N(0, 0.01)$  
    - $\sigma_v \sim InvGamma(0.001, 0.001)$  

#### Posterior vs Prior
![alt text](matlab/fig/ces_sampling_compare.png)

### HSA
$$
\pi_{t}=\alpha\pi_{t-1}+(1-\alpha)\mathbb{E}_{t}\pi_{t+1}+\kappa x_{t}-\theta\hat{N}_{t}+v_{t},\\
N_{t}=\bar{N}_{t}+\hat{N}_{t},\\
\text{where}\\
\hat{N}_{t}=\rho_1 \hat{N}_{t-1}+\rho_2 \hat{N}_{t-2}+\epsilon_{t}\\
\bar{N}_{t}=n+\bar{N}_{t-1}+\eta_{t},
$$
- Priors:
    - $\alpha \sim N(0.5, 0.1)$  
    - $\kappa \sim N(0, 0.01)$  
    - $\theta \sim N(0, 0.01)$  
    - $\sigma_v \sim InvGamma(0.001, 0.001)$  

#### Posterior vs Prior
![alt text](matlab/fig/hsa_sample_compare.png)

### Estimated coefficients

![text](estimated_coef/results_table_fc.md)

### Savage-Dickey Density Ratio
- CES ($\kappa$)
![text](estimated_coef/ces_sddr_table.md)

- HSA ($\kappa$ & $\theta$)
![text](estimated_coef/hsa_sddr_table.md)

### Marginal Likelihood (Chib)
- CES
![text](estimated_coef/ces_ml_table_fc.md)
- HSA
![text](estimated_coef/hsa_ml_table_fc.md)

### Check N Decomposition
![alt text](matlab/fig/decomposition_pre.png)
![alt text](matlab/fig/decomposition_post.png)
