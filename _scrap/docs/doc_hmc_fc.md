
# HMC
Evaluate models with
    - Savage-Dickey Density ratio for ($\theta = 0 \text{ or } \theta \neq 0$)
    The Savage–Dickey density ratio is defined as
    $$
    BF_{01} = \frac{p(\theta = \theta_0 \mid \text{prior})}{p(\theta = \theta_0 \mid \text{posterior})},
    $$
    where $BF_{01}$ quantifies the evidence in favor of $H_0$ relative to $H_1$.
    Values of $BF_{01} < 1$ indicate evidence for $H_1$, while values of $BF_{01} > 1$ indicate evidence for $H_0$.
    $H_0$: $\theta = \theta_0$, $H_1$: $\theta \neq 0$

    - WAIC

# Estimation
### CES (Benchmark)
$$
\pi_{t}=\alpha\pi_{t-1}+(1-\alpha)\mathbb{E}_{t}\pi_{t-1}+\kappa x_{t}+v_{t}
$$
- Priors: 
    - $\alpha \sim N(0.5, 0.1)$  
    - $\kappa \sim N(0, 0.1)$  
    - $\sigma_v \sim InvGamma(0.001, 0.001)$  

#### Posterior vs Prior
![alt text](figure/ces_results_hmc_fc.png)

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
    - $\kappa \sim N(0, 0.1)$  
    - $\theta \sim N(0, 0.1)$  
    - $\sigma_v \sim InvGamma(0.001, 0.001)$  

#### Posterior vs Prior
![alt text](figure/hsa_results_hmc_fc.png)

### Estimated coefficients

![text](estimated_coef/results_table_hmc_fc.md)

### Savage-Dickey Density Ratio
- CES ($\kappa$)
![text](estimated_coef/ces_sddr_hmc.md)

- HSA ($\kappa$ & $\theta$)
![text](estimated_coef/hsa_sddr_hmc.md)

### WAIC
![alt text](figure/waic_hmc_post.png) 
![alt text](figure/waic_hmc_pre.png)

### Check N Decomposition
![alt text](figure/decomposition_hmc_fc.png)
