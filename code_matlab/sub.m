clear; clc;
% Load data
data = func_load_data();
% set variables (pi = core cpi, Epi = Spf cpi)
pi = data.pi_cpi;
Epi = data.Epi_spf_cpi;
pi_prev = data.pi_cpi_prev;
N = data.N;
Nhat = data.N_BN;
Nbar = data.N_BN_trend;
oil = data.oil;
%% ============================================================================================
% Setting for Gibbs Sampling 
% Gibbs Sampling (HSA joint trend-cycle decomposition)
priors.mu_alpha=0.7; priors.sigma_alpha=0.2;
priors.mu_kappa=0;   priors.sigma_kappa=0.2;
priors.mu_theta=0;   priors.sigma_theta=0.2;
priors.mu_beta=0;    priors.sigma_beta=0.2;
priors.a_v=0.001; priors.b_v=0.001; 
priors.a_eps=0.001; priors.b_eps=0.001; 
priors.a_eta=0.001; priors.b_eta=0.001;
priors.a_u = 0.001; priors.b_u=0.001;
priors.mu_rho1 = 0.5;  sigma_rho1  = 1;
priors.mu_rho2 = -0.5; sigma_rho2  = 1;
priors.mu_n = 0; priors.sigma_n = 1;
priors.mu_x_star = 0; sigma_x_star   = 1;
priors.a_x_obs = 0.001; b_x_obs = 0.001;
opts.seed=123; opts.constrain_alpha=true; opts.enforce_stationary=true;
opts.store_every=2; opts.r_target_scale=0.05; opts.r_rw_scale=0.05;
% sampling
sample = 18000;
burn_in = 2000;
%% ============================================================================================
disp("===Gibbs Sampling (HSA joint trend-cycle decomposition)===============================");
% Output gap = GDPC1 (BN)
x = data.output_gap_BN;
nkpc_hsa_decomp_outputgap_tv = func_nkpc_hsa_tv(pi, pi_prev, Epi, x, Nbar, Nhat, burn_in, sample, priors, opts);
x = data.unemp_gap;
nkpc_hsa_decomp_unemp_gap_tv = func_nkpc_hsa_tv(pi, pi_prev, Epi, x, Nbar, Nhat, burn_in, sample, priors, opts);
disp("Done!!!!!!!")
%%
results_list_decomp = {nkpc_hsa_decomp_outputgap_tv};
decomp_names = {'Output gap (BN)'};
out_decomp = func_plot_posterior_tv(results_list_decomp, decomp_names);
results_list_decomp = {nkpc_hsa_decomp_unemp_gap_tv};
decomp_names = {'Unemp Gap'};
out_decomp = func_plot_posterior_tv(results_list_decomp, decomp_names);
%%
func_plot_kappa_tv(nkpc_hsa_decomp_outputgap_tv);
func_plot_kappa_tv(nkpc_hsa_decomp_unemp_gap_tv);