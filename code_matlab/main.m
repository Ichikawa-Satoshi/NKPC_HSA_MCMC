clear; clc;
% Load data
data = func_load_data();
% set variables (pi = core cpi, Epi = Spf cpi)
pi = data.pi_cpi;
Epi = data.Epi_spf_cpi;
pi_prev = data.pi_cpi_prev;
N = data.N;
Nhat = data.N_BN;
oil = data.oil;
%% ============================================================================================
disp("=== OLS CES ===================================================================================");
[ols_ces, mods2] = func_ols_ces_system(data, pi, Epi, pi_prev, oil, false);
disp(ols_ces)
save('results/ols_ces.mat', 'ols_ces');
disp("============================================================================================")
disp("=== OLS HSA ===================================================================================");
[ols_hsa, mods3] = func_ols_hsa_system(data, pi, Epi, pi_prev, Nhat, oil, false);
disp(ols_hsa)
save('results/ols_hsa.mat', 'ols_hsa');
disp("============================================================================================")
%% ============================================================================================
% Setting for Gibbs Sampling 
% Gibbs Sampling (HSA)
priors.mu_alpha=0;   priors.sigma_alpha=1;
priors.mu_kappa=0;   priors.sigma_kappa=1;
priors.mu_theta=0;   priors.sigma_theta=1;
priors.a_v=0.001; priors.b_v=0.001; 
priors.a_eps=0.001; priors.b_eps=0.001; 
priors.a_eta=0.001; priors.b_eta=0.001;
priors.mu_rho1 = 0.5;  sigma_rho1  = 1;
priors.mu_rho2 = -0.5; sigma_rho2  = 1;
priors.mu_n = 0; priors.sigma_n = 1;
priors.mu_x_star = 0; sigma_x_star   = 1;
priors.a_x_obs = 0.001; b_x_obs = 0.001;
opts.seed=123; opts.constrain_alpha=true; opts.enforce_stationary=true;
opts.store_every=2; opts.r_target_scale=0.05; opts.r_rw_scale=0.05;

% sampling
sample = 9000;
burn_in = 1000;
%% ============================================================================================
disp("===Gibbs Sampling (CES)====================================");
disp("Output gap = GDPC1 (BN)")
x = data.output_gap_BN;
nkpc_ces_outputgap = func_nkpc_ces(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
save('results/nkpc_ces_outputgap.mat', 'nkpc_ces_outputgap');
disp("Output gap = Unemployment gap")
x = data.unemp_gap;
nkpc_ces_unempgap = func_nkpc_ces(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
save('results/nkpc_ces_unempgap.mat', 'nkpc_ces_unempgap');
disp("Output gap = Inverse of markup (BN)")
x = data.markup_BN_inv;
nkpc_ces_markupinv = func_nkpc_ces(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
save('results/nkpc_ces_markupinv.mat', 'nkpc_ces_markupinv');
disp("===Gibbs Sampling (CES with the measurement error of the output gap)====================================");
disp("Output gap = GDPC1 (BN)")
x = data.output_gap_BN;
nkpc_ces_outputgap_xerror = func_nkpc_ces_xerror(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
save('results/nkpc_ces_outputgap_xerror.mat', 'nkpc_ces_outputgap_xerror');
disp("Output gap = Unemployment gap")
x = data.unemp_gap;
nkpc_ces_unempgap_xerror = func_nkpc_ces_xerror(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
save('results/nkpc_ces_unempgap_xerror.mat', 'nkpc_ces_unempgap_xerror');
disp("Output gap = Inverse of markup (BN)")
x = data.markup_BN_inv;
nkpc_ces_markupinv_xerror = func_nkpc_ces_xerror(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
save('results/nkpc_ces_markupinv_xerror.mat', 'nkpc_ces_markupinv_xerror');
disp("============================================================================================")
%% ============================================================================================
disp("===Gibbs Sampling (HSA)====================");
disp("Output gap = GDPC1 (BN)")
x = data.output_gap_BN;
nkpc_hsa_outputgap = func_nkpc_hsa(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
save('results/nkpc_hsa_outputgap.mat', 'nkpc_hsa_outputgap');
disp("Output gap = Unemployment gap")
x = data.unemp_gap;
nkpc_hsa_unempgap = func_nkpc_hsa(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
save('results/nkpc_hsa_unempgap.mat', 'nkpc_hsa_unempgap');
disp("Output gap = Inverse of markup (BN)")
x = data.markup_BN_inv;
nkpc_hsa_markupinv = func_nkpc_hsa(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
save('results/nkpc_hsa_markupinv.mat', 'nkpc_hsa_markupinv');

disp("===Gibbs Sampling (HSA with the measurement error of output gap)===================");
disp("Output gap = GDPC1 (BN)")
x = data.output_gap_BN;
nkpc_hsa_outputgap_xerror = func_nkpc_hsa_xerror(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
save('results/nkpc_hsa_outputgap_xerror.mat', 'nkpc_hsa_outputgap_xerror');
disp("Output gap = Unemployment gap")
x = data.unemp_gap;
nkpc_hsa_unempgap_xerror = func_nkpc_hsa_xerror(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
save('results/nkpc_hsa_unempgap_xerror.mat', 'nkpc_hsa_unempgap_xerror');
disp("Output gap = Inverse of markup (BN)")
x = data.markup_BN_inv;
nkpc_hsa_markupinv_xerror = func_nkpc_hsa_xerror(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
save('results/nkpc_hsa_markupinv_xerror.mat', 'nkpc_hsa_markupinv_xerror');
disp("============================================================================================")
%% ============================================================================================
disp("===Gibbs Sampling (HSA joint trend-cycle decomposition)===============================");
disp("Output gap = GDPC1 (BN)")
x = data.output_gap_BN;
nkpc_hsa_decomp_outputgap = func_nkpc_hsa_decomp(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
save('results/nkpc_hsa_decomp_outputgap.mat', 'nkpc_hsa_decomp_outputgap');
disp("Output gap = Unemployment gap")
x = data.unemp_gap;
nkpc_hsa_decomp_unemp_gap = func_nkpc_hsa_decomp(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
save('results/nkpc_hsa_decomp_unemp_gap.mat', 'nkpc_hsa_decomp_unemp_gap');
disp("Output gap = Inverse of markup (BN)")
x = data.markup_BN_inv;
nkpc_hsa_decomp_markupinv = func_nkpc_hsa_decomp(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
save('results/nkpc_hsa_decomp_markupinv.mat', 'nkpc_hsa_decomp_markupinv');

disp("===Gibbs Sampling (HSA joint trend-cycle decomposition with measurement error of output gap) =================");
disp("Output gap = GDPC1 (BN)")
x = data.output_gap_BN;
nkpc_hsa_decomp_outputgap_xerror = func_nkpc_hsa_decomp_xerror(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
save('results/nkpc_hsa_decomp_outputgap_xerror.mat', 'nkpc_hsa_decomp_outputgap_xerror');
disp("============================================================================================")
disp("Output gap = Unemployment gap")
x = data.unemp_gap;
nkpc_hsa_decomp_unemp_gap_xerror = func_nkpc_hsa_decomp_xerror(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
save('results/nkpc_hsa_decomp_unemp_gap_xerror.mat', 'nkpc_hsa_decomp_unemp_gap_xerror');
disp("============================================================================================")
disp("Output gap = Inverse of markup (BN)")
x = data.markup_BN_inv;
nkpc_hsa_decomp_markupinv_xerror = func_nkpc_hsa_decomp_xerror(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
save('results/nkpc_hsa_decomp_markupinv_xerror.mat', 'nkpc_hsa_decomp_markupinv_xerror');
disp("Done!!!!!!!")
disp("============================================================================================")