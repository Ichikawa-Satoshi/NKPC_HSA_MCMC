clear; clc;
data = func_load_data();
pi = data.pi_cpi;
Epi = data.Epi_spf_cpi;
pi_prev = data.pi_cpi_prev;
N = data.N;
Nhat = data.N_BN;


opts.seed = 123;
opts.constrain_alpha = true;
opts.enforce_stationary = true;
opts.store_every = 2;
opts.r_target_scale = 0.05;
opts.r_rw_scale = 0.05;

sample = 15000;
burn_in = 5000;
sigma_list = [0.1, 0.2, 0.5, 1, 10, 100];
for s = sigma_list
    fprintf("=== Running Gibbs with sigma = %.2f ===\n", s);
    priors.mu_alpha = 0.5; priors.sigma_alpha = s;
    priors.mu_kappa = 0;   priors.sigma_kappa = s;
    priors.mu_theta = 0;   priors.sigma_theta = s;
    priors.a_v = 2; priors.b_v = 2;
    priors.a_eps = 2; priors.b_eps = 2;
    priors.a_eta = 2; priors.b_eta = 2;
    priors.mu_rho1 = 0.5;  priors.sigma_rho1 = 1;
    priors.mu_rho2 = -0.5; priors.sigma_rho2 = 1;
    priors.mu_n = 0; priors.sigma_n = 1;
    priors.mu_x_star = 0; priors.sigma_x_star = 1;
    priors.a_x_obs = 2; priors.b_x_obs = 2;    
    %% ============================================================================================
    disp("===Gibbs Sampling (CES)====================================");
    % Output gap = GDPC1 (BN)
    x = data.output_gap_BN;
    nkpc_ces_outputgap = func_nkpc_ces(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_ces_outputgap_sigma%.2f.mat', s), 'nkpc_ces_outputgap');
    % Output gap = Unemployment gap 
    x = data.unemp_gap;
    nkpc_ces_unempgap = func_nkpc_ces(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_ces_unempgap_sigma%.2f.mat', s), 'nkpc_ces_unempgap');
    % Output gap = Inverse of markup (BN)
    x = data.markup_BN_inv;
    nkpc_ces_markupinv = func_nkpc_ces(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_ces_markupinv_sigma%.2f.mat', s), 'nkpc_ces_markupinv');
    %% ============================================================================================
    disp("===Gibbs Sampling (CES with the measurement error of the output gap)====================================");
    % Output gap = GDPC1 (BN)
    x = data.output_gap_BN;
    nkpc_ces_outputgap_xerror = func_nkpc_ces_xerror(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_ces_outputgap_xerror_sigma%.2f.mat', s), 'nkpc_ces_outputgap_xerror');
    % Output gap = Unemployment gap 
    x = data.unemp_gap;
    nkpc_ces_unempgap_xerror = func_nkpc_ces_xerror(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_ces_unempgap_xerror_sigma%.2f.mat', s), 'nkpc_ces_unempgap_xerror');
    % Output gap = Inverse of markup (BN)
    x = data.markup_BN_inv;
    nkpc_ces_markupinv_xerror = func_nkpc_ces_xerror(pi, pi_prev, Epi, x, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_ces_markupinv_xerror_sigma%.2f.mat', s), 'nkpc_ces_markupinv_xerror');
    %% ============================================================================================
    disp("===Gibbs Sampling (HSA)====================");
    % Output gap = GDPC1 (BN)
    x = data.output_gap_BN;
    nkpc_hsa_outputgap = func_nkpc_hsa(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_outputgap_sigma%.2f.mat', s), 'nkpc_hsa_outputgap');
    % Output gap = Unemployment gap 
    x = data.unemp_gap;
    nkpc_hsa_unempgap = func_nkpc_hsa(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_unempgap_sigma%.2f.mat', s), 'nkpc_hsa_unempgap');
    % Output gap = Inverse of markup (BN)
    x = data.markup_BN_inv;
    nkpc_hsa_markupinv = func_nkpc_hsa(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_markupinv_sigma%.2f.mat', s), 'nkpc_hsa_markupinv');
    %% ============================================================================================
    disp("===Gibbs Sampling (HSA with the measurement error of output gap)===================");
    % Output gap = GDPC1 (BN)
    x = data.output_gap_BN;
    nkpc_hsa_outputgap_xerror = func_nkpc_hsa_xerror(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_outputgap_xerror_sigma%.2f.mat', s), 'nkpc_hsa_outputgap_xerror');
    % Output gap = Unemployment gap 
    x = data.unemp_gap;
    nkpc_hsa_unempgap_xerror = func_nkpc_hsa_xerror(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_unempgap_xerror_sigma%.2f.mat', s), 'nkpc_hsa_unempgap_xerror');
    % Output gap = Inverse of markup (BN)
    x = data.markup_BN_inv;
    nkpc_hsa_markupinv_xerror = func_nkpc_hsa_xerror(pi, pi_prev, Epi, x, Nhat, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_markupinv_xerror_sigma%.2f.mat', s), 'nkpc_hsa_markupinv_xerror');
    %% ============================================================================================
    disp("===Gibbs Sampling (HSA joint trend-cycle decomposition)===============================");
    % Output gap = GDPC1 (BN)
    x = data.output_gap_BN;
    nkpc_hsa_decomp_outputgap = func_nkpc_hsa_decomp(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_decomp_outputgap_sigma%.2f.mat', s), 'nkpc_hsa_decomp_outputgap');
    % Output gap = Unemployment gap 
    x = data.unemp_gap;
    nkpc_hsa_decomp_unemp_gap = func_nkpc_hsa_decomp(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_decomp_unemp_gap_sigma%.2f.mat', s), 'nkpc_hsa_decomp_unemp_gap');
    % Output gap = Inverse of markup (BN)
    x = data.markup_BN_inv;
    nkpc_hsa_decomp_markupinv = func_nkpc_hsa_decomp(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_decomp_markupinv_sigma%.2f.mat', s), 'nkpc_hsa_decomp_markupinv');
    %% ============================================================================================
    disp("===Gibbs Sampling (HSA joint trend-cycle decomposition with measurement error of output gap) =================");
    % Output gap = GDPC1 (BN)
    x = data.output_gap_BN;
    nkpc_hsa_decomp_outputgap_xerror = func_nkpc_hsa_decomp_xerror(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_decomp_outputgap_xerror_sigma%.2f.mat', s), 'nkpc_hsa_decomp_outputgap_xerror');
    % Output gap = Unemployment gap 
    x = data.unemp_gap;
    nkpc_hsa_decomp_unemp_gap_xerror = func_nkpc_hsa_decomp_xerror(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_decomp_unemp_gap_xerror_sigma%.2f.mat', s), 'nkpc_hsa_decomp_unemp_gap_xerror');
    % Output gap = Inverse of markup (BN)
    x = data.markup_BN_inv;
    nkpc_hsa_decomp_markupinv_xerror = func_nkpc_hsa_decomp_xerror(pi, pi_prev, Epi, x, N, burn_in, sample, priors, opts);
    save(sprintf('results/nkpc_hsa_decomp_markupinv_xerror_sigma%.2f.mat', s), 'nkpc_hsa_decomp_markupinv_xerror');
end
disp("All sigma runs completed!");