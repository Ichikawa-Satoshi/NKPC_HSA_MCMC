clear; clc;
% Load data
data = func_load_data();
% set variables (pi = core cpi, Epi = Spf cpi)
pi = data.pi_cpi;
Epi = data.Epi_spf_cpi;
pi_prev = data.pi_cpi_prev;
N = data.N;
Nhat = data.N_BN;
%% ============================================================================================
disp("=== OLS CES ===================================================================================");
[ols_ces, mods2] = func_ols_ces_system(data, pi, Epi, pi_prev);
disp(ols_ces)
save('results/ols_ces.mat', 'ols_ces');
disp("=== OLS HSA ===================================================================================");
[ols_hsa, mods3] = func_ols_hsa_system(data, pi, Epi, pi_prev, Nhat);
disp(ols_hsa)
save('results/ols_hsa.mat', 'ols_hsa');

pi = data.pi_cpi_core;
Epi = data.Epi_spf_cpi;
pi_prev = data.pi_cpi_core_prev;
%% ============================================================================================
disp("=== OLS CES ===================================================================================");
[ols_ces_core, mods2_core] = func_ols_ces_system(data, pi, Epi, pi_prev);
disp(ols_ces_core)
save('results/ols_ces_core.mat', 'ols_ces_core');
disp("=== OLS HSA ===================================================================================");
[ols_hsa_core, mods3_core] = func_ols_hsa_system(data, pi, Epi, pi_prev, Nhat);
disp(ols_hsa_core)
save('results/ols_hsa_core.mat', 'ols_hsa_core');