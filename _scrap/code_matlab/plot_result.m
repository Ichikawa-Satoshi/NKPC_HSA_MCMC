clear all;
% s = [0.1, 0.2, 0.5, 1, 10, 100];
s = 1;
load(sprintf('results/nkpc_ces_outputgap_sigma%.2f.mat', s),'nkpc_ces_outputgap');
load(sprintf('results/nkpc_ces_unempgap_sigma%.2f.mat', s),'nkpc_ces_unempgap');
load(sprintf('results/nkpc_ces_markupinv_sigma%.2f.mat', s),'nkpc_ces_markupinv');
results_list_ces = {nkpc_ces_outputgap, nkpc_ces_unempgap, nkpc_ces_markupinv};
model_names_ces  = {'Output gap (BN)','Unemployment gap','Markup^{-1} (BN)'};
out = func_plot_posterior(results_list_ces, model_names_ces, ...
       struct('export_path','results/figs/ces_grid.png','show_alpha_bounds',true));
func_plot_trace(results_list_ces, model_names_ces);
%%
load(sprintf('results/nkpc_ces_outputgap_xerror_sigma%.2f.mat', s),'nkpc_ces_outputgap_xerror');
load(sprintf('results/nkpc_ces_unempgap_xerror_sigma%.2f.mat', s),'nkpc_ces_unempgap_xerror');
load(sprintf('results/nkpc_ces_markupinv_xerror_sigma%.2f.mat', s),'nkpc_ces_markupinv_xerror');
results_list_ces_xerror = {nkpc_ces_outputgap, nkpc_ces_unempgap, nkpc_ces_markupinv};
model_names_ces_xerror  = {'Output gap (BN)','Unemployment gap','Markup^{-1} (BN)'};
out = func_plot_posterior(results_list_ces_xerror, model_names_ces_xerror, ...
       struct('export_path','results/figs/ces_xerror_grid.png','show_alpha_bounds',true));
func_plot_trace(results_list_ces_xerror, model_names_ces_xerror);
%%
load(sprintf('results/nkpc_hsa_outputgap_sigma%.2f.mat', s),'nkpc_hsa_outputgap');
load(sprintf('results/nkpc_hsa_unempgap_sigma%.2f.mat', s),'nkpc_hsa_unempgap');
load(sprintf('results/nkpc_hsa_markupinv_sigma%.2f.mat', s),'nkpc_hsa_markupinv');
results_list = {nkpc_hsa_outputgap, nkpc_hsa_unempgap, nkpc_hsa_markupinv};
model_names  = {'Output gap (BN)','Unemployment gap','Markup^{-1} (BN)'};
out = func_plot_posterior(results_list, model_names, ...
       struct('export_path','results/figs/hsa_grid.png','show_alpha_bounds',true));
func_plot_trace(results_list, model_names);
%%
load(sprintf('results/nkpc_hsa_outputgap_xerror_sigma%.2f.mat', s),'nkpc_hsa_outputgap_xerror');
load(sprintf('results/nkpc_hsa_unempgap_xerror_sigma%.2f.mat', s),'nkpc_hsa_unempgap_xerror');
load(sprintf('results/nkpc_hsa_markupinv_xerror_sigma%.2f.mat', s),'nkpc_hsa_markupinv_xerror');
results_list_xerror = {nkpc_hsa_outputgap_xerror, nkpc_hsa_unempgap_xerror, nkpc_hsa_markupinv_xerror};
xerror_names = {'Output gap (BN)','Unemployment gap','Markup^{-1} (BN)'};
out_xerror = func_plot_posterior(results_list_xerror, xerror_names, ...
       struct('export_path','results/figs/hsa_grid_xerror.png','show_alpha_bounds',true));
func_plot_trace(results_list_xerror, xerror_names);
%%
load(sprintf('results/nkpc_hsa_decomp_outputgap_sigma%.2f.mat', s),'nkpc_hsa_decomp_outputgap');
load(sprintf('results/nkpc_hsa_decomp_unemp_gap_sigma%.2f.mat', s),'nkpc_hsa_decomp_unemp_gap');
load(sprintf('results/nkpc_hsa_decomp_markupinv_sigma%.2f.mat', s),'nkpc_hsa_decomp_markupinv');
results_list_decomp = {nkpc_hsa_decomp_outputgap, nkpc_hsa_decomp_unemp_gap, nkpc_hsa_decomp_markupinv};
decomp_names = {'Output gap (BN)','Unemployment gap','Markup^{-1} (BN)'};
out_decomp = func_plot_posterior(results_list_decomp, decomp_names, ...
       struct('export_path','results/figs/hsa_grid_decomp.png','show_alpha_bounds',true));
func_plot_trace(results_list_decomp, decomp_names);
%%
load(sprintf('results/nkpc_hsa_decomp_outputgap_xerror_sigma%.2f.mat', s),'nkpc_hsa_decomp_outputgap_xerror');
load(sprintf('results/nkpc_hsa_decomp_unemp_gap_xerror_sigma%.2f.mat', s),'nkpc_hsa_decomp_unemp_gap_xerror');
load(sprintf('results/nkpc_hsa_decomp_markupinv_xerror_sigma%.2f.mat', s),'nkpc_hsa_decomp_markupinv_xerror');

results_list_decomp_xerror = {nkpc_hsa_decomp_outputgap_xerror, nkpc_hsa_decomp_unemp_gap_xerror, nkpc_hsa_decomp_markupinv_xerror};
xerror_names_decomp = {'Output gap (BN)','Unemployment gap','Markup^{-1} (BN)'};
out_decomp_xerror = func_plot_posterior(results_list_decomp_xerror, xerror_names_decomp, ...
       struct('export_path','results/figs/hsa_grid_decomp_xerror.png','show_alpha_bounds',true));
func_plot_trace(results_list_decomp_xerror, xerror_names_decomp);