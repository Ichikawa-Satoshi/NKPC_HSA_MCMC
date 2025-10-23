%% ============================================================
%  MAKE_HSA_SUMMARY_TABLES
%  Summarize posterior mean and 95% CI for alpha, kappa, theta
%  across model types and output gap definitions
% =============================================================
clear; clc;
% -------- Load all results --------
load('results/ols_ces.mat','ols_ces');
load('results/ols_hsa.mat','ols_hsa');

% s = [0.1, 0.2, 0.5, 1, 10, 100];
s = 100;
load(sprintf('results/nkpc_ces_outputgap_sigma%.2f.mat', s),'nkpc_ces_outputgap');
load(sprintf('results/nkpc_ces_unempgap_sigma%.2f.mat', s),'nkpc_ces_unempgap');
load(sprintf('results/nkpc_ces_markupinv_sigma%.2f.mat', s),'nkpc_ces_markupinv');
results_list_ces = {nkpc_ces_outputgap, nkpc_ces_unempgap, nkpc_ces_markupinv};
model_names_ces  = {'Output gap (BN)','Unemployment gap', 'Inverse of markup (BN)'};
load(sprintf('results/nkpc_ces_outputgap_xerror_sigma%.2f.mat', s),'nkpc_ces_outputgap_xerror');
load(sprintf('results/nkpc_ces_unempgap_xerror_sigma%.2f.mat', s),'nkpc_ces_unempgap_xerror');
load(sprintf('results/nkpc_ces_markupinv_xerror_sigma%.2f.mat', s),'nkpc_ces_markupinv_xerror');
results_list_ces_xerror = {nkpc_ces_outputgap, nkpc_ces_unempgap, nkpc_ces_markupinv};
model_names_ces_xerror  = {'Output gap (BN)','Unemployment gap', 'Inverse of markup (BN)'};
load(sprintf('results/nkpc_hsa_outputgap_sigma%.2f.mat', s),'nkpc_hsa_outputgap');
load(sprintf('results/nkpc_hsa_unempgap_sigma%.2f.mat', s),'nkpc_hsa_unempgap');
load(sprintf('results/nkpc_hsa_markupinv_sigma%.2f.mat', s),'nkpc_hsa_markupinv');
results_list = {nkpc_hsa_outputgap, nkpc_hsa_unempgap, nkpc_hsa_markupinv};
model_names  = {'Output gap (BN)','Unemployment gap', 'Inverse of markup (BN)'};
load(sprintf('results/nkpc_hsa_outputgap_xerror_sigma%.2f.mat', s),'nkpc_hsa_outputgap_xerror');
load(sprintf('results/nkpc_hsa_unempgap_xerror_sigma%.2f.mat', s),'nkpc_hsa_unempgap_xerror');
load(sprintf('results/nkpc_hsa_markupinv_xerror_sigma%.2f.mat', s),'nkpc_hsa_markupinv_xerror');
results_list_xerror = {nkpc_hsa_outputgap_xerror, nkpc_hsa_unempgap_xerror, nkpc_hsa_markupinv_xerror};
xerror_names = {'Output gap (BN)','Unemployment gap', 'Inverse of markup (BN)'};
load(sprintf('results/nkpc_hsa_decomp_outputgap_sigma%.2f.mat', s),'nkpc_hsa_decomp_outputgap');
load(sprintf('results/nkpc_hsa_decomp_unemp_gap_sigma%.2f.mat', s),'nkpc_hsa_decomp_unemp_gap');
load(sprintf('results/nkpc_hsa_decomp_markupinv_sigma%.2f.mat', s),'nkpc_hsa_decomp_markupinv');
results_list_decomp = {nkpc_hsa_decomp_outputgap, nkpc_hsa_decomp_unemp_gap, nkpc_hsa_decomp_markupinv};
decomp_names = {'Output gap (BN)','Unemployment gap', 'Inverse of markup (BN)'};
load(sprintf('results/nkpc_hsa_decomp_outputgap_xerror_sigma%.2f.mat', s),'nkpc_hsa_decomp_outputgap_xerror');
load(sprintf('results/nkpc_hsa_decomp_unemp_gap_xerror_sigma%.2f.mat', s),'nkpc_hsa_decomp_unemp_gap_xerror');
load(sprintf('results/nkpc_hsa_decomp_markupinv_xerror_sigma%.2f.mat', s),'nkpc_hsa_decomp_markupinv_xerror');
results_list_decomp_xerror = {nkpc_hsa_decomp_outputgap_xerror, nkpc_hsa_decomp_unemp_gap_xerror, nkpc_hsa_decomp_markupinv_xerror};
xerror_names_decomp = {'Output gap (BN)','Unemployment gap', 'Inverse of markup (BN)'};
%% -------- Helper function --------
function T = summarize_ols_for_merge(ols_data, model_label)
    if istable(ols_data)
        C = table2cell(ols_data);
    elseif iscell(ols_data)
        C = ols_data;
    else
        error('ols_data must be a table or cell array.');
    end
    if size(C,2) < 7
        error('OLS data needs at least 7 columns: {Gap, Var, Coef, SE, p, R2, AdjR2}');
    end

    Gap = string(C(:,1));
    Var = string(C(:,2));
    Coef = cellfun(@double, C(:,3));
    SE   = cellfun(@double, C(:,4));
    pval = cellfun(@double, C(:,5)); 
    % R2   = cellfun(@double, C(:,6));
    % AdjR2= cellfun(@double, C(:,7));

    % -------------
    gaps = unique(Gap, 'stable');
    nG   = numel(gaps);

    ModelType   = repmat(string(model_label), nG, 1);
    GapMeasure  = gaps;
    Alpha95     = strings(nG,1);
    Kappa95     = strings(nG,1);
    Theta95     = strings(nG,1);
    Alpha_sd    = NaN(nG,1);
    Kappa_sd    = NaN(nG,1);
    Theta_sd    = NaN(nG,1);    
    for g = 1:nG
        rows = (Gap == gaps(g));
        % alpha: pi_prev
        r = find(rows & Var == "pi_prev", 1, 'first');
        if ~isempty(r)
            Alpha95(g)  = sprintf('%.3f (%.3f)', Coef(r), pval(r));
            Alpha_sd(g) = SE(r);
        else
            Alpha95(g)  = "—";
        end
        % kappa: x
        r = find(rows & Var == "x", 1, 'first');
        if ~isempty(r)
            Kappa95(g)  = sprintf('%.3f (%.3f)', Coef(r), pval(r));
            Kappa_sd(g) = SE(r);
        else
            Kappa95(g)  = "—";
        end

        % theta: Nhat
        r = find(rows & Var == "Nhat", 1, 'first');
        if ~isempty(r)
            Theta95(g)  = sprintf('%.3f (%.3f)', -1*Coef(r), pval(r));
            Theta_sd(g) = SE(r);
        else
            Theta95(g)  = "—";
        end
    end

    % ---- Bayes ----
    T = table(ModelType, GapMeasure, Alpha95, Kappa95, Theta95, ...
              Alpha_sd, Kappa_sd, Theta_sd, ...
              'VariableNames', {'ModelType','GapMeasure','Alpha(p-value)','Kappa(p-value/SDDR)','Theta(p-value/SDDR)', ...
         'Alpha_sd','Kappa_sd','Theta_sd'});
end
function T = summarize_results(results_list, names, label, opts)

    if nargin < 4, opts = struct; end
    if ~isfield(opts,'null_kappa'), opts.null_kappa = 0; end
    if ~isfield(opts,'null_theta'), opts.null_theta = 0; end
    if ~isfield(opts,'null_alpha'), opts.null_alpha = []; end 

    n = numel(results_list);
    data = cell(n, 8);

    for i = 1:n
        R  = results_list{i};
        pr = struct; if isfield(R,'priors') && isstruct(R.priors), pr = R.priors; end

        % ===== alpha =====
        a_mean = R.alpha.mean;
        mu_a   = getd(pr,'mu_alpha',0.5);
        sd_a   = getd(pr,'sigma_alpha',1);
        a_sddr = NaN;
        if ~isempty(opts.null_alpha)
            a_sddr = local_sddr(R.alpha.draws, opts.null_alpha, mu_a, sd_a);
        end
        alpha_str = format_mean_sddr(a_mean, a_sddr);  % "m (SDDR)" / "m (—)"
        a_sd = R.alpha.std;

        % ===== kappa =====
        k_mean = R.kappa.mean;
        mu_k   = getd(pr,'mu_kappa',0);
        sd_k   = getd(pr,'sigma_kappa',1);
        k_sddr = local_sddr(R.kappa.draws, opts.null_kappa, mu_k, sd_k);
        kappa_str = format_mean_sddr(k_mean, k_sddr);
        k_sd = R.kappa.std;

        % ===== theta　=====
        hasTheta = isfield(R,'theta') && isstruct(R.theta) && ...
                   isfield(R.theta,'draws') && ~isempty(R.theta.draws) && ...
                   isfield(R.theta,'mean')  && ~isempty(R.theta.mean);
        if hasTheta
            t_mean = R.theta.mean;
            mu_t   = getd(pr,'mu_theta',0);
            sd_t   = getd(pr,'sigma_theta',1);
            t_sddr = local_sddr(R.theta.draws, opts.null_theta, mu_t, sd_t);
            theta_str = format_mean_sddr(t_mean, t_sddr);
            t_sd = R.theta.std;
        else
            theta_str = '—';
            t_sd = NaN;
        end

        % ===== row =====
        data{i,1} = label;
        data{i,2} = names{i};
        data{i,3} = alpha_str;   
        data{i,4} = kappa_str;   
        data{i,5} = theta_str;   
        data{i,6} = a_sd;
        data{i,7} = k_sd;
        data{i,8} = t_sd;
    end

    T = cell2table(data, 'VariableNames', ...
        {'ModelType','GapMeasure','Alpha(p-value)','Kappa(p-value/SDDR)','Theta(p-value/SDDR)', ...
         'Alpha_sd','Kappa_sd','Theta_sd'});
end

% ===== helpers =====
function s = format_mean_sddr(meanval, sddr)
    if isnan(sddr)
        s = sprintf('%.3f (—)', meanval);
    else
        s = sprintf('%.3f (%.3f)', meanval, sddr);
    end
end

function r = local_sddr(draws, null_val, mu, sigma)
    % SDDR = posterior(null) / prior(null)
    % posterior: KDE, prior: Normal(mu, sigma)
    epsv = 1e-12;
    try
        post_at = ksdensity(draws, null_val, 'Function','pdf');
    catch
        xs = linspace(min(draws), max(draws), 512);
        pdfs = ksdensity(draws, xs, 'Function','pdf');
        post_at = interp1(xs, pdfs, null_val, 'linear', 0);
    end
    prior_at = normpdf(null_val, mu, sigma);
    r = max(post_at, epsv) / max(prior_at, epsv);
end

function val = getd(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), val = s.(f); else, val = d; end
end
%% -------- Build tables --------
T0 = summarize_results(results_list_ces, model_names_ces, 'CES');
T1 = summarize_results(results_list_ces_xerror, model_names_ces_xerror, 'CES + xerror');
T2 = summarize_results(results_list, model_names, 'HSA');
T3 = summarize_results(results_list_xerror, xerror_names, 'HSA + xerror');
T4 = summarize_results(results_list_decomp, decomp_names, 'HSA (decomp)');
T5 = summarize_results(results_list_decomp_xerror, xerror_names_decomp, 'HSA (decomp + xerror)');
% ---- OLS summary ----
T_ols_ces = summarize_ols_for_merge(ols_ces, 'CES（OLS）');
T_ols_hsa = summarize_ols_for_merge(ols_hsa, 'HSA (OLS)');
% ---- Combine all ----
AllTables = [T_ols_ces; T_ols_hsa; T0; T1; T2; T3; T4; T5;];
AllTables = sortrows(AllTables, {'GapMeasure'});
%% -------- Display --------
disp('================================================================');
disp('Posterior Summary (mean [2.5%,97.5%]) for α, κ, θ');
disp('================================================================');
disp(AllTables);

% %% -------- Export to Excel --------
outpath = 'results/hsa_summary_tables.xlsx';
writetable(AllTables, outpath, 'Sheet','Results');