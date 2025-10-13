function out = plot_hsa_results_compare(results_list, model_names, opts)
% PLOT_HSA_RESULTS_COMPARE_LIST
% Compare multiple pre/post HSA-NKPC results in one go (list-based input).
% Each pair (pre, post) is plotted in one row across 5 parameters:
%   alpha, kappa, theta, rho1, rho2.
% Two figures are produced:
%   (1) Trace plots (pre vs post)  -- layout: K rows x 5 cols
%   (2) Prior vs Posterior overlays (pre vs post) -- layout: K rows x 5 cols
%
% INPUTS
%   results_list : 1x(2*K) cell array: {pre1, post1, pre2, post2, ...}
%                  Each element is a struct with fields:
%                    .alpha.draws, .kappa.draws, .theta.draws, .rho1.draws, .rho2.draws
%   model_names  : 1x(2*K) cellstr or string array:
%                    {'Name — before','Name — after', ...}
%                  (Optional) If length(model_names) == K, the function will
%                  auto-expand to "— before/— after".
%   opts         : struct (optional)
%       .export_path_traces     : "" or file path to save the trace figure
%       .export_path_densities  : "" or file path to save the density figure
%       .priors : struct to override Normal priors used for plotting:
%           .alpha_mu (0.5), .alpha_sig (0.05)
%           .kappa_mu (0.01), .kappa_sig (0.05)
%           .theta_mu (0.01), .theta_sig (0.05)
%           .rho1_mu  (0.5),  .rho1_sig  (0.05)
%           .rho2_mu  (-0.5), .rho2_sig  (0.05)
%
% OUTPUT
%   out : Kx1 struct with summary stats per pair:
%       .PairLabel
%       .ModelBefore, .ModelAfter
%       .means.(alpha|kappa|theta|rho1|rho2).pre/post
%       .BF01.kappa.pre/post   (Savage–Dickey at 0 with Normal prior)
%       .BF01.theta.pre/post   (Savage–Dickey at 0 with Normal prior)
%
% EXAMPLE
%   results_list = {res_out_pre, res_out_post, res_gap_pre, res_gap_post};
%   model_names  = {'Output — before','Output — after','Gap — before','Gap — after'};
%   plot_hsa_results_compare_list(results_list, model_names);

    % ----- defaults & checks -----
    if nargin < 2 || isempty(model_names)
        model_names = arrayfun(@(i) sprintf('Model %d', i), 1:numel(results_list), 'UniformOutput', false);
    end
    if nargin < 3, opts = struct; end
    if ~isfield(opts,'export_path_traces'),    opts.export_path_traces    = "hsa_results_compare.png"; end
    if ~isfield(opts,'export_path_densities'), opts.export_path_densities = "hsa_sample_compare.png"; end
    if ~isfield(opts,'priors'), opts.priors = struct; end

    pri = opts.priors;
    pri.alpha_mu = 0.5;  
    pri.alpha_sig = 0.1; 
    pri.kappa_mu = 0; 
    pri.kappa_sig = 0.01;
    pri.theta_mu = 0;
    pri.theta_sig = 0.01;
    pri.rho1_mu  = 0.5;  
    pri.rho1_sig = 0.05; 
    pri.rho2_mu  = -0.5; 
    pri.rho2_sig = 0.05; 

    n = numel(results_list);
    if mod(n,2) ~= 0
        error('results_list must contain an even number of elements (pre/post pairs).');
    end
    K = n/2;

    % Allow model_names to be length K (pair labels). If so, expand to before/after.
    if numel(model_names) == K
        expanded = cell(1, 2*K);
        for p = 1:K
            expanded{2*p-1} = sprintf('%s — before', model_names{p});
            expanded{2*p  } = sprintf('%s — after',  model_names{p});
        end
        model_names = expanded;
    elseif numel(model_names) ~= 2*K
        error('model_names must have K or 2*K entries.');
    end

    % ----- output allocation -----
    out = repmat(struct( ...
        'PairLabel',"", 'ModelBefore',"", 'ModelAfter',"", ...
        'means',struct, 'BF01',struct), K, 1);

    % ----- figure: trace plots -----
    fig_tr = figure('Name','HSA Traces (pre/post, multiple pairs)', ...
                    'Position', [80, 60, 1900, max(360, 220*K)], 'Color','w');
    tl_tr  = tiledlayout(K, 3, 'Padding','compact','TileSpacing','compact');

    % ----- figure: densities (prior vs posterior, pre vs post) -----
    fig_de = figure('Name','HSA Prior vs Posterior (pre/post, multiple pairs)', ...
                    'Position', [80, 460, 1900, max(360, 240*K)], 'Color','w');
    tl_de  = tiledlayout(K, 3, 'Padding','compact','TileSpacing','compact');

    % Helper for Savage–Dickey BF_01 at 0 with Normal prior (posterior/prior)
    sddr_bf01 = @(x_post, f_post, mu, sig) ...
        ( max(1e-12, interp1(x_post, f_post, 0, 'linear', 0)) / max(1e-12, normpdf(0, mu, sig)) );

    % ----- loop over pairs -----
    for p = 1:K
        ib = 2*p-1; ia = 2*p;
        Rb = results_list{ib};
        Ra = results_list{ia};
        name_b = model_names{ib};
        name_a = model_names{ia};
        pair_label = strip_pair_label(name_b, name_a);

        % ---- TRACE PLOTS (row p) ----
        % alpha
        nexttile(tl_tr, (p-1)*3 + 1); hold on; grid on
        plot(Rb.alpha.draws,'-'); plot(Ra.alpha.draws,'-');
        title('\alpha posterior draws'); xlabel('Iteration'); ylabel('\alpha'); legend({'pre','post'},'Location','best');

        % kappa
        nexttile(tl_tr, (p-1)*3 + 2); hold on; grid on
        plot(Rb.kappa.draws,'-'); plot(Ra.kappa.draws,'-');
        title('\kappa posterior draws'); xlabel('Iteration'); ylabel('\kappa'); legend({'pre','post'},'Location','best');

        % theta
        nexttile(tl_tr, (p-1)*3 + 3); hold on; grid on
        plot(Rb.theta.draws,'-'); plot(Ra.theta.draws,'-');
        title('\theta posterior draws'); xlabel('Iteration'); ylabel('\theta'); legend({'pre','post'},'Location','best');

        % ---- DENSITY PLOTS (row p) ----
        % alpha
        nexttile(tl_de, (p-1)*3 + 1); hold on; grid on
        xa = linspace( min([Rb.alpha.draws; Ra.alpha.draws]) - 0.5, ...
                       max([Rb.alpha.draws; Ra.alpha.draws]) + 0.5, 1000 );
        prior_a = normpdf(xa, pri.alpha_mu, pri.alpha_sig);
        [post_a_pre,  xa_pre ] = ksdensity(Rb.alpha.draws);
        [post_a_post, xa_post] = ksdensity(Ra.alpha.draws);
        plot(xa, prior_a, 'k--','LineWidth',2,'DisplayName','Prior');
        plot(xa_pre,  post_a_pre,  '-', 'LineWidth',2,'DisplayName','Posterior (pre)');
        plot(xa_post, post_a_post, '-', 'LineWidth',2,'DisplayName','Posterior (post)');
        xlabel('\alpha'); ylabel('Density');
        pm_a_pre  = mean(Rb.alpha.draws,'omitnan');
        pm_a_post = mean(Ra.alpha.draws,'omitnan');
        title({ sprintf('%s — \\alpha: Prior vs Posterior', pair_label), ...
                sprintf('Prior mean=%.3f | Post: pre=%.3f, post=%.3f', pri.alpha_mu, pm_a_pre, pm_a_post) }, ...
              'Interpreter','tex');
        legend('Location','best');

        % kappa
        nexttile(tl_de, (p-1)*3 + 2); hold on; grid on
        xk = linspace( min([Rb.kappa.draws; Ra.kappa.draws]) - 0.05, ...
                       max([Rb.kappa.draws; Ra.kappa.draws]) + 0.05, 1000 );
        prior_k = normpdf(xk, pri.kappa_mu, pri.kappa_sig);
        [post_k_pre,  xk_pre ] = ksdensity(Rb.kappa.draws);
        [post_k_post, xk_post] = ksdensity(Ra.kappa.draws);
        plot(xk, prior_k, 'k--','LineWidth',2,'DisplayName','Prior');
        plot(xk_pre,  post_k_pre,  '-', 'LineWidth',2,'DisplayName','Posterior (pre)');
        plot(xk_post, post_k_post, '-', 'LineWidth',2,'DisplayName','Posterior (post)');
        yl = ylim; plot([0 0], yl, 'k:', 'LineWidth',1.2, 'DisplayName','H_0');
        xlabel('\kappa'); ylabel('Density');
        pm_k_pre  = mean(Rb.kappa.draws,'omitnan');
        pm_k_post = mean(Ra.kappa.draws,'omitnan');
        bf01_k_pre  = sddr_bf01(xk_pre,  post_k_pre,  pri.kappa_mu, pri.kappa_sig);
        bf01_k_post = sddr_bf01(xk_post, post_k_post, pri.kappa_mu, pri.kappa_sig);
        title({ sprintf('%s — \\kappa: Prior vs Posterior', pair_label), ...
                sprintf('Prior mean=%.3f | Post: pre=%.3f, post=%.3f', pri.kappa_mu, pm_k_pre, pm_k_post), ...
                sprintf('BF_{01}(\\kappa=0): pre=%.2f, post=%.2f', bf01_k_pre, bf01_k_post) }, ...
              'Interpreter','tex');
        legend('Location','best');

        % theta
        nexttile(tl_de, (p-1)*3 + 3); hold on; grid on
        xt = linspace( min([Rb.theta.draws; Ra.theta.draws]) - 0.05, ...
                       max([Rb.theta.draws; Ra.theta.draws]) + 0.05, 1000);
        prior_t = normpdf(xt, pri.theta_mu, pri.theta_sig);
        [post_t_pre,  xt_pre ] = ksdensity(Rb.theta.draws);
        [post_t_post, xt_post] = ksdensity(Ra.theta.draws);
        plot(xt, prior_t, 'k--','LineWidth',2,'DisplayName','Prior');
        plot(xt_pre,  post_t_pre,  '-', 'LineWidth',2,'DisplayName','Posterior (pre)');
        plot(xt_post, post_t_post, '-', 'LineWidth',2,'DisplayName','Posterior (post)');
        yl = ylim; plot([0 0], yl, 'k:', 'LineWidth',1.2, 'DisplayName','H_0');
        xlabel('\theta'); ylabel('Density');
        pm_t_pre  = mean(Rb.theta.draws,'omitnan');
        pm_t_post = mean(Ra.theta.draws,'omitnan');
        bf01_t_pre  = sddr_bf01(xt_pre,  post_t_pre,  pri.theta_mu, pri.theta_sig);
        bf01_t_post = sddr_bf01(xt_post, post_t_post, pri.theta_mu, pri.theta_sig);
        title({ sprintf('%s — \\theta: Prior vs Posterior', pair_label), ...
                sprintf('Prior mean=%.3f | Post: pre=%.3f, post=%.3f', pri.theta_mu, pm_t_pre, pm_t_post), ...
                sprintf('BF_{01}(\\theta=0): pre=%.2f, post=%.2f', bf01_t_pre, bf01_t_post) }, ...
              'Interpreter','tex');
        legend('Location','best');

        % ---- collect output ----
        out(p).PairLabel   = string(pair_label);
        out(p).ModelBefore = string(name_b);
        out(p).ModelAfter  = string(name_a);
        out(p).means.alpha.pre  = pm_a_pre;   out(p).means.alpha.post  = pm_a_post;
        out(p).means.kappa.pre  = pm_k_pre;   out(p).means.kappa.post  = pm_k_post;
        out(p).means.theta.pre  = pm_t_pre;   out(p).means.theta.post  = pm_t_post;
        out(p).BF01.kappa.pre   = bf01_k_pre;
        out(p).BF01.kappa.post  = bf01_k_post;
        out(p).BF01.theta.pre   = bf01_t_pre;
        out(p).BF01.theta.post  = bf01_t_post;
    end

    % Super titles
    title(tl_tr, 'HSA NKPC — Trace plots (pre vs post)', 'FontWeight','bold');
    title(tl_de, 'HSA NKPC — Prior vs Posterior (pre vs post)', 'FontWeight','bold');

    exportgraphics(fig_tr, fullfile("fig", opts.export_path_traces), 'Resolution', 200, 'BackgroundColor','white');
    exportgraphics(fig_de, fullfile("fig", opts.export_path_densities), 'Resolution', 200, 'BackgroundColor','white');
end

% ---------------- helpers ----------------
function lbl = strip_pair_label(name_before, name_after)
    % Return the common prefix after removing " — before/after"
    nb = string(name_before); na = string(name_after);
    nb = replace(nb, "—", "-"); na = replace(na, "—", "-");
    nb = regexprep(nb, '\s*-\s*before$', '', 'ignorecase');
    na = regexprep(na, '\s*-\s*after$',  '', 'ignorecase');
    maxprefix = min(strlength(nb), strlength(na));
    k = 0;
    for i = 1:maxprefix
        if nb.extractBetween(i,i) == na.extractBetween(i,i), k = i; else, break; end
    end
    if k >= 1
        lbl = strtrim(extractBefore(nb, k+1));
        if lbl == "", lbl = strtrim(nb); end
    else
        lbl = strtrim(regexprep(string(name_before), '\s*-\s*before$', '', 'ignorecase'));
    end
end

% function tf = ischar_or_string(x)
%     tf = ischar(x) || isa(x,'string');
% end