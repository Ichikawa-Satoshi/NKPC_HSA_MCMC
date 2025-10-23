function out = plot_ces_results_compare(results_list, model_names, opts)
% PLOT_CES_RESULTS_COMPARE
%   α ~ Normal(0.5, 0.1), κ ~ Normal(0, 0.01)
% Always saves the figure under "figure/".

    if nargin < 2 || isempty(model_names)
        model_names = arrayfun(@(i) sprintf('Model %d', i), 1:numel(results_list), 'UniformOutput', false);
    end
    if nargin < 3, opts = struct; end
    if ~isfield(opts,'export_path'), opts.export_path = "ces_sampling_compare.png"; end  % default filename

    % === Priors ===
    mu_a  = 0.5;  sig_a = 0.1;
    mu_k  = 0.0;  sig_k = 0.01;
    test_value = 0;

    n = numel(results_list);
    if mod(n,2) ~= 0
        error('results_list and model_names must contain an even number of entries (before/after pairs).');
    end
    K = n/2;

    fig_h = figure('Position',[60 60 1300 max(350, 260*K)], 'Color','w');
    tl = tiledlayout(K, 2, 'TileSpacing','compact','Padding','compact');

    out = repmat(struct('PairLabel',"", 'ModelBefore',"", 'ModelAfter',"", ...
        'post_mean_alpha_before',NaN, 'post_mean_alpha_after',NaN, 'delta_alpha_mean',NaN, ...
        'post_mean_kappa_before',NaN, 'post_mean_kappa_after',NaN, 'delta_kappa_mean',NaN, ...
        'BF01_kappa_before',NaN, 'BF01_kappa_after',NaN), K,1);

    % Precompute priors for plotting
    xa_prior = linspace(0,1,800);
    prior_a  = normpdf(xa_prior, mu_a, sig_a);
    xk_prior = linspace(mu_k - 5*sig_k, mu_k + 5*sig_k, 800);
    prior_k  = normpdf(xk_prior, mu_k, sig_k);
    prior_at0 = normpdf(test_value, mu_k, sig_k);

    for p = 1:K
        i_before = 2*p - 1;  i_after  = 2*p;
        Rb = results_list{i_before};  Ra = results_list{i_after};
        name_before = model_names{i_before};  name_after  = model_names{i_after};

        % === α ===
        nexttile((p-1)*2 + 1); hold on; grid on
        [post_ab, xa_b] = ksdensity(Rb.alpha.draws);
        [post_aa, xa_a] = ksdensity(Ra.alpha.draws);
        plot(xa_prior, prior_a, 'r--', 'LineWidth',1.4, 'DisplayName','Prior (Normal)');
        plot(xa_b, post_ab, 'b-', 'LineWidth',1.8, 'DisplayName','Posterior (before)');
        plot(xa_a, post_aa, 'g-', 'LineWidth',1.8, 'DisplayName','Posterior (after)');
        mean_ab = mean(Rb.alpha.draws,'omitnan');  mean_aa = mean(Ra.alpha.draws,'omitnan');
        xlim([min([0, xa_b, xa_a]), max([1, xa_b, xa_a])]);
        title({ strip_pair_label(name_before, name_after), '\alpha: Prior & Posterior', ...
                sprintf('Prior mean = %.3f | Post mean: before=%.3f, after=%.3f | \x394=%.3f', ...
                        mu_a, mean_ab, mean_aa, mean_aa-mean_ab)}, 'Interpreter','tex');
        xlabel('\alpha','Interpreter','tex'); ylabel('Density'); legend('Location','best');

        % === κ ===
        nexttile((p-1)*2 + 2); hold on; grid on
        [post_kb, xk_b] = ksdensity(Rb.kappa.draws);
        [post_ka, xk_a] = ksdensity(Ra.kappa.draws);
        plot(xk_prior, prior_k, 'r--', 'LineWidth',1.4, 'DisplayName','Prior (Normal)');
        plot(xk_b, post_kb, 'b-', 'LineWidth',1.8, 'DisplayName','Posterior (before)');
        plot(xk_a, post_ka, 'g-', 'LineWidth',1.8, 'DisplayName','Posterior (after)');
        xline(test_value,'k:','LineWidth',1.1,'DisplayName','H_0');
        mean_kb = mean(Rb.kappa.draws,'omitnan');  mean_ka = mean(Ra.kappa.draws,'omitnan');
        post_at0_b = interp1(xk_b, post_kb, test_value, 'linear', 0);
        post_at0_a = interp1(xk_a, post_ka, test_value, 'linear', 0);
        BF01_b = max(1e-12, post_at0_b) / max(1e-12, prior_at0);
        BF01_a = max(1e-12, post_at0_a) / max(1e-12, prior_at0);
        xlim([min([xk_prior, xk_b, xk_a]), max([xk_prior, xk_b, xk_a])]);
        title({ '\kappa: Prior & Posterior', ...
                sprintf('Prior mean = %.4f | Post mean: before=%.4f, after=%.4f | \x394=%.4f', ...
                        mu_k, mean_kb, mean_ka, mean_ka-mean_kb), ...
                sprintf('BF_{01}(before)=%.3f | BF_{01}(after)=%.3f', BF01_b, BF01_a)}, 'Interpreter','tex');
        xlabel('\kappa','Interpreter','tex'); ylabel('Density'); legend('Location','best');

        % collect
        out(p).PairLabel = string(strip_pair_label(name_before, name_after));
        out(p).ModelBefore = string(name_before);  out(p).ModelAfter  = string(name_after);
        out(p).post_mean_alpha_before = mean_ab;   out(p).post_mean_alpha_after  = mean_aa;
        out(p).delta_alpha_mean       = mean_aa - mean_ab;
        out(p).post_mean_kappa_before = mean_kb;   out(p).post_mean_kappa_after  = mean_ka;
        out(p).delta_kappa_mean       = mean_ka - mean_kb;
        out(p).BF01_kappa_before = BF01_b;         out(p).BF01_kappa_after  = BF01_a;
    end

    % Overall title
    title(tl, 'CES: Prior vs Posterior (Before vs After)', 'FontWeight','bold');

    % ----- simple save (no helpers) -----
    if ~exist("fig",'dir'), mkdir("fig"); end
    % ensure extension (if none, append .png)
    [~, ~, ext] = fileparts(char(opts.export_path));
    if isempty(ext), opts.export_path = opts.export_path + ".png"; end
    exportgraphics(fig_h, fullfile("fig", opts.export_path), 'Resolution', 200, 'BackgroundColor','white');
    fprintf('Saved: %s\n', fullfile("fig", opts.export_path));
end

% ---- helpers ------------------------------------------------------------
function lbl = strip_pair_label(name_before, name_after)
    % Remove " — before" / " — after" and return the common prefix
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