function out = plot_ces_results(results_list, model_names, opts)
% PLOT_CES_RESULTS
%   alpha ~ Normal(0.5, 0.1)
%   kappa ~ Normal(0, 0.01)
%
% Usage:
%   out = plot_ces_results(results_list, model_names);
%   out = plot_ces_results(results_list, model_names, struct('export_path','figs/ces_grid.png'));

    if nargin < 2 || isempty(model_names)
        model_names = arrayfun(@(i) sprintf('Model %d', i), 1:numel(results_list), 'UniformOutput', false);
    end
    if nargin < 3, opts = struct; end
    if ~isfield(opts,'export_path'), opts.export_path = ""; end

    % alpha prior: Normal(0.5, 0.1)
    mu_a  = 0.5;
    sig_a = 0.1;

    % kappa prior: Normal(0, 0.01)
    mu_kappa = 0.0;
    sig_kappa = 0.01;

    n_models = numel(results_list);
    figure('Position',[80 80 1400 max(350, 240*n_models)], 'Color','w');
    tiledlayout(n_models, 2, 'TileSpacing','compact', 'Padding','compact');

    out = repmat(struct('Model',"", ...
                        'BF01_kappa',NaN, ...
                        'post_mean_alpha',NaN, ...
                        'post_mean_kappa',NaN), n_models,1);

    for m = 1:n_models
        R = results_list{m};
        mname = model_names{m};

        %% ===== α (Prior: Normal(0.5, 0.1) ) =====
        nexttile((m-1)*2 + 1); hold on; grid on

        % posterior (KDE)
        [post_a, xa_post] = ksdensity(R.alpha.draws);
        plot(xa_post, post_a, 'b-', 'LineWidth', 1.6, 'DisplayName','Posterior');

        % prior (Normal)
        xa = linspace(min(0, min(xa_post)), max(1, max(xa_post)), 800);
        prior_a = normpdf(xa, mu_a, sig_a);
        plot(xa, prior_a, 'r--', 'LineWidth',1.6, 'DisplayName','Prior (Normal)');

        % aB=12; bB=12; xa = linspace(0,1,800); prior_a = betapdf(xa,aB,bB);
        % plot(xa, prior_a, 'r--', 'LineWidth',1.6, 'DisplayName','Prior (Beta(12,12))');
        % mu_a = aB/(aB+bB);

        post_mean_a = mean(R.alpha.draws,'omitnan');
        title({ sprintf('%s', mname), '\alpha: Prior vs Posterior', ...
                sprintf('Prior mean = %.3f,  Post mean = %.3f', mu_a, post_mean_a)}, ...
                'Interpreter','tex');
        xlabel('\alpha','Interpreter','tex'); ylabel('Density');
        legend('Location','best');

        %% ===== κ (Prior: Normal(0, 0.01)) =====
        nexttile((m-1)*2 + 2); hold on; grid on

        % posterior (KDE)
        [post_k, xk_post] = ksdensity(R.kappa.draws);
        plot(xk_post, post_k, 'b-', 'LineWidth', 1.6, 'DisplayName','Posterior');

        % prior (Normal)
        xk = linspace(min(min(xk_post), mu_kappa - 5*sig_kappa), ...
                      max(max(xk_post), mu_kappa + 5*sig_kappa), 800);
        prior_k = normpdf(xk, mu_kappa, sig_kappa);
        plot(xk, prior_k, 'r--', 'LineWidth', 1.6, 'DisplayName','Prior (Normal)');

        % Savage–Dickey BF_01 at 0（post/prior）
        test_value = 0;
        prior_at0 = normpdf(test_value, mu_kappa, sig_kappa);
        post_at0  = interp1(xk_post, post_k, test_value, 'linear', 0);
        BF01_kappa = max(1e-12, post_at0) / max(1e-12, prior_at0);

        post_mean_k = mean(R.kappa.draws,'omitnan');
        xline(test_value,'k:','LineWidth',1.1,'DisplayName','H_0');

        title({ '\kappa: Prior vs Posterior', ...
                sprintf('Prior mean = %.7f,  Post mean = %.7f', mu_kappa, post_mean_k), ...
                sprintf('BF_{01}(\\kappa=0) = %.3f', BF01_kappa)}, 'Interpreter','tex');
        xlabel('\kappa','Interpreter','tex'); ylabel('Density');
        legend('Location','best');

        % collect
        out(m).Model = string(mname);
        out(m).BF01_kappa = BF01_kappa;
        out(m).post_mean_alpha = post_mean_a;
        out(m).post_mean_kappa = post_mean_k;
    end
end