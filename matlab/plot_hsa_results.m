function plot_hsa_results(results_list, model_names, opts)
% PLOT_HSA_RESULTS
%   alpha ~ Normal(0.5, 0.1)
%   kappa ~ Normal(0, 0.01)
%   theta ~ Normal(0, 0.01)
%
% Usage:
%   out = plot_hsa_results(results_list, model_names);
%   out = plot_hsa_results(results_list, model_names, struct('export_path','figs/hsa_grid.png'));

    if nargin < 2 || isempty(model_names)
        model_names = arrayfun(@(i) sprintf('Model %d', i), 1:numel(results_list), 'UniformOutput', false);
    end
    if nargin < 3, opts = struct; end
    if ~isfield(opts,'export_path'), opts.export_path = ""; end

    % priors
    mu_a  = 0.5; sig_a = 0.1;
    mu_kappa = 0.0; sig_kappa = 0.01;
    mu_theta = 0.0; sig_theta = 0.01;

    n_models = numel(results_list);
    f=figure('Position',[80 80 1800 max(350, 260*n_models)], 'Color','w');
    tiledlayout(n_models, 3, 'TileSpacing','compact', 'Padding','compact');

    out = repmat(struct('Model',"", ...
                        'BF01_kappa',NaN, ...
                        'BF01_theta',NaN, ...
                        'post_mean_alpha',NaN, ...
                        'post_mean_kappa',NaN, ...
                        'post_mean_theta',NaN), n_models,1);

    for m = 1:n_models
        R = results_list{m};
        mname = model_names{m};

        %% ===== α =====
        nexttile((m-1)*3 + 1); hold on; grid on
        [post_a, xa_post] = ksdensity(R.alpha.draws);
        plot(xa_post, post_a, 'b-', 'LineWidth',1.6,'DisplayName','Posterior');
        xa = linspace(min(0, min(xa_post)), max(1, max(xa_post)), 800);
        prior_a = normpdf(xa, mu_a, sig_a);
        plot(xa, prior_a, 'r--','LineWidth',1.6,'DisplayName','Prior');
        post_mean_a = mean(R.alpha.draws,'omitnan');
        title({ sprintf('%s', mname), '\alpha: Prior vs Posterior', ...
            sprintf('Prior mean = %.3f, Post mean = %.3f', mu_a, post_mean_a)}, ...
            'Interpreter','tex');
        xlabel('\alpha'); ylabel('Density'); legend('Location','best');

        %% ===== κ =====
        nexttile((m-1)*3 + 2); hold on; grid on
        [post_k, xk_post] = ksdensity(R.kappa.draws);
        plot(xk_post, post_k, 'b-','LineWidth',1.6,'DisplayName','Posterior');
        xk = linspace(min(min(xk_post), mu_kappa-5*sig_kappa), ...
                      max(max(xk_post), mu_kappa+5*sig_kappa), 800);
        prior_k = normpdf(xk, mu_kappa, sig_kappa);
        plot(xk, prior_k, 'r--','LineWidth',1.6,'DisplayName','Prior');
        prior_at0 = normpdf(0, mu_kappa, sig_kappa);
        post_at0  = interp1(xk_post, post_k, 0, 'linear', 0);
        BF01_kappa = max(1e-12, post_at0) / max(1e-12, prior_at0);
        post_mean_k = mean(R.kappa.draws,'omitnan');
        xline(0,'k:','LineWidth',1.1,'DisplayName','H_0');
        title({ '\kappa: Prior vs Posterior', ...
            sprintf('Prior mean = %.7f, Post mean = %.7f', mu_kappa, post_mean_k), ...
            sprintf('BF_{01}(\\kappa=0) = %.3f', BF01_kappa)}, 'Interpreter','tex');
        xlabel('\kappa'); ylabel('Density'); legend('Location','best');

        %% ===== θ =====
        nexttile((m-1)*3 + 3); hold on; grid on
        [post_t, xt_post] = ksdensity(R.theta.draws);
        plot(xt_post, post_t, 'b-','LineWidth',1.6,'DisplayName','Posterior');
        xt = linspace(min(min(xt_post), mu_theta-5*sig_theta), ...
                      max(max(xt_post), mu_theta+5*sig_theta), 800);
        prior_t = normpdf(xt, mu_theta, sig_theta);
        plot(xt, prior_t, 'r--','LineWidth',1.6,'DisplayName','Prior');
        prior_at0 = normpdf(0, mu_theta, sig_theta);
        post_at0  = interp1(xt_post, post_t, 0, 'linear', 0);
        BF01_theta = max(1e-12, post_at0) / max(1e-12, prior_at0);
        post_mean_t = mean(R.theta.draws,'omitnan');
        xline(0,'k:','LineWidth',1.1,'DisplayName','H_0');
        title({ '\theta: Prior vs Posterior', ...
            sprintf('Prior mean = %.4f, Post mean = %.4f', mu_theta, post_mean_t), ...
            sprintf('BF_{01}(\\theta=0) = %.3f', BF01_theta)}, 'Interpreter','tex');
        xlabel('\theta'); ylabel('Density'); legend('Location','best');

        % collect
        out(m).Model = string(mname);
        out(m).BF01_kappa = BF01_kappa;
        out(m).BF01_theta = BF01_theta;
        out(m).post_mean_alpha = post_mean_a;
        out(m).post_mean_kappa = post_mean_k;
        out(m).post_mean_theta = post_mean_t;
    end
end