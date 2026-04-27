function out = func_plot_posterior_tv(results_list, model_names, opts)
    if nargin < 2 || isempty(model_names)
        model_names = arrayfun(@(i) sprintf('Model %d', i), 1:numel(results_list), 'UniformOutput', false);
    end
    if nargin < 3, opts = struct; end
    if ~isfield(opts,'export_path'), opts.export_path = ""; end
    if ~isfield(opts,'show_alpha_bounds'), opts.show_alpha_bounds = true; end
    if ~isfield(opts,'ks_points'), opts.ks_points = 800; end

    n_models = numel(results_list);

    % alpha, theta, beta
    f = figure('Position',[80 80 2100 max(350, 260*n_models)], 'Color','w');
    tiledlayout(n_models, 3, 'TileSpacing','compact', 'Padding','compact');

    out = repmat(struct('Model',"", ...
                        'BF01_theta',NaN, ...
                        'post_mean_alpha',NaN, ...
                        'post_mean_theta',NaN, ...
                        'post_mean_delta',NaN), n_models,1);

    for m = 1:n_models
        R = results_list{m};
        mname = model_names{m};

        %==== priors (fallback) ====
        pr = struct;
        if isfield(R,'priors') && isstruct(R.priors), pr = R.priors; end
        mu_a      = getd(pr,'mu_alpha',    0.5);
        sig_a     = getd(pr,'sigma_alpha', 1);
        mu_theta  = getd(pr,'mu_theta',    0.0);
        sig_theta = getd(pr,'sigma_theta', 1);
        mu_delta   = getd(pr,'mu_delta',     0.0);
        sig_delta  = getd(pr,'sigma_delta',  1);

        %% ===== 1) α =====
        nexttile((m-1)*4 + 1); hold on; grid on
        a_draws = R.alpha.draws(:);
        [post_a, xa_post] = ksdensity(a_draws);
        plot(xa_post, post_a, 'b-', 'LineWidth',1.6,'DisplayName','Posterior');

        xa_min = min([0, min(xa_post)]); 
        xa_max = max([1, max(xa_post)]);
        xa = linspace(xa_min, xa_max, opts.ks_points);
        prior_a = normpdf(xa, mu_a, sig_a);
        plot(xa, prior_a, 'r--','LineWidth',1.6,'DisplayName','Prior');

        if opts.show_alpha_bounds
            xline(0,'k:','HandleVisibility','off'); 
            xline(1,'k:','HandleVisibility','off');
        end

        post_mean_a = mean(a_draws,'omitnan');
        title({ sprintf('%s', mname), '\alpha: Prior vs Posterior', ...
            sprintf('Prior mean = %.3f, Post mean = %.3f', mu_a, post_mean_a)}, ...
            'Interpreter','tex');
        xlabel('\alpha'); ylabel('Density'); legend('Location','best');

        %% ===== 2) θ =====
        nexttile((m-1)*4 + 2); hold on; grid on
        hasTheta = isfield(R,'theta') && isstruct(R.theta) && ...
                   isfield(R.theta,'draws') && ~isempty(R.theta.draws);

        if hasTheta
            t_draws = R.theta.draws(:);
            [post_t, xt_post] = ksdensity(t_draws);
            plot(xt_post, post_t, 'b-','LineWidth',1.6,'DisplayName','Posterior');

            xt = linspace(min(min(xt_post), mu_theta-5*sig_theta), ...
                          max(max(xt_post), mu_theta+5*sig_theta), opts.ks_points);
            prior_t = normpdf(xt, mu_theta, sig_theta);
            plot(xt, prior_t, 'r--','LineWidth',1.6,'DisplayName','Prior');

            prior_at0 = normpdf(0, mu_theta, sig_theta);
            post_at0  = interp1(xt_post, post_t, 0, 'linear', 0);
            BF01_theta = max(1e-12, post_at0) / max(1e-12, prior_at0);

            post_mean_t = mean(t_draws,'omitnan');
            xline(0,'k:','LineWidth',1.1,'DisplayName','H_0');
            title({ '\theta: Prior vs Posterior', ...
                sprintf('Prior mean = %.6f, Post mean = %.6f', mu_theta, post_mean_t), ...
                sprintf('BF_{01}(\\theta=0) = %.3f', BF01_theta)}, 'Interpreter','tex');
            xlabel('\theta'); ylabel('Density'); legend('Location','best');
        else
            axis off
            text(0.5,0.6,'\theta not estimated','HorizontalAlignment','center','FontSize',12,'Interpreter','tex');
            text(0.5,0.4,'(skipped)','HorizontalAlignment','center','FontSize',11);
            BF01_theta = NaN; post_mean_t = NaN;
        end

        %% ===== 3) delta =====
        nexttile((m-1)*4 + 3); hold on; grid on
        hasdelta = isfield(R,'delta') && isstruct(R.delta) && isfield(R.delta,'draws') ...
                  && ~isempty(R.delta.draws);

        if hasdelta
            d_draws = R.delta.draws(:);
            [post_d, xd_post] = ksdensity(d_draws);
            plot(xd_post, post_d, 'b-','LineWidth',1.6,'DisplayName','Posterior');

            xd = linspace(min(min(xd_post), mu_delta-5*sig_delta), ...
                          max(max(xd_post), mu_delta+5*sig_delta), opts.ks_points);
            prior_d = normpdf(xd, mu_delta, sig_delta);
            plot(xd, prior_d, 'r--','LineWidth',1.6,'DisplayName','Prior');

            post_mean_delta = mean(d_draws,'omitnan');
            xline(0,'k:','LineWidth',1.1,'DisplayName','0');
            title({ '\delta: Prior vs Posterior', ...
                sprintf('Prior mean = %.6f, Post mean = %.6f', mu_delta, post_mean_delta)}, ...
                'Interpreter','tex');
            xlabel('\delta'); ylabel('Density'); legend('Location','best');
        else
            axis off
            text(0.5,0.6,'\beta not estimated','HorizontalAlignment','center','FontSize',12,'Interpreter','tex');
            text(0.5,0.4,'(skipped)','HorizontalAlignment','center','FontSize',11);
            post_mean_delta = NaN;
        end

        %==== collect ====
        out(m).Model            = string(mname);
        out(m).BF01_theta       = BF01_theta;
        out(m).post_mean_alpha  = post_mean_a;
        out(m).post_mean_theta  = post_mean_t;
        out(m).post_mean_delta   = post_mean_delta;
    end

    % save
    if ~strcmp(opts.export_path, "")
        outdir = fileparts(opts.export_path);
        if ~isempty(outdir) && ~exist(outdir,'dir'), mkdir(outdir); end
        exportgraphics(f, opts.export_path, 'Resolution', 200);
    end
end

%----- helper -----
function val = getd(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f)), val = s.(f); else, val = d; end
end