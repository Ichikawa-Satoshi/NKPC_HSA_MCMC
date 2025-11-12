function func_plot_kappa_tv(results, opts)
% Plot time-varying kappa_t with posterior mean and 95% credible intervals
%
% results: struct containing either
%   - results.kappa_draws (n_draws x T)
%   - or results.states.kappa_mean (T x 1)
%
% opts: optional struct
%   - show_CI (default: true)
%   - smooth_window (default: 0)
%   - export_path (default: "")

    if nargin < 2, opts = struct; end
    show_CI = getd(opts, 'show_CI', true);
    smooth_window = getd(opts, 'smooth_window', 0);
    export_path = getd(opts, 'export_path', "");

    figure('Color','w','Position',[200 200 900 400]); hold on; grid on;

    % ----- Extract data -----
    if isfield(results, 'kappa_draws')
        kappa_draws = results.kappa_draws;
        [n_draws, T] = size(kappa_draws);

        % posterior summary
        mean_kappa = mean(kappa_draws,1,'omitnan');
        q_lo = quantile(kappa_draws, 0.025, 1);
        q_hi = quantile(kappa_draws, 0.975, 1);

    elseif isfield(results, 'states') && isfield(results.states, 'kappa_mean')
        % Fallback: only posterior mean available
        mean_kappa = results.states.kappa_mean(:)';
        T = numel(mean_kappa);
        q_lo = mean_kappa; q_hi = mean_kappa;
        warning('Only kappa mean found in results.states; CI not available.');
        show_CI = false;
    else
        error('results must contain kappa_draws or states.kappa_mean');
    end

    % Optional smoothing
    if smooth_window > 1
        mean_kappa = movmean(mean_kappa, smooth_window);
        q_lo = movmean(q_lo, smooth_window);
        q_hi = movmean(q_hi, smooth_window);
    end

    % ----- Plot -----
    t = 1:T;
    if show_CI
        fill([t, fliplr(t)], [q_lo, fliplr(q_hi)], [0.8 0.8 1], ...
             'EdgeColor','none','FaceAlpha',0.4,'DisplayName','95% CI');
    end
    plot(t, mean_kappa, 'b-', 'LineWidth',2, 'DisplayName','Posterior mean');
    yline(0,'k:','LineWidth',1);
    xlabel('Time'); ylabel('\kappa_t');
    title('\kappa_t: posterior mean and 95% credible interval');
    legend('Location','best');

    % ----- Export -----
    if ~strcmp(export_path, "")
        outdir = fileparts(export_path);
        if ~isempty(outdir) && ~exist(outdir,'dir'), mkdir(outdir); end
        exportgraphics(gcf, export_path, 'Resolution', 200);
    end
end

% helper
function val = getd(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f))
        val = s.(f);
    else
        val = d;
    end
end