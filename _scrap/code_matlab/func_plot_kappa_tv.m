function func_plot_kappa_tv(results, opts)
% Plot time-varying kappa_t with posterior mean and 95% credible intervals
%
% results: struct containing either
%   - results.states.kappa_draws (n_draws x T)  <-- Modified to look here first
%   - or results.kappa_draws
%   - or results.states.kappa_mean
%
% opts: optional struct
%   - show_CI (default: true)
%   - smooth_window (default: 0)
%   - export_path (default: "")
%   - dates (default: [], optional T x 1 datetime or datenum)

    if nargin < 2, opts = struct; end
    show_CI       = getd(opts, 'show_CI', true);
    smooth_window = getd(opts, 'smooth_window', 0);
    export_path   = getd(opts, 'export_path', "");
    dates_vec     = getd(opts, 'dates', []);

    figure('Color','w','Position',[200 200 900 400]); hold on; grid on;

    %% ----- Extract data -----
    % 1. Try to find draws in .states (matching previous code)
    if isfield(results, 'states') && isfield(results.states, 'kappa_draws')
        kappa_draws = results.states.kappa_draws;
    % 2. Try to find draws in top level
    elseif isfield(results, 'kappa_draws')
        kappa_draws = results.kappa_draws;
    else
        kappa_draws = [];
    end

    if ~isempty(kappa_draws)
        [~, T] = size(kappa_draws); % assuming n_draws x T
        % posterior summary
        mean_kappa = mean(kappa_draws, 1, 'omitnan');
        q_lo = quantile(kappa_draws, 0.025, 1);
        q_hi = quantile(kappa_draws, 0.975, 1);
    elseif isfield(results, 'states') && isfield(results.states, 'kappa_mean')
        % Fallback: only posterior mean available
        mean_kappa = results.states.kappa_mean(:)';
        T = numel(mean_kappa);
        q_lo = mean_kappa; q_hi = mean_kappa;
        if show_CI
            warning('Only kappa mean found; CI cannot be computed.');
            show_CI = false;
        end
    else
        error('results must contain states.kappa_draws, kappa_draws or states.kappa_mean');
    end

    %% ----- Optional smoothing -----
    if smooth_window > 1
        mean_kappa = movmean(mean_kappa, smooth_window);
        q_lo = movmean(q_lo, smooth_window);
        q_hi = movmean(q_hi, smooth_window);
    end

    %% ----- Plot -----
    % Define X-axis (Dates or Index)
    if ~isempty(dates_vec) && numel(dates_vec) == T
        t_axis = dates_vec;
        is_date = true;
    else
        t_axis = 1:T;
        is_date = false;
    end
    
    handles = [];
    labels = {};

    % Plot CI
    if show_CI
        % Make polygon for fill
        x_poly = [t_axis(:); flipud(t_axis(:))];
        y_poly = [q_lo(:);   flipud(q_hi(:))];
        
        h_ci = fill(x_poly, y_poly, [0.8 0.8 1], ...
             'EdgeColor','none','FaceAlpha',0.4);
        handles(end+1) = h_ci;
        labels{end+1}  = '95% CI';
    end

    % Plot Mean
    h_mean = plot(t_axis, mean_kappa, 'b-', 'LineWidth', 2);
    handles(end+1) = h_mean;
    labels{end+1}  = 'Posterior mean';

    % Zero line
    yline(0, 'k:', 'LineWidth', 1, 'HandleVisibility','off'); % Exclude from legend

    % Formatting
    if is_date
        xlabel('Date'); 
        datetick('x','yyyy','keeplimits'); % Adjust format as needed
    else
        xlabel('Time'); 
    end
    ylabel('\kappa_t');
    title('\kappa_t: posterior mean and 95% credible interval');
    
    % Legend with specific handles
    legend(handles, labels, 'Location', 'best');
    
    %% ----- Export -----
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