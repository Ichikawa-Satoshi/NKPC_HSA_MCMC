function plot_decomposition(results_list, model_names, N_data, dates)
    n_models = length(results_list);
    n_cols = 2;                                % number of columns
    n_rows = ceil(n_models / n_cols);          % number of rows
    figure('Position',[100 100 1400 300*n_rows]);

    for m = 1:n_models
        results = results_list{m};
        Nbar = results.states.Nbar_mean;
        Nhat = results.states.Nhat_mean;

        subplot(n_rows, n_cols, m);

        % Left axis: Observed & Trend
        yyaxis left
        plot(dates, N_data, 'k-', 'LineWidth',1.5, 'DisplayName','Observed N_t'); hold on;
        plot(dates, Nbar, 'b-', 'LineWidth',2, 'DisplayName','Trend N̄_t');
        ylabel('Level');
        
        % Right axis: Cycle
        yyaxis right
        plot(dates, Nhat, 'r--', 'LineWidth',2, 'DisplayName','Cycle N̂_t');
        ylabel('Cycle');

        % Common settings
        xlabel('Year');
        title(sprintf('Decomposition of N_t: %s', model_names{m}));
        grid on;

        % Add legend (combined)
        yyaxis left
        legend('Location','southwest');
    end
end