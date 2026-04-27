function out =func_plot_trace(results_list, model_names)
    if nargin < 2 || isempty(model_names)
        model_names = arrayfun(@(i) sprintf('Model %d', i), 1:numel(results_list), 'UniformOutput', false);
    end
    n_models = numel(results_list);
    figure('Position',[100 100 1400 280*n_models], 'Color','w');
    tl = tiledlayout(n_models, 3, 'TileSpacing','compact', 'Padding','compact');
    for m = 1:n_models
        R = results_list{m};
        mname = model_names{m};
        % --- α ---
        ax = nexttile((m-1)*3 + 1);
        if isfield(R,'alpha') && isfield(R.alpha,'draws')
            plot(R.alpha.draws, 'LineWidth', 0.8); grid on
        else
            cla; text(0.5,0.5,'(no alpha.draws)','HorizontalAlignment','center'); axis off
        end
        title({'\alpha posterior draws'}, 'Interpreter','tex');
        ylabel('\alpha','Interpreter','tex');
        xlabel('Iteration');

        % --- κ ---
        ax = nexttile((m-1)*3 + 2);
        if isfield(R,'kappa') && isfield(R.kappa,'draws')
            plot(R.kappa.draws, 'LineWidth', 0.8); grid on
        else
            cla; text(0.5,0.5,'(no kappa.draws)','HorizontalAlignment','center'); axis off
        end
        title({mname,'\kappa posterior draws'}, 'Interpreter','tex'); % 
        ylabel('\kappa','Interpreter','tex');
        xlabel('Iteration');

        % --- theta ---
        ax = nexttile((m-1)*3 + 3);
        if isfield(R,'theta') && isfield(R.theta,'draws')
            plot(R.theta.draws, 'LineWidth', 0.8); grid on
        else
            cla; text(0.5,0.5,'(no theta.draws)','HorizontalAlignment','center'); axis off
        end
        title({mname,'\theta posterior draws'}, 'Interpreter','tex'); % 
        ylabel('\theta','Interpreter','tex');
        xlabel('Iteration');
    end

end