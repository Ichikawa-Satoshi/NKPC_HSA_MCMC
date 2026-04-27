function summary_table = func_make_result_table(all_outputs, model_labels, block_labels)
% Build summary table combining posterior summaries and Bayes factors
%
% all_outputs  : cell array of structs returned from func_plot_posterior or plot_hsa_results
% model_labels : e.g. {'HSA','HSA-xerror','HSA-decomp','HSA-decomp-xerror'}
% block_labels : cell array of model names within each block, e.g. {'Output gap','Unemp gap','Markup^{-1}'}

    nBlocks = numel(all_outputs);      % number of model families (HSA variants)
    nModels = numel(all_outputs{1});   % number of gaps per block
    entries = [];

    for b = 1:nBlocks
        out = all_outputs{b};
        label = model_labels{b};
        for m = 1:nModels
            row = out(m);
            entries = [entries; 
                {label, block_labels{m}, ...
                 row.post_mean_alpha, ...
                 row.post_mean_kappa, ...
                 row.post_mean_theta, ...
                 row.BF01_kappa, ...
                 row.BF01_theta}];
        end
    end

    summary_table = cell2table(entries, ...
        'VariableNames', {'ModelType','GapMeasure','alpha_mean','kappa_mean','theta_mean','BF01_kappa','BF01_theta'});
end