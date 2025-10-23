function [results, models] = func_ols_hsa_system(data, pi, Epi, pi_prev, Nhat)
%REGRESSION_SYSTEM Run OLS specs for inflation without Newey-West.
%   [RESULTS, MODELS] = REGRESSION_SYSTEM(DATA, OPTS)
%   DATA : quarterly timetable from load_data()
%   The dependent variable is pi_cpi_core.
%   Baseline regressors: Epi_spf_cpi, pi_cpi_core_prev, and x where:
%       1) output_gap_BN
%       2) unemp_gap
%       3) markup_BN_inv
    specs = {
        "Output gap (BN)",        data.output_gap_BN;
        "Unemployment gap",       data.unemp_gap;
        "Inverse of markup (BN)", data.markup_BN_inv
    };
    results = table();
    models  = struct([]);
    baseTbl = table(pi, Epi, pi_prev,Nhat, 'VariableNames', {'pi','Epi','pi_prev', 'Nhat'});
    for i = 1:size(specs,1)
        label = specs{i,1};
        x     = specs{i,2};
        T = [baseTbl, table(x,'VariableNames',{'x'})];
        T = rmmissing(T);
        mdl = fitlm(T, "pi ~ -1 + pi_prev + Epi + x + Nhat");
        models(i).name  = char(label);
        models(i).model = mdl;
        ct  = mdl.Coefficients;
        R2  = mdl.Rsquared.Ordinary;
        R2a = mdl.Rsquared.Adjusted;
        tmp = table( repmat(label,height(ct),1), ...
                     string(ct.Properties.RowNames), ...
                     ct.Estimate, ct.SE, ct.pValue, ...
                     repmat(R2,height(ct),1), repmat(R2a,height(ct),1), ...
                     'VariableNames', {'Model','Variable','Coef','SE','pValue','R2','AdjR2'});
        results = [results; tmp]; %#ok<AGROW>
    end
    % pretty rounding
    results.Coef   = round(results.Coef, 4);
    results.SE     = round(results.SE, 4);
    results.pValue = round(results.pValue, 4);
    results.R2     = round(results.R2, 3);
    results.AdjR2  = round(results.AdjR2, 3);
end