function [results, models] = func_ols_ces_system(data, pi, Epi, pi_prev, oil, IncludeOil)
%FUNC_OLS_CES_SYSTEM Run OLS specs (CES) optionally including oil variable.
%   [RESULTS, MODELS] = func_ols_ces_system(DATA, pi, Epi, pi_prev, IncludeOil)

    if nargin < 5
        IncludeOil = false;
    end

    % ----- candidate x's -----
    specs = {
        "Output gap (BN)",        data.output_gap_BN;
        "Unemployment gap",       data.unemp_gap;
        "Inverse of markup (BN)", data.markup_BN_inv
    };

    % ----- base table -----
    if IncludeOil
        baseTbl = table(pi, Epi, pi_prev, oil, ...
            'VariableNames', {'pi','Epi','pi_prev','oil'});
    else
        baseTbl = table(pi, Epi, pi_prev, ...
            'VariableNames', {'pi','Epi','pi_prev'});
    end

    % ----- loop over specs -----
    results = table();
    models  = struct([]);
    for i = 1:size(specs,1)
        label = specs{i,1};
        x     = specs{i,2};

        T = [baseTbl, table(x,'VariableNames',{'x'})];
        T = rmmissing(T);       
        formula = "pi ~ -1 + pi_prev + Epi + x";
        if IncludeOil
            formula = formula + " + oil";
        end

        mdl = fitlm(T, formula);
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

    % ----- rounding -----
    results.Coef   = round(results.Coef, 4);
    results.SE     = round(results.SE, 4);
    results.pValue = round(results.pValue, 4);
    results.R2     = round(results.R2, 3);
    results.AdjR2  = round(results.AdjR2, 3);
end