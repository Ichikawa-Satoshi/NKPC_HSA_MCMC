data = load_data();
% plot_dashboard(data)
%% Sampling
burn_in = 2000;
sample = 18000;

% helper to subset all fields by logical index
subset = @(s,idx) struct( ...
    'pi',            s.pi(idx), ...
    'pi_prev',       s.pi_prev(idx), ...
    'Epi',           s.Epi(idx), ...
    'output_gap_BN', s.output_gap_BN(idx), ...
    'unemp_gap',     s.unemp_gap(idx), ...
    'markup_inv',    s.markup_inv(idx), ...
    'markup_BN_inv', s.markup_BN_inv(idx), ...
    'markup',        s.markup(idx), ...
    'N',             s.N(idx), ...
    'DATE',          s.DATE(idx) ...
);
%% ===== CES =====
results_output = estimate_ces(data.pi, data.pi_prev, data.Epi, data.output_gap_BN, burn_in, sample);
results_unemp_gap     = estimate_ces(data.pi, data.pi_prev, data.Epi, data.unemp_gap,     burn_in, sample);
results_markup        = estimate_ces(data.pi, data.pi_prev, data.Epi, data.markup_inv,    burn_in, sample);
results_markup_BN     = estimate_ces(data.pi, data.pi_prev, data.Epi, data.markup_BN_inv, burn_in, sample);
results_list_ces = {results_output, results_unemp_gap, results_markup, results_markup_BN};
model_names_ces  = {'CES: Output Gap (BN)','CES: Unemp Gap','CES: Markup','CES: Markup (BN)'};
outDir = 'fig';
plot_ces_sampling(results_list_ces, model_names_ces);
f1 = gcf;
exportgraphics(f1, fullfile(outDir,'ces_sampling.png'), 'Resolution',300);
plot_ces_results(results_list_ces, model_names_ces);
f2 = gcf;
exportgraphics(f2, fullfile(outDir,'ces_results.png'), 'Resolution',300);
%% SDDR
ces_sddr = table('Size',[numel(results_list_ces) 2], ...
               'VariableTypes', {'string','double'}, ...
               'VariableNames', {'Model','SDDR_kappa'});
for i = 1:numel(results_list_ces)
    res = results_list_ces{i};
    sddr_val = compute_sddr_ces(res);   
    ces_sddr.Model(i) = model_names_ces{i};
    ces_sddr.SDDR_kappa(i) = sddr_val;
end
vals = ces_sddr.SDDR_kappa';
ces_sddr_wide = array2table(vals, ...
    'VariableNames', strrep(ces_sddr.Model,"CES: ",""));
disp(ces_sddr_wide)
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname  = fullfile(outDir, 'ces_sddr.md');
fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);
fprintf(fid, '| %s |\n', strjoin(ces_sddr_wide.Properties.VariableNames, ' | '));
fprintf(fid, '|%s|\n', strjoin(repmat({'---'},1,width(ces_sddr_wide)),'|'));
row = strings(1, width(ces_sddr_wide));
for j = 1:width(ces_sddr_wide)
    val = ces_sddr_wide{1,j};
    if isnan(val)
        row(j) = "";
    else
        row(j) = string(sprintf('%.5f', val));
    end
end
fprintf(fid, '| %s |\n', strjoin(row, ' | '));
fclose(fid);
%% ML
ces_ml = table('Size',[numel(results_list_ces) 2], ...
               'VariableTypes', {'string','double'}, ...
               'VariableNames', {'Model','ML'});
for i = 1:numel(results_list_ces)
    res = results_list_ces{i};
    ml_val = res.marginal_likelihood.log_ml;   
    ces_ml.Model(i) = model_names_ces{i};
    ces_ml.ML(i) = ml_val;
end
vals = ces_ml.ML';
ces_ml_wide = array2table(vals, ...
    'VariableNames', strrep(ces_ml.Model,"CES: ",""));
disp(ces_ml_wide)
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname  = fullfile(outDir, 'ces_ml.md');
fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);
fprintf(fid, '| %s |\n', strjoin(ces_sddr_wide.Properties.VariableNames, ' | '));
fprintf(fid, '|%s|\n', strjoin(repmat({'---'},1,width(ces_sddr_wide)),'|'));
row = strings(1, width(ces_ml_wide));
for j = 1:width(ces_ml_wide)
    val = ces_ml_wide{1,j};
    if isnan(val)
        row(j) = "";
    else
        row(j) = string(sprintf('%.5f', val));
    end
end
fprintf(fid, '| %s |\n', strjoin(row, ' | '));
fclose(fid);
%% ===== HSA =====
results_HSA_output    = estimate_hsa(data.pi, data.pi_prev, data.Epi, data.output_gap_BN, data.N, burn_in, sample);
results_HSA_markup    = estimate_hsa(data.pi, data.pi_prev, data.Epi, data.markup,        data.N, burn_in, sample);
results_HSA_markup_BN = estimate_hsa(data.pi, data.pi_prev, data.Epi, data.markup_BN_inv, data.N, burn_in, sample);
results_HSA_unemp_gap = estimate_hsa(data.pi, data.pi_prev, data.Epi, data.unemp_gap,     data.N, burn_in, sample);
results_list_hsa = {results_HSA_output, results_HSA_unemp_gap, results_HSA_markup, results_HSA_markup_BN};
model_names_hsa  = {'HSA: Output Gap (BN)','HSA: Unemp Gap','HSA: Markup','HSA: Markup (BN)'};

%% ML
hsa_ml = table('Size',[numel(results_list_hsa) 2], ...
               'VariableTypes', {'string','double'}, ...
               'VariableNames', {'Model','ML'});
for i = 1:numel(results_list_hsa)
    res = results_list_hsa{i};
    ml_val = res.marginal_likelihood.log_ml;   
    hsa_ml.Model(i) = model_names_ces{i};
    hsa_ml.ML(i) = ml_val;
end
vals = hsa_ml.ML';
hsa_ml_wide = array2table(vals, ...
    'VariableNames', strrep(hsa_ml.Model,"HSA: ",""));
disp(hsa_ml_wide)
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname  = fullfile(outDir, 'hsa_ml.md');
fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);
fprintf(fid, '| %s |\n', strjoin(hsa_ml_wide.Properties.VariableNames, ' | '));
fprintf(fid, '|%s|\n', strjoin(repmat({'---'},1,width(hsa_ml_wide)),'|'));
row = strings(1, width(hsa_ml_wide));
for j = 1:width(hsa_ml_wide)
    val = hsa_ml_wide{1,j};
    if isnan(val)
        row(j) = "";
    else
        row(j) = string(sprintf('%.5f', val));
    end
end
fprintf(fid, '| %s |\n', strjoin(row, ' | '));
fclose(fid);
%% Plot results
outDir = 'fig';
plot_hsa_sampling(results_list_hsa, model_names_hsa);
f3 = gcf;
exportgraphics(f3, fullfile(outDir,'hsa_sampling.png'), 'Resolution',300);
plot_hsa_results(results_list_hsa, model_names_hsa);
f4 = gcf;
exportgraphics(f4, fullfile(outDir,'hsa_results.png'), 'Resolution',300);
%% Decomposition check
plot_decomposition(results_list_hsa,  model_names_hsa,  data.N,  data.DATE);
f5 = gcf;
exportgraphics(f5, fullfile(outDir,'decomposition.png'), 'Resolution',300);
%% SDDR
hsa_sddr = table('Size',[numel(results_list_hsa) 2], ...
               'VariableTypes', {'string','double'}, ...
               'VariableNames', {'Model','SDDR_kappa'});
for i = 1:numel(results_list_hsa)
    res = results_list_hsa{i};                 
    s   = compute_sddr_hsa(res);               % struct: .kappa, .theta
    hsa_sddr.Model(i)      = model_names_hsa{i};
    hsa_sddr.SDDR_kappa(i) = s.kappa;
    hsa_sddr.SDDR_theta(i) = s.theta;
end
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname = fullfile(outDir, 'hsa_sddr.md');

fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);
fprintf(fid, '| Model | SDDR_kappa | SDDR_theta |\n');
fprintf(fid, '|---|---|---|\n');
for i = 1:height(hsa_sddr)
    fprintf(fid, '| %s | %.5f | %.5f |\n', ...
        hsa_sddr.Model(i), ...
        hsa_sddr.SDDR_kappa(i), ...
        hsa_sddr.SDDR_theta(i));
end
disp(hsa_sddr)
fclose(fid);
%% output table
results_list = {results_output, results_HSA_output,...
                results_unemp_gap, results_HSA_unemp_gap,...
                results_markup, results_HSA_markup,...
                results_markup_BN,results_HSA_markup_BN};
model_names  = {'CES: Output Gap (BN)','HSA: Output Gap (BN)',...
                'CES: Unemp Gap','HSA: Unemp Gap',...
                'CES: Markup','HSA: Markup', ...
                'CES: Markup (BN)', 'HSA: Markup (BN)'};
params = {'alpha','kappa','theta','rho1','rho2','n'};
T = table('Size',[numel(results_list) numel(params)+1], ...
          'VariableTypes', ['string', repmat({'double'},1,numel(params))], ...
          'VariableNames', ['Model', params]);
for i = 1:numel(results_list)
    T.Model(i) = model_names{i};
    for j = 1:numel(params)
        param = params{j};
        if isfield(results_list{i}, param) && isfield(results_list{i}.(param), 'mean')
            T{i, param} = results_list{i}.(param).mean;
        else
            T{i, param} = NaN;
        end
    end
end
disp(T)
fid = fopen('../estimated_coef/results_table.md','w');
varNames = T.Properties.VariableNames;
fprintf(fid, '| %s |\n', strjoin(varNames, ' | '));
fprintf(fid, '|%s|\n', strjoin(repmat({'---'},1,numel(varNames)),'|'));
for i = 1:height(T)
    row = strings(1, numel(varNames));  % ← string 
    for j = 1:numel(varNames)
        val = T{i,j};
        if isstring(val) || ischar(val)
            row(j) = string(val);
        elseif isnumeric(val)
            if isnan(val)
                row(j) = "";
            else
                row(j) = string(sprintf('%.6f', val));
            end
        else
            row(j) = "";
        end
    end
    fprintf(fid, '| %s |\n', strjoin(row, ' | '));
end
fclose(fid);