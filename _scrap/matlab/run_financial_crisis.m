
data = load_data();
%% Sampling
burn_in = 5000;
sample = 10000;
%% ===== split settings =====
crisis_start = datetime(2008,4,1);
crisis_end   = datetime(2009,4,1);

if isdatetime(data.DATE)
    idx_pre  = data.DATE < crisis_start;
    idx_post = data.DATE > crisis_end;
else
    crisis_snum = datenum(crisis_start);
    crisis_enum = datenum(crisis_end);
    idx_pre  = data.DATE < crisis_snum;
    idx_post = data.DATE > crisis_enum;
end

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

data_pre  = subset(data, idx_pre);
data_post = subset(data, idx_post);

%% ===== CES (pre) =====
results_output_pre = estimate_ces(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.output_gap_BN, burn_in, sample);
results_unemp_gap_pre     = estimate_ces(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.unemp_gap,     burn_in, sample);
results_markup_pre        = estimate_ces(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.markup_inv,    burn_in, sample);
results_markup_BN_pre     = estimate_ces(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.markup_BN_inv, burn_in, sample);

%% ===== CES (post) =====
results_output_post = estimate_ces(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.output_gap_BN, burn_in, sample);
results_unemp_gap_post  = estimate_ces(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.unemp_gap,     burn_in, sample);
results_markup_post     = estimate_ces(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.markup_inv,    burn_in, sample);
results_markup_BN_post  = estimate_ces(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.markup_BN_inv, burn_in, sample);

%% 
results_list_ces  = {results_output_pre, results_output_post, ...
                     results_unemp_gap_pre, results_unemp_gap_post, ...
                     results_markup_pre, results_markup_post, ...
                     results_markup_BN_pre, results_markup_BN_post};
model_names_ces  = {'CES: Output — before', 'CES: Output — after', ...
                 'CES: Unemp Gap — before', 'CES: Unemp Gap — after', ...
                 'CES: Markup — before', 'CES: Markup — after', ... 
                 'CES: Markup (BN) — before','CES: Markup (BN) — after'};
plot_ces_results_compare(results_list_ces,model_names_ces);
%% CES SDDR
ces_sddr = table('Size',[numel(results_list_ces) 2], ...
               'VariableTypes', {'string','double'}, ...
               'VariableNames', {'Model','SDDR_kappa'});
for i = 1:numel(results_list_ces)
    res = results_list_ces{i};
    sddr_val = compute_sddr_ces(res);   
    ces_sddr.Model(i) = model_names_ces{i};
    ces_sddr.SDDR_kappa(i) = sddr_val;
end
ces_sddr.Var  = strtrim(extractBetween(ces_sddr.Model, "CES: ", " — "));
ces_sddr.Time = strtrim(extractAfter(ces_sddr.Model,  "— "));  % "before" or "after"
ces_sddr_wide = unstack(ces_sddr, "SDDR_kappa", "Time", "GroupingVariables", "Var");
if any(strcmp(ces_sddr_wide.Properties.VariableNames,'before')) && ...
   any(strcmp(ces_sddr_wide.Properties.VariableNames,'after'))
    ces_sddr_wide = movevars(ces_sddr_wide, {'before','after'}, 'After', 'Var');
    ces_sddr_wide.Properties.VariableNames = {'Var','Before','After'};
end
order = categorical(ces_sddr_wide.Var, ["Output","Unemp Gap","Markup","Markup (BN)"], 'Ordinal',true);
[~,ix] = sort(order);
ces_sddr_wide = ces_sddr_wide(ix,:);
disp(ces_sddr_wide)
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname = fullfile(outDir, 'ces_sddr_table.md');
fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);
fprintf(fid, '| CES | Before(kappa) | After(kappa) |\n');
fprintf(fid, '|---|---|---|\n');
for i = 1:height(ces_sddr_wide)
    fprintf(fid, '| %s | %.5f | %.5f |\n', ...
        ces_sddr_wide.Var(i), ...
        ces_sddr_wide.Before(i), ...
        ces_sddr_wide.After(i));
end
fclose(fid);
%% ML (make a Var × Before/After table just like SDDR)
ces_ml = table('Size',[numel(results_list_ces) 2], ...
               'VariableTypes', {'string','double'}, ...
               'VariableNames', {'Model','ML'});
for i = 1:numel(results_list_ces)
    res = results_list_ces{i};
    ml_val = res.marginal_likelihood.log_ml;
    ces_ml.Model(i) = model_names_ces{i};
    ces_ml.ML(i)    = ml_val;
end

% Parse "Var" and "Time" from the model name: "CES: <Var> — <Time>"
ces_ml.Var  = strtrim(extractBetween(ces_ml.Model, "CES: ", " — "));
ces_ml.Time = strtrim(extractAfter(ces_ml.Model,  "— "));  % "before" or "after"

% Wide form: columns Before / After
ces_ml_wide = unstack(ces_ml, "ML", "Time", "GroupingVariables", "Var");
if any(strcmp(ces_ml_wide.Properties.VariableNames,'before')) && ...
   any(strcmp(ces_ml_wide.Properties.VariableNames,'after'))
    ces_ml_wide = movevars(ces_ml_wide, {'before','after'}, 'After', 'Var');
    ces_ml_wide.Properties.VariableNames = {'Var','Before','After'};
end

% Order rows to match your preferred order
order = categorical(ces_ml_wide.Var, ["Output","Unemp Gap","Markup","Markup (BN)"], 'Ordinal', true);
[~,ix] = sort(order);
ces_ml_wide = ces_ml_wide(ix,:);

disp(ces_ml_wide)

% Write Markdown table (same look-and-feel as SDDR)
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname  = fullfile(outDir, 'ces_ml_table_fc.md');
fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);

fprintf(fid, '| CES | Before(ML) | After(ML) |\n');
fprintf(fid, '|---|---|---|\n');
for i = 1:height(ces_ml_wide)
    before_str = ifelse(isnan(ces_ml_wide.Before(i)), "", sprintf('%.5f', ces_ml_wide.Before(i)));
    after_str  = ifelse(isnan(ces_ml_wide.After(i)),  "", sprintf('%.5f', ces_ml_wide.After(i)));
    fprintf(fid, '| %s | %s | %s |\n', ces_ml_wide.Var(i), before_str, after_str);
end
fclose(fid);

% --- small inline helper ---
function out = ifelse(cond, a, b)
    if cond, out = a; else, out = b; end
end
%% ===== HSA (pre) =====
results_HSA_output_pre    = estimate_hsa(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.output_gap_BN, data_pre.N, burn_in, sample);
results_HSA_markup_pre    = estimate_hsa(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.markup,        data_pre.N, burn_in, sample);
results_HSA_markup_BN_pre = estimate_hsa(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.markup_BN_inv, data_pre.N, burn_in, sample);
results_HSA_unemp_gap_pre = estimate_hsa(data_pre.pi, data_pre.pi_prev, data_pre.Epi, data_pre.unemp_gap,     data_pre.N, burn_in, sample);

%% ===== HSA (post) =====
results_HSA_output_post    = estimate_hsa(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.output_gap_BN, data_post.N, burn_in, sample);
results_HSA_markup_post    = estimate_hsa(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.markup,        data_post.N, burn_in, sample);
results_HSA_markup_BN_post = estimate_hsa(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.markup_BN_inv, data_post.N, burn_in, sample);
results_HSA_unemp_gap_post = estimate_hsa(data_post.pi, data_post.pi_prev, data_post.Epi, data_post.unemp_gap,     data_post.N, burn_in, sample);
%% 
results_list_hsa  = {results_HSA_output_pre, results_HSA_output_post, ...
                 results_HSA_unemp_gap_pre, results_HSA_unemp_gap_post, ...
                 results_HSA_markup_pre, results_HSA_markup_post,...
                 results_HSA_markup_BN_pre, results_HSA_markup_BN_post};
model_names_hsa   = {'HSA: Output — before', 'HSA: Output — after', ...
                 'HSA: Unemp Gap — before', 'HSA: Unemp Gap — after', ...
                 'HSA: Markup — before', 'HSA: Markup — after', ... 
                 'HSA: Markup (BN) — before','HSA: Markup (BN) — after'};
plot_hsa_results_compare(results_list_hsa, model_names_hsa);
%% ===== Decomposition plots (HSA only; pre/post) =====
results_list_pre  = {results_HSA_output_pre, results_HSA_unemp_gap_pre, results_HSA_markup_pre, results_HSA_markup_BN_pre};
model_names_pre   = {'HSA: Output — before', 'HSA: Unemp Gap — before', 'HSA: Markup — before', 'HSA: Markup (BN) — before'};
plot_decomposition(results_list_pre,  model_names_pre,  data_pre.N,  data_pre.DATE);
f1 = gcf;
exportgraphics(f1, fullfile("fig",'decomposition_pre.png'), 'Resolution',300);
results_list_post = {results_HSA_output_post, results_HSA_unemp_gap_post, results_HSA_markup_post, results_HSA_markup_BN_post};
model_names_post  = {'HSA: Output — after', 'HSA: Unemp Gap — after', 'HSA: Markup — after', 'HSA: Markup (BN) — after'};
plot_decomposition(results_list_post, model_names_post, data_post.N, data_post.DATE);
f2 = gcf;
exportgraphics(f2, fullfile("fig",'decomposition_post.png'), 'Resolution',300);
%% output table
results_list  = {results_output_pre, results_output_post, ...
                 results_HSA_output_pre, results_HSA_output_post, ...
                 results_unemp_gap_pre, results_unemp_gap_post, ...
                 results_HSA_unemp_gap_pre, results_HSA_unemp_gap_post, ...
                 results_markup_pre, results_markup_post, ...
                 results_HSA_markup_pre, results_HSA_markup_post, ...
                 results_markup_BN_pre, results_markup_BN_post, ...
                 results_HSA_markup_BN_pre, results_HSA_markup_BN_post};
model_names   = {'CES: Output — before', 'CES: Output — after', ...
                 'HSA: Output — before', 'HSA: Output — after', ...
                 'CES: Unemp Gap — before', 'CES: Unemp Gap — after', ...
                 'HSA: Unemp Gap — before', 'HSA: Unemp Gap — after', ...
                 'CES: Markup — before', 'CES: Markup — after', ... 
                 'HSA: Markup — before', 'HSA: Markup — after', ... 
                 'CES: Markup (BN) — before','CES: Markup (BN) — after',...
                 'HSA: Markup (BN) — before','HSA: Markup (BN) — after'};

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
fid = fopen('../estimated_coef/results_table_fc.md','w');
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
%% HSA SDDR (long table)
results_list_hsa  = {results_HSA_output_pre, results_HSA_output_post, ...
                     results_HSA_unemp_gap_pre, results_HSA_unemp_gap_post, ...
                     results_HSA_markup_pre, results_HSA_markup_post, ...
                     results_HSA_markup_BN_pre, results_HSA_markup_BN_post};

model_names_hsa   = {'HSA: Output — before', 'HSA: Output — after', ...
                     'HSA: Unemp Gap — before', 'HSA: Unemp Gap — after', ...
                     'HSA: Markup — before', 'HSA: Markup — after', ...
                     'HSA: Markup (BN) — before','HSA: Markup (BN) — after'};

hsa_sddr = table('Size',[numel(results_list_hsa) 3], ...
    'VariableTypes', {'string','double','double'}, ...
    'VariableNames', {'Model','SDDR_kappa','SDDR_theta'});

for i = 1:numel(results_list_hsa)
    res = results_list_hsa{i};                 
    s   = compute_sddr_hsa(res);               % struct: .kappa, .theta
    hsa_sddr.Model(i)      = model_names_hsa{i};
    hsa_sddr.SDDR_kappa(i) = s.kappa;
    hsa_sddr.SDDR_theta(i) = s.theta;
end
hsa_sddr.Var  = strtrim(extractBetween(hsa_sddr.Model, "HSA: ", " — "));
hsa_sddr.Time = strtrim(extractAfter( hsa_sddr.Model,  "— "));  % "before"/"after"
hsa_kappa_wide = unstack(hsa_sddr, "SDDR_kappa", "Time", "GroupingVariables", "Var");
hsa_kappa_wide = movevars(hsa_kappa_wide, {'before','after'}, 'After', 'Var');
hsa_kappa_wide.Properties.VariableNames = {'Var','Before_kappa','After_kappa'};
hsa_theta_wide = unstack(hsa_sddr, "SDDR_theta", "Time", "GroupingVariables", "Var");
hsa_theta_wide = movevars(hsa_theta_wide, {'before','after'}, 'After', 'Var');
hsa_theta_wide.Properties.VariableNames = {'Var','Before_theta','After_theta'};
hsa_sddr_wide = join(hsa_kappa_wide, hsa_theta_wide, 'Keys','Var');
order = categorical(hsa_sddr_wide.Var, ["Output","Unemp Gap","Markup","Markup (BN)"], 'Ordinal',true);
[~,ix] = sort(order);
hsa_sddr_wide = hsa_sddr_wide(ix,:);
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname  = fullfile(outDir,'hsa_sddr_table.md');
fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);
fprintf(fid, '| HSA | Before (kappa) | After (kappa) | Before (theta) | After (theta) |\n');
fprintf(fid, '|---|---|---|---|---|\n');
for i = 1:height(hsa_sddr_wide)
    fprintf(fid, '| %s | %.5f | %.5f | %.5f | %.5f |\n', ...
        hsa_sddr_wide.Var(i), ...
        hsa_sddr_wide.Before_kappa(i), hsa_sddr_wide.After_kappa(i), ...
        hsa_sddr_wide.Before_theta(i), hsa_sddr_wide.After_theta(i));
end
fclose(fid);
%% HSA ML (make a Var × Before/After table just like HSA SDDR)
hsa_ml = table('Size',[numel(results_list_hsa) 2], ...
               'VariableTypes', {'string','double'}, ...
               'VariableNames', {'Model','ML'});

for i = 1:numel(results_list_hsa)
    res = results_list_hsa{i};
    ml_val = res.marginal_likelihood.log_ml;
    hsa_ml.Model(i) = model_names_hsa{i};   
    hsa_ml.ML(i)    = ml_val;
end

% Parse Var / Time from "HSA: <Var> — <Time>"
hsa_ml.Var  = strtrim(extractBetween(hsa_ml.Model, "HSA: ", " — "));
hsa_ml.Time = strtrim(extractAfter( hsa_ml.Model,  "— "));  % "before" or "after"

% Wide form with Before / After columns
hsa_ml_wide = unstack(hsa_ml, "ML", "Time", "GroupingVariables", "Var");

% If both columns exist, reorder and rename them
if any(strcmp(hsa_ml_wide.Properties.VariableNames,'before')) && ...
   any(strcmp(hsa_ml_wide.Properties.VariableNames,'after'))
    hsa_ml_wide = movevars(hsa_ml_wide, {'before','after'}, 'After', 'Var');
    hsa_ml_wide.Properties.VariableNames = {'Var','Before','After'};
else
    % In case only one side is present, still keep consistent names
    if any(strcmp(hsa_ml_wide.Properties.VariableNames,'before'))
        hsa_ml_wide.Properties.VariableNames = {'Var','Before'};
        hsa_ml_wide.After = NaN(height(hsa_ml_wide),1);
    elseif any(strcmp(hsa_ml_wide.Properties.VariableNames,'after'))
        hsa_ml_wide.Properties.VariableNames = {'Var','After'};
        hsa_ml_wide.Before = NaN(height(hsa_ml_wide),1);
        hsa_ml_wide = movevars(hsa_ml_wide,'Before','After','Before');
    end
end

% Preferred row order
order = categorical(hsa_ml_wide.Var, ["Output","Unemp Gap","Markup","Markup (BN)"], 'Ordinal',true);
[~,ix] = sort(order);
hsa_ml_wide = hsa_ml_wide(ix,:);

disp(hsa_ml_wide)

% Write Markdown table: | HSA | Before(ML) | After(ML) |
outDir = '../estimated_coef';
if ~exist(outDir,'dir'); mkdir(outDir); end
fname  = fullfile(outDir, 'hsa_ml_table_fc.md');
fid = fopen(fname,'w');
assert(fid~=-1, 'Cannot open %s', fname);

fprintf(fid, '| HSA | Before(ML) | After(ML) |\n');
fprintf(fid, '|---|---|---|\n');
for i = 1:height(hsa_ml_wide)
    before_str = ifelse(isnan(hsa_ml_wide.Before(i)), "", sprintf('%.5f', hsa_ml_wide.Before(i)));
    after_str  = ifelse(isnan(hsa_ml_wide.After(i)),  "", sprintf('%.5f', hsa_ml_wide.After(i)));
    fprintf(fid, '| %s | %s | %s |\n', hsa_ml_wide.Var(i), before_str, after_str);
end
fclose(fid);
