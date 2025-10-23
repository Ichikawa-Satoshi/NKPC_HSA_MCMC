function data = func_load_data()
%LOAD_DATA Replicates the Python data prep in MATLAB using tables/timetables.
% Returns a MATLAB timetable with quarterly observations and the same columns
% as your pandas pipeline, with *_prev lagged variables and rows with NaNs removed.

    %% ---- Helper inline funcs ----
    yoy_pct = @(x) 100*(x./lag1(x,4) - 1);                % exact YoY pct change
    log_yoy = @(x) 100*(log(x) - lag1(log(x),4));         % log-diff YoY
    qstart  = @(y,q) datetime(y, 3*(q-1)+1, 1);           % quarter start
    toTT    = @(t) table2timetable(t, 'RowTimes', 'DATE');

    %% ---- SPF data (xlsx) ----
    spf = readtable('./data/inflation/Inflation.xlsx');
    spf.DATE = qstart(spf.YEAR, spf.QUARTER);
    spf.Epi_spf_gdp = spf.INFPGDP1YR;
    spf.Epi_spf_cpi = spf.INFCPI1YR;
    spf = spf(:, {'DATE','Epi_spf_gdp','Epi_spf_cpi'});
    tt_spf = toTT(spf);

    %% ---- CPI headline (CSV, quarterly, YoY%) ----
    cpi = readtable('./data/inflation/CPIAUCSL.csv');
    if ~isdatetime(cpi.DATE), cpi.DATE = datetime(cpi.DATE, 'InputFormat','yyyy-MM-dd'); end
    tt_cpi_m = toTT(cpi(:, {'DATE','CPIAUCSL'}));
    tt_cpi_q = retime(tt_cpi_m, 'regular', 'mean', 'TimeStep', calquarters(1));
    tt_cpi_q.pi_cpi = yoy_pct(tt_cpi_q.CPIAUCSL);
    tt_cpi_q = tt_cpi_q(:, 'pi_cpi');

    %% ---- Core CPI (CSV, log YoY) ----
    core = readtable('./data/inflation/CPILFESL.csv');
    if any(strcmpi(core.Properties.VariableNames, 'observation_date'))
        core.DATE = datetime(core.observation_date, 'InputFormat','yyyy-MM-dd');
    elseif ~any(strcmpi(core.Properties.VariableNames,'DATE'))
        error('CPILFESL.csv needs DATE or observation_date column.');
    end
    tt_core = toTT(core(:, {'DATE','CPILFESL'}));
    tt_core.pi_cpi_core = log_yoy(tt_core.CPILFESL);
    tt_core = tt_core(:, 'pi_cpi_core');

    %% ---- PCE headline (CSV, log YoY) ----
    pce = readtable('./data/inflation/PCEPI.csv');
    pce.DATE = datetime(pce.observation_date, 'InputFormat','yyyy-MM-dd');
    tt_pce = toTT(pce(:, {'DATE','PCEPI'}));
    tt_pce.pi_pce = log_yoy(tt_pce.PCEPI);
    tt_pce = tt_pce(:, 'pi_pce');

    %% ---- PCE core (CSV, log YoY) ----
    pce_core = readtable('./data/inflation/PCEPILFE.csv');
    pce_core.DATE = datetime(pce_core.observation_date, 'InputFormat','yyyy-MM-dd');
    tt_pce_core = toTT(pce_core(:, {'DATE','PCEPILFE'}));
    tt_pce_core.pi_pce_core = log_yoy(tt_pce_core.PCEPILFE);
    tt_pce_core = tt_pce_core(:, 'pi_pce_core');

    %% ---- HHI / number of firms (annual -> quarterly cubic interp) ----
    hhi = readtable('./data/competition/BN_N_26.csv');
    hhi.N = hhi.original_series;
    hhi.year = year(datetime(hhi.date, 'InputFormat','yyyy-MM-dd'));
    y = year(datetime(hhi.year,1,1));
    hhi.year = datetime(y,1,1);
    hhi = table2timetable(hhi, 'RowTimes','year');
    tt_hhi_q = retime(hhi(:, 'N'), 'regular', 'pchip', 'TimeStep', calquarters(1));
    tt_hhi_q.DATE = tt_hhi_q.Properties.RowTimes;
    tt_hhi_q = movevars(tt_hhi_q, {'DATE','N'}, 'Before',1);           
    tt_hhi_q = tt_hhi_q(:, {'N'});

    %% ---- N_BN (annual -> quarterly cubic interp) ----
    nb = readtable('./data/competition/BN_N_26.csv');
    nb.N_BN = nb.cycle;
    nb.year = year(datetime(nb.date, 'InputFormat','yyyy-MM-dd'));
    nb.DATE = datetime(nb.year,1,1);
    nb = nb(:, {'DATE','N_BN'});
    nb = nb(~any(ismissing(nb),2), :);
    tt_nb = toTT(nb);
    tt_nb_q = retime(tt_nb, 'regular', 'pchip', 'TimeStep', calquarters(1));
    tt_nb_q = tt_nb_q(:, 'N_BN');
    %% N projected
    n_proj =  readtable('./data/num_firm/projected_n_bn.csv');
    n_proj.N_p_BN = n_proj.cycle;
    n_proj.year = year(datetime(n_proj.date, 'InputFormat','yyyy-MM-dd'));
    n_proj.DATE = datetime(n_proj.year,1,1);
    n_proj = n_proj(:, {'DATE','N_p_BN'});
    tt_n_proj = toTT(n_proj);
    %% ---- Markup (levels) ----
    mk = readtable('./data/markup/nekarda_ramey_markups.xlsx');
    mk.DATE = datetime(mk.qdate, 'InputFormat','yyyy-MM-dd');
    mk.markup = mk.mu_bus;
    mk = mk(:, {'DATE','markup'});
    mk = mk(~any(ismissing(mk),2), :);
    tt_mk = toTT(mk);

    %% ---- Detrended Markup (BN inverse) ----
    mk_bn = readtable('./data/markup/BN_markup_inv.csv');
    mk_bn.markup_BN_inv = mk_bn.cycle;
    mk_bn.DATE = datetime(mk_bn.date, 'InputFormat','yyyy-MM-dd');
    mk_bn = mk_bn(:, {'DATE','markup_BN_inv'});
    mk_bn = mk_bn(~any(ismissing(mk_bn),2), :);
    tt_mk_bn = toTT(mk_bn);

    %% ---- Unemployment gap (NROU - UNRATENSA) ----
    nairu = readtable('./data/unemp_gap/NROU.csv');
    nairu.DATE = datetime(nairu.observation_date, 'InputFormat','yyyy-MM-dd');
    unemp = readtable('./data/unemp_gap/UNRATENSA.csv');
    unemp.DATE = datetime(unemp.observation_date, 'InputFormat','yyyy-MM-dd');

    tt_nairu = toTT(nairu(:, {'DATE','NROU'}));
    tt_unemp = toTT(unemp(:, {'DATE','UNRATENSA'}));
    tt_gap = synchronize(tt_nairu, tt_unemp, 'union');
    tt_gap.unemp_gap = tt_gap.NROU - tt_gap.UNRATENSA;
    tt_gap = tt_gap(:, 'unemp_gap');
    tt_gap = rmmissing(tt_gap);

    %% ---- Output gap data (BN filter) ----
    out = readtable('./data/output_gap/BN_filter_GDPC1_quaterly.csv');
    out.output_BN = out.GDPC1_transformed_series;
    out.output_gap_BN = out.cycle;
    out.output = log(out.GDPC1_original_series * 0.01);
    out.DATE = datetime(out.date, 'InputFormat','yyyy-MM-dd');
    out.output_trend_BN = out.output_BN - out.output_gap_BN;
    out = out(:, {'DATE','output_BN','output_gap_BN','output_trend_BN','output'});
    out = out(~any(ismissing(out),2), :);
    tt_out = toTT(out);

    %% ---- One-year inflation expectations (quarterly mean, *100) ----
    epi = readtable('./data/inflation/one_year_inflation_expectation.csv');
    epi.DATE = datetime(epi.Date, 'InputFormat','yyyy-MM-d');
    if ~isnumeric(epi.Epi)
        tmp = string(epi.Epi);
        parts = split(tmp, ","); hasTwo = width(parts) >= 2;
        val = nan(height(epi),1);
        val(hasTwo) = str2double(parts(hasTwo,2));
        val(~hasTwo) = str2double(tmp(~hasTwo));
        epi_val = val * 100;
    else
        epi_val = epi.Epi * 100;
    end
    tt_epi_m = toTT(table(epi.DATE, epi_val, 'VariableNames', {'DATE','Epi'}));
    tt_epi = retime(tt_epi_m, 'regular', 'mean', 'TimeStep', calquarters(1));

    %% ---- Oil prices (WTI deflated by CPI, log YoY) ----
    oil = readtable('./data/others/WTISPLC_CPIAUCSL.csv');
    oil.DATE = datetime(oil.observation_date, 'InputFormat','yyyy-MM-dd');
    tt_oil = toTT(oil(:, {'DATE','WTISPLC_CPIAUCSL'}));
    tt_oil.log_oil = log(tt_oil.WTISPLC_CPIAUCSL);
    tt_oil.oil = tt_oil.log_oil - lag1(tt_oil.log_oil, 4);
    tt_oil = tt_oil(:, 'oil');

    %% ---- Merge all by DATE (outer union) ----    
    data = synchronize( ...
        tt_cpi_q, tt_epi, tt_spf, tt_core, tt_out, tt_hhi_q, tt_nb_q, ...
        tt_mk, tt_pce_core, tt_pce, tt_mk_bn, tt_gap, tt_oil, ...
        'union');

    %% ---- Lags ----
    data.pi_cpi_prev       = lag1(data.pi_cpi, 1);
    data.pi_cpi_core_prev  = lag1(data.pi_cpi_core, 1);
    data.pi_pce_prev       = lag1(data.pi_pce, 1);
    data.pi_pce_core_prev  = lag1(data.pi_pce_core, 1);

    %% ---- Drop rows with any NaNs (like pandas dropna) ----
    data = rmmissing(data);
end

%% ---- Local helper: simple lag with k periods for vectors/timetable vars ----
function y = lag1(x, k)
    if istimetable(x)
        v = x.Variables;
        yv = [nan(k, size(v,2)); v(1:end-k, :)];
        y = array2timetable(yv, 'RowTimes', x.Properties.RowTimes, ...
            'VariableNames', x.Properties.VariableNames);
    else
        y = [nan(k, size(x,2)); x(1:end-k, :)];
    end
end