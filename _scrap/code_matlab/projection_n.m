% Read CSV files
clear;
data_N = readtable('data/num_firm/num_firm.xlsx');
data_N.Nb = data_N.birth * 1000;
data_N.Nd = data_N.death * 1000;
data_N.date = data_N.period;
% Calculate Nchange
data_N.Nchange = data_N.Nb - data_N.Nd;
% Filter by date
data_N = data_N(data_N.date >= datetime('1993-04-01'), :);
% Calculate cumulative sum with initial value
initial_value = 5387000;
Nchange_shifted = [0; data_N.Nchange(1:end-1)];
data_N.N = initial_value + cumsum(Nchange_shifted);
data_N = data_N(:, {'date', 'N'});
data_N = table2timetable(data_N, 'RowTimes', 'date');
data_N_jan = data_N(month(data_N.date) == 1, :);

hhi = readtable('./data/competition/BN_N_26.csv');
hhi.date = datetime(hhi.date, 'InputFormat','yyyy-MM-dd');  
hhi.N_annual = hhi.original_series;                         
hhi = hhi(:, {'date','N_annual'});
hhi = table2timetable(hhi, 'RowTimes','date');
n_annual_tt = synchronize(hhi, data_N_jan, 'intersection');

%% Projection
dN_annual          = diff(n_annual_tt.N_annual);
dlogFirms_annual   = diff(log(n_annual_tt.N));
mdl_annual = fitlm(dlogFirms_annual, dN_annual, 'Intercept', true);
beta = mdl_annual.Coefficients.Estimate(1);
disp(mdl_annual)
dlogFirms_q = [NaN; diff(log(data_N.N))];
dN_q_hat    = beta * dlogFirms_q;
N_q_hat0 = cumsum(fillmissing(dN_q_hat,'constant',0));
data_N.N_q_hat0 = N_q_hat0;
output_path = 'data/num_firm/N_q_hat0_projection.csv';
writetimetable(data_N(:, {'N_q_hat0'}), output_path);