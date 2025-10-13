function data = func_load_data()
% === Data preparation ===
%% CPI
data_cpi = readtable("../../data/inflation/CPIAUCSL.csv");
data_cpi.DATE = datetime(data_cpi.DATE,'InputFormat','yyyy-MM-dd');
data_cpi = table2timetable(data_cpi);
data_cpi = retime(data_cpi,'quarterly','mean');
% Store back to table
data_cpi = timetable2table(data_cpi);
% Year-over-year (4-quarter) inflation rate in %
C = data_cpi.CPIAUCSL;
L4 = lagmatrix(C,4);                   
pi_yoy = 100 * (C ./ L4 - 1);
data_cpi.pi = pi_yoy;
%% HHI
data_HHI_annual = readtable("../../data/competition/hhi_from_Gustavo_2019.csv");
data_HHI_annual.HHI = data_HHI_annual.Value * 0.0001;
data_HHI_annual.DATE = datetime(data_HHI_annual.Year,1,1);
data_HHI_annual = table2timetable(data_HHI_annual,'RowTimes','DATE');
% Convert to quarterly frequency
data_HHI = retime(data_HHI_annual,'quarter');
% Linear interpolation
data_HHI.HHI = fillmissing(data_HHI.HHI,'linear');
data_HHI.N = 1 ./ data_HHI.HHI;
data_HHI = timetable2table(data_HHI);
%% De-trended Markup (BN filter)
data_markup_BN = readtable("../../data/markup/BN_markup_inv.csv");
data_markup_BN.DATE = datetime(data_markup_BN.date,'InputFormat','yyyy-MM-dd');
data_markup_BN.markup_BN_inv = data_markup_BN.cycle;
data_markup_BN = data_markup_BN(:,{'DATE','markup_BN_inv'});
data_markup_BN = rmmissing(data_markup_BN);
%% Unemployment gap
data_nairu = readtable("../../data/unemp_gap/NROU.csv");
data_nairu.DATE = datetime(data_nairu.observation_date,'InputFormat','yyyy-MM-dd');
data_unemp = readtable("../../data/unemp_gap/UNRATENSA.csv");
data_unemp.DATE = datetime(data_unemp.observation_date,'InputFormat','yyyy-MM-dd');
% Merge NAIRU and unemployment
data_unempgap = outerjoin(data_nairu,data_unemp,'Keys','DATE','MergeKeys',true);
% Compute unemployment gap
data_unempgap.unemp_gap = data_unempgap.NROU - data_unempgap.UNRATENSA;
data_unempgap = rmmissing(data_unempgap(:,{'DATE','unemp_gap'}));

%% Output gap (BN filter)
data_output = readtable("../../data/output_gap/BN_filter_GDPC1_quaterly.csv");
data_output.DATE = datetime(data_output.date,'InputFormat','yyyy-MM-dd');
data_output.output_BN = data_output.GDPC1_transformed_series;
data_output.output_gap_BN = data_output.cycle;
data_output.output = log(data_output.GDPC1_original_series * 0.01);
data_output.output_trend_BN = data_output.output_BN - data_output.output_gap_BN;
data_output = data_output(:,{'DATE','output_BN','output_gap_BN','output_trend_BN','output'});
data_output = rmmissing(data_output);
%% Inflation expectations
data_Epi = readtable("../../data/inflation/Epi.xlsx");
data_Epi.DATE = datetime(data_Epi.("DATE"),'InputFormat','yyyy-MM-dd');
data_Epi = table2timetable(data_Epi);
data_Epi = retime(data_Epi,'quarter','mean');
data_Epi.Epi = 100 * data_Epi.("Epi");
data_Epi = timetable2table(data_Epi(:,{'Epi'}));
%% Merge all datasets
data = outerjoin(data_cpi,data_Epi,'Keys','DATE','MergeKeys',true);
data = outerjoin(data,data_output,'Keys','DATE','MergeKeys',true);
data = outerjoin(data,data_HHI,'Keys','DATE','MergeKeys',true);
data = outerjoin(data,data_markup_BN,'Keys','DATE','MergeKeys',true);
data = outerjoin(data,data_unempgap,'Keys','DATE','MergeKeys',true);
% Create lagged variables
data.pi_prev = [NaN; data.pi(1:end-1)];
% Drop rows with missing values
data = rmmissing(data);
end