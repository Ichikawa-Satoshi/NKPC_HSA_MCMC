import pandas as pd
import numpy as np
def load_data(): 
    # Data
    data_spf = pd.read_excel("./data/inflation/Inflation.xlsx")
    data_spf["DATE"] = pd.PeriodIndex.from_fields(year=data_spf["YEAR"], quarter=data_spf["QUARTER"], freq="Q").to_timestamp(how="S")
    data_spf["Epi_spf_gdp"] = data_spf["INFPGDP1YR"]
    data_spf["Epi_spf_cpi"] = data_spf["INFCPI1YR"]
    data_spf = data_spf[["DATE", "Epi_spf_gdp", "Epi_spf_cpi"]]

    # CPI data
    data_cpi = pd.read_csv("./data/inflation/CPIAUCSL.csv", parse_dates=["DATE"])
    data_cpi.set_index("DATE", inplace=True)
    data_cpi = data_cpi.resample("QS").mean()
    data_cpi['pi_cpi'] = data_cpi['CPIAUCSL'].pct_change(4) * 100
    data_cpi = data_cpi['pi_cpi']
    data_cpi = data_cpi.reset_index()
    data_cpi['DATE'] = pd.to_datetime(data_cpi['DATE'])

    # Core CPI
    data_corecpi = pd.read_csv("./data/inflation/CPILFESL.csv")
    data_corecpi["DATE"] = pd.to_datetime(data_corecpi["observation_date"])
    data_corecpi["log_pi"]= np.log(data_corecpi["CPILFESL"])
    data_corecpi['pi_cpi_core'] = data_corecpi["log_pi"].diff(4)*100

    # PCE 
    data_pi_pce = pd.read_csv("./data/inflation/PCEPI.csv")
    data_pi_pce["DATE"] = pd.to_datetime(data_pi_pce["observation_date"])
    data_pi_pce["log_pi_pce"]= np.log(data_pi_pce["PCEPI"])
    data_pi_pce['pi_pce'] = data_pi_pce["log_pi_pce"].diff(4)*100
    data_pi_pce = data_pi_pce[["DATE", "pi_pce"]]

    # Core PCE
    data_pi_pce_core = pd.read_csv("./data/inflation/PCEPILFE.csv")
    data_pi_pce_core["DATE"] = pd.to_datetime(data_pi_pce_core["observation_date"])
    data_pi_pce_core["log_pi_pce_core"]= np.log(data_pi_pce_core["PCEPILFE"])
    data_pi_pce_core['pi_pce_core'] = data_pi_pce_core["log_pi_pce_core"].diff(4)*100
    data_pi_pce_core = data_pi_pce_core[["DATE", "pi_pce_core"]]

    # Number of Firm data
    data_HHI_annual = pd.read_csv("./data/competition/hhi_from_Gustavo_2019.csv")
    data_HHI_annual["HHI"] = data_HHI_annual["Value"] * 0.0001
    data_HHI_annual['year'] = pd.to_datetime(data_HHI_annual['Year'], format='%Y') + pd.offsets.YearBegin(0)
    data_HHI_annual = data_HHI_annual.set_index('year')
    data_HHI = data_HHI_annual.resample('QS').asfreq()
    data_HHI['HHI'] = data_HHI['HHI'].interpolate(method='cubic')
    data_HHI = data_HHI.reset_index()
    data_HHI['DATE'] = pd.to_datetime(data_HHI['year'])
    data_HHI = data_HHI[["DATE", "HHI"]]
    data_HHI["N"] = 1/data_HHI["HHI"]

    # N BN
    data_N_BN = pd.read_csv("./data/competition/n_from_Gustavo_2019_BN.csv")
    data_N_BN['N_BN'] = data_N_BN["cycle"] * 0.01
    data_N_BN = data_N_BN[['date', "N_BN"]]
    data_N_BN["year"] = pd.to_datetime(data_N_BN["date"]).dt.year
    data_N_BN['DATE'] = pd.to_datetime(data_N_BN['year'], format='%Y') + pd.offsets.YearBegin(0)
    data_N_BN = data_N_BN[["DATE","N_BN"]].dropna()
    data_N_BN = data_N_BN.reset_index()
    data_N_BN = data_N_BN.set_index('DATE')
    data_N_BN = data_N_BN.resample('QS').asfreq()
    data_N_BN['N_BN'] = data_N_BN['N_BN'].interpolate(method='cubic')
    # data_N_BN['DATE'] = pd.to_datetime(data_N_BN['date'])
    data_N_BN = data_N_BN["N_BN"]

    # Markup data
    data_markup = pd.read_excel("./data/markup/nekarda_ramey_markups.xlsx")
    data_markup['DATE'] = pd.to_datetime(data_markup['qdate'], format='%Y-%m-%d')
    data_markup['markup'] = data_markup['mu_bus']
    data_markup = data_markup[['DATE', 'markup']].dropna()

    # De-trended Markup data
    data_markup_BN = pd.read_csv("./data/markup/BN_markup_inv.csv")
    data_markup_BN['markup_BN_inv'] = data_markup_BN["cycle"] 
    data_markup_BN['DATE'] = data_markup_BN["date"]
    data_markup_BN = data_markup_BN[["DATE","markup_BN_inv"]].dropna()
    data_markup_BN['DATE'] = pd.to_datetime(data_markup_BN['DATE'])

    # unemployment gap
    # NROU
    data_nairu = pd.read_csv("./data/unemp_gap/NROU.csv")
    data_nairu["DATE"] = pd.to_datetime(data_nairu["observation_date"])
    # unemployment
    data_unemp = pd.read_csv("./data/unemp_gap/UNRATENSA.csv")
    data_unemp["DATE"] = pd.to_datetime(data_unemp["observation_date"])
    data_unempgap = pd.merge(data_nairu, data_unemp, on="DATE", how="outer")
    data_unempgap['unemp_gap'] = data_unempgap['NROU'] - data_unempgap['UNRATENSA']
    data_unempgap = data_unempgap.dropna()
    data_unempgap = data_unempgap[["DATE", "unemp_gap"]]

    # output gap data
    data_output = pd.read_csv("./data/output_gap/BN_filter_GDPC1_quaterly.csv")
    data_output['output_BN'] = data_output['GDPC1_transformed_series']
    data_output['output_gap_BN'] = data_output["cycle"]
    data_output['output'] = np.log(data_output["GDPC1_original_series"] * 0.01)
    data_output['DATE'] = data_output["date"]
    data_output["output_trend_BN"] = data_output["output_BN"] - data_output["output_gap_BN"]
    data_output = data_output[["DATE", "output_BN", "output_gap_BN", "output_trend_BN", "output"]].dropna()
    data_output['DATE'] = pd.to_datetime(data_output['DATE'])

    # inflation expectation data
    data_Epi = pd.read_csv("./data/inflation/one_year_inflation_expectation.csv")
    data_Epi['DATE'] = pd.to_datetime(data_Epi['Model Output Date'])
    data_Epi.set_index("DATE", inplace=True)
    data_Epi = data_Epi[[" 1 year Expected Inflation"]].resample("QS").mean() * 100
    data_Epi['Epi'] = data_Epi[' 1 year Expected Inflation']
    data_Epi = data_Epi.reset_index()
    data_Epi = data_Epi[["DATE", "Epi"]]
    data_Epi['DATE'] = pd.to_datetime(data_Epi['DATE'])

    # Oil prices
    data_oil = pd.read_csv("./data/others/WTISPLC_CPIAUCSL.csv")
    data_oil["DATE"] = pd.to_datetime(data_oil["observation_date"])
    data_oil["log_oil"]= np.log(data_oil["WTISPLC_CPIAUCSL"])
    data_oil['oil'] = data_oil["log_oil"].diff(4)

    data_oil = data_oil[['oil', 'DATE']]
    data_oil = data_oil.reset_index()
    data_oil['DATE'] = pd.to_datetime(data_oil['DATE'])

    # Merge all dataframes
    data = pd.merge(data_cpi, data_Epi, on="DATE", how="outer")
    data = pd.merge(data, data_spf, on="DATE", how="outer")
    data = pd.merge(data, data_corecpi, on="DATE", how="outer")
    data = pd.merge(data, data_output, on="DATE", how="outer")
    data = pd.merge(data, data_HHI, on="DATE", how="outer")
    data = pd.merge(data, data_N_BN, on="DATE", how="outer")
    data = pd.merge(data, data_markup, on="DATE", how="outer")
    data = pd.merge(data, data_pi_pce_core, on="DATE", how="outer")
    data = pd.merge(data, data_pi_pce, on="DATE", how="outer")
    data = pd.merge(data, data_markup_BN, on="DATE", how="outer")
    data = pd.merge(data, data_unempgap, on="DATE", how="outer")
    data = pd.merge(data, data_oil, on="DATE", how="outer")
    data["pi_cpi_prev"] = data["pi_cpi"].shift(1)
    data["pi_cpi_core_prev"] = data["pi_cpi_core"].shift(1)
    data["pi_pce_prev"] = data["pi_pce"].shift(1)
    data["pi_pce_core_prev"] = data["pi_pce_core"].shift(1)
    data = data.dropna()
    return data