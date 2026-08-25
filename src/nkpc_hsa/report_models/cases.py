"""Load the four data cases x twelve empirical specifications from model_ready.csv.

Inflation-expectation pairs (report Panel A) and forcing variables (Panel B):

    inflation in {ppi, cpi, core_cpi}         forcing in {inverse_markup,
                                                          negative_unemployment_gap,
                                                          bn_output_gap, hp_output_gap}

Competition input by case:

    Case 1  Capital IQ quarterly            N_capitaliq_{firmw,revw}
    Case 2  Gustavo, PCHIP-interpolated     N_Gustavo (quarterly)
    Case 3  Gustavo, mixed frequency        N_Gustavo_annual_q4 (Q4 only)
    Case 4  Case 3 + establishment growth   + establishment_stock growth in transition
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nkpc_hsa.paths import data_root
from nkpc_hsa.report_models.engine import CaseData

INFLATION = {
    "ppi": ("pi_ppi", "Epi_spf_gdp"),
    "cpi": ("pi_cpi", "Epi_spf_cpi"),
    "core_cpi": ("pi_cpi_core", "Epi_spf_cpi"),
}
FORCING = {
    "inverse_markup": "markup_BN_inv",
    "inverse_markup_raw": "markup_inv",       # unfiltered inverse markup (labor-share proxy)
    "negative_unemployment_gap": "unemp_gap",
    "bn_output_gap": "output_gap_BN",
    "hp_output_gap": "output_gap_HP",
}
N_VARIANTS = {
    1: {"firm_weighted": "N_capitaliq_firmw", "revenue_weighted": "N_capitaliq_revw",
        "sic4_firm": "N_capitaliq_sic4_firmw", "sic4_rev": "N_capitaliq_sic4_revw",
        "mfg_firm": "N_capitaliq_mfg_firmw"},
    2: {"gustavo": "N_Gustavo"},
    3: {"gustavo": "N_Gustavo_annual_q4"},
    4: {"gustavo": "N_Gustavo_annual_q4"},
}
# Capital IQ coverage/EDGAR transition ends ~1996; the 4-digit variants start
# after it so the effective firm count is not a coverage artifact.
VARIANT_SAMPLE_START = {"sic4_firm": "1997Q1", "sic4_rev": "1997Q1", "mfg_firm": "1997Q1"}
GUSTAVO_LEVEL_COL = "N_Gustavo"          # quarterly level, used to fix the log center
GUSTAVO_ANNUAL_COL = "N_Gustavo_annual_q4"


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    scale = float(np.subtract(*np.quantile(finite, [0.75, 0.25])) / 1.349)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("IQR scale gate failed (non-positive robust scale).")
    return scale


def _load_frame(processed_path: str | None = None) -> pd.DataFrame:
    processed = (Path(processed_path).parent if processed_path
                 else data_root() / "processed")
    path = processed_path or (processed / "model_ready.csv")
    frame = pd.read_csv(path, parse_dates=["DATE"])
    frame.index = pd.PeriodIndex(frame["DATE"], freq="Q")
    frame = frame.drop(columns=["DATE"]).sort_index()
    # Join the 4-digit-SIC Capital IQ series if it has been built.
    for fname, cols in (
        ("capital_iq_N_sic4_quarterly.csv", ("N_capitaliq_sic4_firmw", "N_capitaliq_sic4_revw")),
        ("capital_iq_N_mfg_quarterly.csv", ("N_capitaliq_mfg_firmw",)),
    ):
        path_extra = Path(processed) / fname
        if path_extra.exists():
            s = pd.read_csv(path_extra)
            s.index = pd.PeriodIndex(s["tq"].astype(str).str.upper(), freq="Q")
            for col in cols:
                frame[col] = s[col].reindex(frame.index)
    return frame


def _log_center(level: pd.Series) -> float:
    v = pd.to_numeric(level, errors="coerce").dropna()
    if (v <= 0).any():
        raise ValueError("Competition level must be positive before the log transform.")
    return float((100.0 * np.log(v)).mean())


def _transform(level: pd.Series, center: float) -> pd.Series:
    v = pd.to_numeric(level, errors="coerce")
    return (100.0 * np.log(v) - center) / 10.0


def _seasonal_factors(logv: pd.Series) -> pd.Series:
    """Additive, mean-zero quarterly seasonal factors of a 100*log level series.

    Estimated from the trend-removed series (centered 4-quarter moving average) so
    the factors capture only the within-year seasonal pattern, not the trend.
    """
    s = pd.to_numeric(logv, errors="coerce").dropna()
    trend = s.rolling(4, center=True, min_periods=4).mean()
    dev = (s - trend).dropna()
    fac = dev.groupby(dev.index.quarter).mean()
    return fac - fac.mean()


def _log_level_deseasonalized(level: pd.Series, deseasonalize: bool) -> pd.Series:
    logv = 100.0 * np.log(pd.to_numeric(level, errors="coerce"))
    if deseasonalize:
        fac = _seasonal_factors(logv)
        logv = logv - logv.index.quarter.map(fac).astype(float)
    return logv


def available_specs(case: int) -> list[dict]:
    return [
        {"case": case, "inflation": infl, "forcing": forc, "variant": var}
        for var in N_VARIANTS[case]
        for infl in INFLATION
        for forc in FORCING
    ]


def load_case(
    case: int,
    inflation: str,
    forcing: str,
    variant: str,
    *,
    frame: pd.DataFrame | None = None,
    processed_path: str | None = None,
    deseasonalize: bool = False,
    sample_start: str | None = None,
    control: str | None = None,
) -> CaseData:
    if case not in (1, 2, 3, 4):
        raise ValueError("case must be 1..4")
    frame = _load_frame(processed_path) if frame is None else frame
    pi_col, epi_col = INFLATION[inflation]
    x_col = FORCING[forcing]
    n_col = N_VARIANTS[case][variant]

    pi = pd.to_numeric(frame[pi_col], errors="coerce")
    epi = pd.to_numeric(frame[epi_col], errors="coerce")
    x = pd.to_numeric(frame[x_col], errors="coerce")

    # Competition observation (centered ten-log-points).
    if case == 1:
        logv = _log_level_deseasonalized(frame[n_col], deseasonalize)
        center = float(logv.dropna().mean())
        n_series = (logv - center) / 10.0
        comp_index = n_series.dropna().index
    else:
        center = _log_center(frame[GUSTAVO_LEVEL_COL])
        if case == 2:
            n_series = _transform(frame[GUSTAVO_LEVEL_COL], center)
            comp_index = n_series.dropna().index
        else:
            n_series = _transform(frame[GUSTAVO_ANNUAL_COL], center)
            comp_index = frame[GUSTAVO_LEVEL_COL].dropna().index  # quarterly span

    # Establishment growth (Case 4 transition input).
    gE_full = None
    if case == 4:
        stock = pd.to_numeric(frame["establishment_stock"], errors="coerce")
        gE_full = np.log(stock).diff()
        gE_full = gE_full - gE_full.mean()

    # Build the contiguous estimation window.
    always = pi.notna() & epi.notna() & x.notna()
    if case == 4:
        always = always & gE_full.notna()
    comp_lo, comp_hi = comp_index.min(), comp_index.max()
    always = always & (frame.index >= comp_lo) & (frame.index <= comp_hi)
    start_period = sample_start or VARIANT_SAMPLE_START.get(variant)
    if start_period is not None:
        always = always & (frame.index >= pd.Period(start_period, freq="Q"))
    idx = frame.index[always]
    if len(idx) < 24:
        raise ValueError(f"Sample too short for case {case} {inflation}/{forcing}/{variant}: {len(idx)}")
    # Longest contiguous quarterly run.
    start, end = _longest_contiguous(idx)
    window = pd.period_range(start, end, freq="Q")

    pi_w = pi.reindex(window).to_numpy(float)
    epi_w = epi.reindex(window).to_numpy(float)
    x_w = x.reindex(window).to_numpy(float)
    n_w = n_series.reindex(window).to_numpy(float)
    # Lagged inflation pi_{t-1} for the hybrid NKPC; first period backfilled with
    # the pre-window quarter when available, else the in-window mean.
    pi_lag_w = pi.shift(1).reindex(window).to_numpy(float)
    if not np.isfinite(pi_lag_w[0]):
        pi_lag_w[0] = float(np.nanmean(pi_w))
    if not (np.isfinite(pi_w).all() and np.isfinite(epi_w).all() and np.isfinite(x_w).all()):
        raise ValueError("Non-contiguous core series inside the estimation window.")

    gE_w = None
    s_E = None
    if case == 4:
        gE_w = gE_full.reindex(window).to_numpy(float)
        gE_w = np.nan_to_num(gE_w, nan=0.0)
        s_E = _robust_scale(gE_w)

    # s_N per report: native observed competition series (annual for Gustavo cases).
    if case in (3, 4):
        s_N = _robust_scale(_transform(frame[GUSTAVO_ANNUAL_COL], center).dropna().to_numpy())
    else:
        s_N = _robust_scale(n_w[np.isfinite(n_w)])

    label = f"case{case}__{inflation}__{forcing}__{variant}"
    return CaseData(
        case=case, label=label, periods=window,
        pi=pi_w, epi=epi_w, x=x_w, n_obs=n_w,
        exact_anchor=(case == 4), gE=gE_w,
        s_x=_robust_scale(x_w), s_N=s_N, s_pi=_robust_scale(pi_w), s_E=s_E,
        pi_lag=pi_lag_w,
        control=(_control_series(frame, control, window) if control else None),
        control_name=control,
    )


def _control_series(frame: pd.DataFrame, name: str, window) -> np.ndarray:
    c = pd.to_numeric(frame[name], errors="coerce").reindex(window).to_numpy(float)
    if not np.isfinite(c).all():
        raise ValueError(f"control '{name}' has missing values inside the estimation window")
    return c - np.nanmean(c)


def _longest_contiguous(idx: pd.PeriodIndex) -> tuple[pd.Period, pd.Period]:
    ordinals = np.array([p.ordinal for p in idx])
    best_len, best = 0, (idx[0], idx[0])
    run_start = 0
    for i in range(1, len(ordinals) + 1):
        if i == len(ordinals) or ordinals[i] != ordinals[i - 1] + 1:
            run_len = i - run_start
            if run_len > best_len:
                best_len = run_len
                best = (idx[run_start], idx[i - 1])
            run_start = i
    return best
