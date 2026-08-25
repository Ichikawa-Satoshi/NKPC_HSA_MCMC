"""Estimation helpers for the HSA PPI identification bundle.

Thin wrappers over the observed-HHI toolkit (tests/observed_hhi) specialised to the
theory-near cell: Capital IQ firm-weighted competition x PPI x inverse-markup x SPF
expectations, with the competition-level term (psi) EXCLUDED (theory-faithful).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nkpc_hsa.report_models.cases import _load_frame  # noqa: F401  (re-exported)
from tests.observed_hhi.functions import (
    ObservedHHISample,
    fit_observed_hhi_model,
    summarize_observed_fit,
    transform_inverse_hhi,
    fast_component,
    timed_fast_component,
)

HSA_PARAMS = ["kappa_0", "kappa_1", "theta_0", "theta_hsa", "gamma"]


def build_sample(frame: pd.DataFrame, config: dict, activity: str = None,
                 sample_start: str | None = None) -> ObservedHHISample:
    """Build the complete-case PPI x <activity> x CapIQ-firm sample from ``sample_start``.

    ``activity`` is a key in config['activities'] (defaults to primary_activity).
    """
    cell = config["cell"]
    activity = activity or config.get("primary_activity", "inverse_markup")
    act_col = config["activities"][activity]["column"]
    y_col, lag_col = cell["inflation"]
    cols = [y_col, lag_col, cell["expectation"], act_col, cell["competition"]]
    sub = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sample_start is not None:
        sub = sub[sub.index >= pd.Period(sample_start, freq="Q")]
    return ObservedHHISample(
        periods=sub.index,
        y=sub[y_col].to_numpy(float),
        pi_lag=sub[lag_col].to_numpy(float),
        expectation=sub[cell["expectation"]].to_numpy(float),
        activity=sub[act_col].to_numpy(float),
        q=transform_inverse_hhi(sub[cell["competition"]].to_numpy(float)),
        inflation="ppi",
        activity_name=activity,
        hhi_variant=cell["competition"],
    )


def gustavo_capiq_quarterly(frame: pd.DataFrame, competition_col: str,
                            max_weight: float = 3.0) -> tuple[pd.Series, dict]:
    r"""Temporal disaggregation of the annual Gustavo change into quarters, using
    Capital IQ quarterly weights where observed and an average profile where missing:

        \hat{\Delta h}^G_{tq} = w_{tq} * \Delta h^G_t,   \sum_q w_{tq} = 1,
        w_{tq} = \hat w^{CIQ}_{tq}   (year-specific Capital IQ share)  if CIQ observed,
                 \bar w_q            (average profile)                 if CIQ missing.

    Year-specific weights are w_{tq}=dCIQ_{tq}/sum_q dCIQ_{tq}; they fall back to the
    average \bar w_q when unstable (|w|>max_weight, e.g. a near-zero annual change).
    The average \bar w_q = <dCIQ_q, dCIQ_annual>/<dCIQ_annual, dCIQ_annual> (pooled,
    sums to 1). Cumulating gives a quarterly 10 log N whose Q4 matches Gustavo exactly,
    over the full 1974-2013 span (steep-Phillips era included).
    """
    from nkpc_hsa.report_models.cases import GUSTAVO_ANNUAL_COL
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    gA = num(GUSTAVO_ANNUAL_COL).dropna()
    Gann = pd.Series({ix.year: 10.0 * np.log(v) for ix, v in gA.items()})
    dc = (10.0 * np.log(num(competition_col)).dropna()).diff()
    cnt = dc.groupby(dc.index.year).count()
    cl = [y for y in Gann.index if (y - 1) in Gann.index and y in cnt.index and cnt[y] == 4]
    Q = np.array([[dc[pd.Period(f"{y}Q{q}", freq="Q")] for q in (1, 2, 3, 4)] for y in cl])
    ann = Q.sum(1)
    wbar = np.array([float(np.dot(Q[:, q], ann) / np.dot(ann, ann)) for q in range(4)])  # average profile
    years = sorted(Gann.index)
    fi = pd.period_range(f"{years[0]}Q1", f"{years[-1]}Q4", freq="Q")
    Gq = pd.Series(index=fi, dtype=float)
    n_year_specific = 0
    for y in years:
        prev = Gann.get(y - 1, Gann[y]); a = Gann[y] - prev
        w = wbar  # default: average profile (CIQ missing)
        if y in cl:  # CIQ observed for the full year -> year-specific weights
            dcy = np.array([dc[pd.Period(f"{y}Q{q}", freq="Q")] for q in (1, 2, 3, 4)])
            tot = dcy.sum()
            if abs(tot) > 1e-9:
                wy = dcy / tot
                if np.max(np.abs(wy)) <= max_weight:  # stability guard
                    w = wy; n_year_specific += 1
        cum = 0.0
        for q in (1, 2, 3, 4):
            cum += w[q - 1] * a
            Gq[pd.Period(f"{y}Q{q}", freq="Q")] = prev + cum
    return Gq.dropna(), {"wbar": wbar.tolist(), "s": wbar.tolist(),
                         "cl_years": (cl[0], cl[-1]), "n_year_specific": n_year_specific}


def gustavo_capiq_quarterly_v2(frame: pd.DataFrame, competition_col: str) -> tuple[pd.Series, dict]:
    """Indicator-based disaggregation of annual Gustavo (Nbar) with Capital IQ weights:

        dNhat^G_{tq} = w_{tq} * dN^G_t ,   sum_q w_{tq} = 1,
        w_{tq} = w^CIQ_{tq}  (year-specific Capital IQ share)   if CIQ observed that year,
               = wbar_q      (average quarterly share)          if CIQ missing,

    with dN^G_t the annual Gustavo change (Q4-to-Q4). Year-specific weights use that
    year's own Capital IQ quarterly changes; missing years fall back to the average
    profile. Guarded: a near-zero annual Capital IQ change (unstable shares) also
    uses wbar. Cumulating gives a quarterly series whose Q4 matches Gustavo exactly,
    over the full 1974-2013 span. All in 10 log N.
    """
    from nkpc_hsa.report_models.cases import GUSTAVO_ANNUAL_COL
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    gA = num(GUSTAVO_ANNUAL_COL).dropna()
    Gann = pd.Series({ix.year: 10.0 * np.log(v) for ix, v in gA.items()})
    dc = (10.0 * np.log(num(competition_col)).dropna()).diff()
    cnt = dc.groupby(dc.index.year).count()
    cl = [y for y in Gann.index if (y - 1) in Gann.index and y in cnt.index and cnt[y] == 4]
    # average (pooled, robust) shares wbar_q = <dCIQ_q, dCIQ_ann>/<dCIQ_ann, dCIQ_ann>
    Qm = np.array([[dc[pd.Period(f"{y}Q{q}", freq="Q")] for q in (1, 2, 3, 4)] for y in cl])
    ann = Qm.sum(1)
    wbar = np.array([float(np.dot(Qm[:, q], ann) / np.dot(ann, ann)) for q in range(4)])
    scale = float(np.median(np.abs(Qm)))  # guard scale for near-zero annual change
    years = sorted(Gann.index)
    fi = pd.period_range(f"{years[0]}Q1", f"{years[-1]}Q4", freq="Q")
    Gq = pd.Series(index=fi, dtype=float)
    src = {}
    for y in years:
        prev = Gann.get(y - 1, Gann[y]); dG = Gann[y] - prev
        if y in cl:
            dcy = np.array([dc[pd.Period(f"{y}Q{q}", freq="Q")] for q in (1, 2, 3, 4)])
            a = dcy.sum()
            wcand = dcy / a if abs(a) > 0.5 * scale else wbar
            # A tiny annual change or a large within-year swing relative to it makes
            # the shares explode (e.g. Capital IQ's erratic 1998); fall back to wbar.
            if abs(a) > 0.5 * scale and np.max(np.abs(wcand)) <= 3.0:
                w = wcand; src[y] = "ciq"
            else:
                w = wbar; src[y] = "avg(guard)"
        else:
            w = wbar; src[y] = "avg(missing)"
        cum = 0.0
        for q in (1, 2, 3, 4):
            cum += w[q - 1] * dG
            Gq[pd.Period(f"{y}Q{q}", freq="Q")] = prev + cum
    return Gq.dropna(), {"wbar": wbar.tolist(), "years_ciq": (cl[0], cl[-1]) if cl else None, "src": src}


def build_gustavo_capiq_sample(frame: pd.DataFrame, config: dict, activity: str = None,
                               sample_start: str | None = None, inflation: str = None,
                               construction: str = "v2") -> ObservedHHISample:
    """ObservedHHISample with the Gustavo x Capital IQ temporal-disaggregation series.

    construction='v2' uses year-specific Capital IQ weights (fallback to average);
    'avg' uses the average quarterly profile for all years. ``inflation`` overrides
    the config inflation cell (e.g. 'ppi' or 'core_cpi')."""
    # Epi = SPF GDP-deflator forecast for every inflation measure (per spec).
    INFL = {"ppi": ("pi_ppi", "pi_ppi_prev", "Epi_spf_gdp"),
            "core_cpi": ("pi_cpi_core", "pi_cpi_core_prev", "Epi_spf_gdp")}
    cell = config["cell"]
    activity = activity or config.get("primary_activity", "inverse_markup")
    act_col = config["activities"][activity]["column"]
    y_col, lag_col, e_col = INFL[inflation or "ppi"]
    num = lambda c: pd.to_numeric(frame[c], errors="coerce")
    builder = gustavo_capiq_quarterly_v2 if construction == "v2" else gustavo_capiq_quarterly
    Gq, _ = builder(frame, cell["competition"])
    d = pd.concat({"y": num(y_col), "lag": num(lag_col), "e": num(e_col),
                   "x": num(act_col), "Gq": Gq}, axis=1).dropna()
    if sample_start is not None:
        d = d[d.index >= pd.Period(sample_start, freq="Q")]
    return ObservedHHISample(
        periods=d.index, y=d["y"].to_numpy(float), pi_lag=d["lag"].to_numpy(float),
        expectation=d["e"].to_numpy(float), activity=d["x"].to_numpy(float),
        q=transform_inverse_hhi(np.exp(d["Gq"].to_numpy(float) / 10.0)),  # centered 10 log Gq
        inflation=(inflation or "ppi"), activity_name=activity, hhi_variant="gustavo_capiq")


def fit_cell(sample: ObservedHHISample, variant: str, design: dict, sampling: dict) -> pd.DataFrame:
    """Fit one model variant; return a tidy coefficient table with P(>0)."""
    fit = fit_observed_hhi_model(
        sample, cell=1,
        fast_definition=design["fast_definition"],
        timing=design["timing"],
        model_variant=variant,
        error_model=design["error_model"],
        include_level=bool(design["include_level"]),
        zeta_reference=float(design["zeta_reference"]),
        b_x=float(design["b_x"]),
        iterations=int(sampling["iterations"]),
        warmup=int(sampling["warmup"]),
        thin=int(sampling["thin"]),
        chains=int(sampling["chains"]),
        seed=int(sampling["seed"]),
    )
    s = summarize_observed_fit(fit)
    s = s.copy()
    s["variant"] = variant
    s["n"] = len(sample.y)
    s["condition_number"] = fit.design_condition_number
    s["P_positive"] = s.apply(
        lambda r: r["sign_probability"] if r["mean"] > 0 else 1.0 - r["sign_probability"], axis=1
    )
    return s


def effective_slope(sample: ObservedHHISample, design: dict, kappa_0: float, kappa_1: float):
    """kappa_t = kappa_0 + kappa_1 * z over the sample; return z percentiles and slopes."""
    raw_fast = fast_component(sample.q, design["fast_definition"])
    z = sample.q.copy() - np.nanmean(sample.q)
    z = z[np.isfinite(z)]
    pts = {p: float(np.percentile(z, p)) for p in (5, 50, 95)}
    slopes = {k: kappa_0 + kappa_1 * v for k, v in pts.items()}
    z_star = -kappa_0 / kappa_1 if kappa_1 != 0 else np.nan
    frac_pos = float(np.mean(z > z_star)) if np.isfinite(z_star) else np.nan
    return {"z_pct": pts, "slope": slopes, "z_star": z_star, "frac_positive": frac_pos,
            "z_min": float(z.min()), "z_max": float(z.max())}
