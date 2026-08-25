"""Economically disciplined design screen for the deep HSA audit.

This is not the final estimator.  It uses transparent state proxies and exact
Gaussian ARMA likelihoods to reject poorly conditioned architectures before the
expensive joint-state runs.  Deterministic filters are never eligible to become
the confirmatory state posterior.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from nkpc_hsa.phillips.data import load_design_data  # noqa:E402
from tests.hsa_nested_validation.functions import _load_frame, load_experiment  # noqa:E402

BUNDLE = Path(__file__).resolve().parent
NESTED = ROOT / "tests" / "hsa_nested_validation"

PRICE = {
    "ppi": {
        "yoy": ("pi_ppi", "pi_ppi_prev", "Epi_spf_gdp"),
        "qoq": ("qoq_pi_ppi", "qoq_pi_ppi_lag1", "qoq_expectation"),
    },
    "core_cpi": {
        "yoy": ("pi_cpi_core", "pi_cpi_core_prev", "Epi_spf_gdp"),
        "qoq": ("qoq_pi_core_cpi", "qoq_pi_core_cpi_lag1", "qoq_expectation"),
    },
}
ACTIVITY = {
    "yoy": {
        "negative_unemployment_gap": "unemp_gap",
        "inverse_markup": "markup_BN_inv",
    },
    "qoq": {
        "negative_unemployment_gap": "qoq_x_negative_unemployment_gap",
        "inverse_markup": "qoq_x_inverse_markup",
    },
}
ERRORS = {"yoy": ("ma3",), "qoq": ("iid", "ar1")}
ORDERS = {"ma3": (0, 0, 3), "iid": (0, 0, 0), "ar1": (1, 0, 0)}


def _analysis_frame() -> pd.DataFrame:
    """Join frequency-consistent quarterly variables to the legacy YoY frame.

    In particular, the QoQ screen must use the genuine SPF one-quarter-ahead
    forecast.  The backward-compatible ``Epi_spf_gdp`` field is a one-year-
    ahead forecast and is reserved for the YoY screen.
    """
    frame = _load_frame().copy()
    design = load_design_data(
        include_qcew=False, sample_start="1982Q1", sample_end="2013Q4"
    ).quarterly
    quarterly = design[[
        "pi_ppi", "pi_ppi_lag1", "pi_core_cpi", "pi_core_cpi_lag1",
        "expectation", "x_negative_unemployment_gap", "x_inverse_markup",
    ]].rename(columns={
        "pi_ppi": "qoq_pi_ppi",
        "pi_ppi_lag1": "qoq_pi_ppi_lag1",
        "pi_core_cpi": "qoq_pi_core_cpi",
        "pi_core_cpi_lag1": "qoq_pi_core_cpi_lag1",
        "expectation": "qoq_expectation",
        "x_negative_unemployment_gap": "qoq_x_negative_unemployment_gap",
        "x_inverse_markup": "qoq_x_inverse_markup",
    })
    return frame.join(quarterly, how="left")


def _competition_paths() -> dict[str, pd.DataFrame]:
    cfg = load_yaml(NESTED / "config.yaml")
    experiment = load_experiment(cfg)
    periods = experiment.allocation.periods
    q = pd.Series(experiment.allocation_mean_raw - experiment.q0, index=periods, name="q")

    # Current exact-state AR(2) benchmark, using the CES state so inflation does
    # not define the proxy used in this screen.
    saved = NESTED / "results" / "full" / "draws" / "joint_state_split" \
        / "ppi_negative_unemployment_gap" / "ces.npz"
    with np.load(saved, allow_pickle=False) as z:
        p = pd.PeriodIndex(z["periods"].astype(str), freq="Q")
        ar2_q = pd.Series(z["n_total"].mean(axis=(0, 1)), index=p)
        ar2_bar = pd.Series(z["nbar"].mean(axis=(0, 1)), index=p)
        ar2_hat = pd.Series(z["nhat"].mean(axis=(0, 1)), index=p)

    # Measurement-only average-allocation path.  It is a diagnostic proxy for
    # S1: the final S1 model estimates deviations and annual innovations.
    avg = pd.Series(index=periods, dtype=float)
    weights = np.asarray(experiment.allocation.average_weights, float)
    for year in experiment.allocation.annual.index:
        year = int(year)
        previous = float(experiment.allocation.annual.get(year - 1, experiment.allocation.annual[year]))
        change = float(experiment.allocation.annual[year] - previous)
        cumulative = 0.0
        for quarter in range(1, 5):
            cumulative += weights[quarter - 1] * change
            avg[pd.Period(f"{year}Q{quarter}", freq="Q")] = previous + cumulative
    mask = (avg.index >= pd.Period("1974Q4", freq="Q")) & (avg.index <= pd.Period("2013Q4", freq="Q"))
    avg = avg - float(avg[mask].mean())

    # Filtered paths are screening diagnostics only.
    q_sample = q.loc["1974Q4":"2013Q4"]
    hp_paths = {}
    for smooth in (400.0, 1600.0, 10000.0):
        cycle, trend = hpfilter(q_sample.astype(float), lamb=smooth)
        hp_paths[f"hp_{int(smooth)}_diagnostic"] = pd.DataFrame({"q": q_sample, "bar": trend, "hat": cycle})
    ewma_bar = q_sample.ewm(halflife=8.0, adjust=False).mean()

    out = {
        "s0_quarterly_local_level_ar2": pd.DataFrame({"q": ar2_q, "bar": ar2_bar, "hat": ar2_hat}),
        "s1_annual_allocation_proxy": pd.DataFrame({"q": q_sample, "bar": avg.loc[q_sample.index],
                                                     "hat": q_sample - avg.loc[q_sample.index]}),
        "ewma_hl8_diagnostic": pd.DataFrame({"q": q_sample, "bar": ewma_bar, "hat": q_sample - ewma_bar}),
        **hp_paths,
    }
    for value in out.values():
        error = float(np.max(np.abs(value["q"] - value["bar"] - value["hat"])))
        if error > 1e-10:
            raise AssertionError(f"state proxy violates exact identity: {error}")
    return out


def _standard_condition(x: np.ndarray) -> float:
    z = np.asarray(x[:, 1:], float)
    scale = np.std(z, axis=0, ddof=1)
    if np.any(scale < 1e-10):
        return float("inf")
    z = (z - np.mean(z, axis=0)) / scale
    return float(np.linalg.cond(z))


def _orthogonal_share(base: np.ndarray, target: np.ndarray) -> float:
    residual = target - base @ np.linalg.lstsq(base, target, rcond=None)[0]
    denominator = float(target @ target)
    return float(residual @ residual / denominator) if denominator > 1e-12 else 0.0


def _fit(y: np.ndarray, x: np.ndarray, error: str):
    model = SARIMAX(y, exog=x, order=ORDERS[error], trend="n",
                    enforce_stationarity=True, enforce_invertibility=True,
                    concentrate_scale=False)
    result = model.fit(disp=False, maxiter=1000, method="lbfgs")
    k = x.shape[1]
    beta = np.asarray(result.params[:k], float)
    covariance = np.asarray(result.cov_params()[:k, :k], float)
    return result, beta, covariance


def _cell_frame(frame, paths, price, activity, frequency, timing, discovery):
    pi, lag, epi = PRICE[price][frequency]
    data = pd.concat({
        "pi": pd.to_numeric(frame[pi], errors="coerce"),
        "pi_lag": pd.to_numeric(frame[lag], errors="coerce"),
        "epi": pd.to_numeric(frame[epi], errors="coerce"),
        "x": pd.to_numeric(frame[ACTIVITY[frequency][activity]], errors="coerce"),
        "q": paths["q"], "bar": paths["bar"], "hat": paths["hat"],
    }, axis=1)
    data["hat_use"] = data["hat"].shift(1) if timing == "lag1" else data["hat"]
    data = data.loc[discovery[0]:discovery[1]].dropna()
    return data


def main() -> None:
    cfg = load_yaml(BUNDLE / "config.yaml")
    frame = _analysis_frame()
    paths = _competition_paths()
    samples = {
        "discovery": tuple(cfg["sample"]["discovery"]),
        "validation": tuple(cfg["sample"]["validation"]),
        "full": tuple(cfg["sample"]["full"]),
    }
    rows = []
    for sample_split, sample_bounds in samples.items():
      for state_name, state in paths.items():
        eligible = not state_name.endswith("_diagnostic") and "proxy" not in state_name
        for price in cfg["cells"]["prices"]:
          for activity in cfg["cells"]["activities"]:
            for frequency, errors in ERRORS.items():
              for timing in cfg["nkpc"]["timings"]:
                data = _cell_frame(frame, state, price, activity, frequency, timing, sample_bounds)
                if len(data) < 24:
                    continue
                y = data["pi"].to_numpy(float)
                base = np.column_stack((np.ones(len(data)), data["pi_lag"], data["epi"], data["x"]))
                for error in errors:
                    try:
                        ces, _, _ = _fit(y, base, error)
                    except Exception as exc:  # noqa: BLE001
                        rows.append({"sample_split": sample_split,
                                     "state": state_name, "eligible_joint_state": eligible,
                                     "price": price, "activity": activity, "frequency": frequency,
                                     "error": error, "timing": timing, "lambda": np.nan,
                                     "status": f"CES_FAILED: {type(exc).__name__}"})
                        continue
                    for lam in map(float, cfg["nkpc"]["lambda_grid"]):
                        target = lam * data["bar"].to_numpy(float) * data["x"].to_numpy(float) \
                            - data["hat_use"].to_numpy(float)
                        x = np.column_stack((base, target))
                        row = {"sample_split": sample_split,
                               "state": state_name, "eligible_joint_state": eligible,
                               "price": price, "activity": activity, "frequency": frequency,
                               "error": error, "timing": timing, "lambda": lam, "n": len(data),
                               "condition_number": _standard_condition(x),
                               "direct_orthogonal_share": _orthogonal_share(base, target),
                               "status": "OK"}
                        try:
                            hsa, beta, covariance = _fit(y, x, error)
                            sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
                            names = ("intercept", "alpha_b", "alpha_f", "kappa_0", "theta")
                            for j, name in enumerate(names):
                                row[f"{name}_mean"] = beta[j]
                                row[f"{name}_sd"] = sd[j]
                                row[f"{name}_q2.5"] = beta[j] - 1.96 * sd[j]
                                row[f"{name}_q97.5"] = beta[j] + 1.96 * sd[j]
                                row[f"{name}_p_positive"] = float(norm.cdf(beta[j] / max(sd[j], 1e-12)))
                            kappa_probability = []
                            for bar_t in data["bar"].to_numpy(float):
                                loading = np.array([0.0, 0.0, 0.0, 1.0, lam * bar_t])
                                mean = float(loading @ beta)
                                variance = float(loading @ covariance @ loading)
                                kappa_probability.append(float(norm.cdf(mean / np.sqrt(max(variance, 1e-12)))))
                            row["min_kappa_path_p_positive"] = min(kappa_probability)
                            row["bic_hsa_minus_ces"] = float(hsa.bic - ces.bic)
                            row["aic_hsa_minus_ces"] = float(hsa.aic - ces.aic)
                            row["converged_optimizer"] = bool(hsa.mle_retvals.get("converged", False))
                        except Exception as exc:  # noqa: BLE001
                            row["status"] = f"HSA_FAILED: {type(exc).__name__}"
                        rows.append(row)
    result = pd.DataFrame(rows)
    out = BUNDLE / "results" / "screen"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "candidate_screen.csv", index=False)
    gates = cfg["gates"]
    ok = result.status.eq("OK")
    result["screen_identified"] = (
        ok
        & result.alpha_b_p_positive.ge(gates["sign_probability"])
        & result.alpha_f_p_positive.ge(gates["sign_probability"])
        & result.theta_p_positive.ge(gates["sign_probability"])
        & result.min_kappa_path_p_positive.ge(gates["kappa_path_positive_probability"])
        & result.condition_number.le(gates["max_condition_number"])
        & result.direct_orthogonal_share.ge(gates["min_direct_orthogonal_share"])
    )
    result.to_csv(out / "candidate_screen.csv", index=False)
    ranked = result[ok].sort_values(
        ["sample_split", "screen_identified", "bic_hsa_minus_ces", "theta_p_positive"],
        ascending=[True, False, True, False],
    )
    ranked.head(80).to_csv(out / "ranked_candidates.csv", index=False)
    manifest = {
        "revision": cfg["revision"], "rows": len(result),
        "successful_likelihoods": int(ok.sum()),
        "screen_identified": int(result.screen_identified.sum()),
        "eligible_joint_state_passes": int((result.screen_identified & result.eligible_joint_state).sum()),
        "note": "BIC/AIC and deterministic proxies are screening diagnostics, not formal model evidence.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    show = ["sample_split", "state", "price", "activity", "frequency", "error", "timing", "lambda",
            "theta_mean", "theta_p_positive", "min_kappa_path_p_positive",
            "direct_orthogonal_share", "condition_number", "bic_hsa_minus_ces", "screen_identified"]
    print(json.dumps(manifest, indent=2))
    print(ranked[show].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
