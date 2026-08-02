from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _bootstrap import ROOT
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION

DATA = ROOT / "data" / "processed" / "model_ready.csv"
TABLE_DIR = ROOT / "results" / "tables" / "report_additions"
FIGURE_DIR = ROOT / "results" / "figures" / "report"


def _current_posterior() -> Path:
    candidates: list[tuple[str, Path]] = []
    for metadata_path in (ROOT / "results" / "runs").glob("*/metadata.json"):
        posterior = metadata_path.parent / "posterior.nc"
        if not posterior.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if str(metadata.get("estimation_revision", "")) != ESTIMATION_REVISION:
            continue
        if str(metadata.get("model", "")) != "hsa_steady":
            continue
        if str(metadata.get("data_spec", "")) != "unemployment_gap_core":
            continue
        if str(metadata.get("prior_spec", "baseline") or "baseline") != "baseline":
            continue
        if str(metadata.get("period", "full") or "full") != "full":
            continue
        if str(metadata.get("constraint_spec", "unrestricted") or "unrestricted") != "unrestricted":
            continue
        if int(metadata.get("n_iter", 0) or 0) < 12000:
            continue
        candidates.append((str(metadata.get("run_id", metadata_path.parent.name)), posterior))
    if not candidates:
        raise FileNotFoundError("No current-revision HSA steady / core CPI / unemployment baseline posterior found.")
    return max(candidates)[1]


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    return beta, np.sqrt(np.diag(cov)), resid


def _residualize(z: np.ndarray, controls: np.ndarray) -> np.ndarray:
    return z - controls @ np.linalg.lstsq(controls, z, rcond=None)[0]


def _summary(draws: np.ndarray) -> tuple[float, float, float]:
    return (
        float(draws.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _sample_data() -> pd.DataFrame:
    data = pd.read_csv(DATA, parse_dates=["DATE"])
    data["period"] = data["DATE"].dt.to_period("Q")
    for lag in range(5):
        data[f"ppi_lag_{lag}"] = data["pi_ppi"].shift(lag)
    sample = data.loc[
        data["period"].between(pd.Period("1982Q1"), pd.Period("2012Q4"))
    ].copy()
    required = [
        "pi_cpi_core",
        "pi_cpi_core_prev",
        "Epi",
        "unemp_gap",
        "N_Gustavo",
        "N_Gustavo_BN_trend",
    ]
    sample = sample.dropna(subset=required).reset_index(drop=True)
    if len(sample) != 124:
        raise ValueError(f"Expected 124 complete quarters, found {len(sample)}")
    return sample


def _posterior_arrays() -> tuple[np.ndarray, np.ndarray]:
    posterior = _current_posterior()
    idata = az.from_netcdf(posterior)
    nbar = np.asarray(idata.posterior["Nbar"]).reshape(-1, 124)
    kappa_t = np.asarray(idata.posterior["kappa_t"]).reshape(-1, 124)
    (TABLE_DIR / "source_run.txt").write_text(str(posterior.parent) + "\n", encoding="utf-8")
    return nbar, kappa_t


def _write_tex_table(df: pd.DataFrame, filename: str, columns: list[str]) -> None:
    df.reindex(columns=columns).to_latex(
        TABLE_DIR / filename,
        index=False,
        escape=False,
        column_format="l" * len(columns),
    )


def write_report_tex_inputs() -> None:
    magnitude = pd.read_csv(TABLE_DIR / "economic_magnitude.csv")
    magnitude["change"] = magnitude["unemployment_gap_change"].map(
        {
            "1 percentage point": "1ポイント",
            "2 percentage points": "2ポイント",
            "4 percentage points": "4ポイント",
            "one sample standard deviation": "標本内1標準偏差",
        }
    )
    magnitude["posterior mean"] = magnitude["inflation_response_difference_mean_pp"].map(lambda x: f"{x:.3f}")
    magnitude["95\\% interval"] = magnitude.apply(
        lambda r: f"[{r['lower_95']:.3f}, {r['upper_95']:.3f}]", axis=1
    )
    _write_tex_table(magnitude, "economic_magnitude.tex", ["change", "posterior mean", "95\\% interval"])

    trend = pd.read_csv(TABLE_DIR / "competition_trend_sensitivity.csv")
    trend["competition trend"] = trend["competition_trend"].map(
        {"posterior Nbar": r"事後潜在トレンド $\bar N$", "observed N": "観測企業数", "source BN trend": "元データのBNトレンド"}
    )
    trend["trend change"] = trend["trend_change"].map(lambda x: f"{x:+.3f}")
    trend["delta (s.e.)"] = trend.apply(lambda r: f"{r['delta']:.3f} ({r['delta_se']:.3f})", axis=1)
    trend["kappa start"] = trend["kappa_start"].map(lambda x: f"{x:+.3f}")
    trend["kappa end"] = trend["kappa_end"].map(lambda x: f"{x:+.3f}")
    _write_tex_table(
        trend,
        "competition_trend_sensitivity.tex",
        ["competition trend", "trend change", "delta (s.e.)", "kappa start", "kappa end"],
    )

    mechanism = pd.read_csv(TABLE_DIR / "mechanism_endpoints.csv")
    mechanism["series label"] = mechanism["series"].map(
        {"listed firm count": "上場企業数", "aggregate markup": "集計マークアップ", "inverse markup": "逆マークアップ"}
    )
    mechanism["1982"] = mechanism["1982_average"].map(lambda x: f"{x:.3f}")
    mechanism["2012"] = mechanism["2012_average"].map(lambda x: f"{x:.3f}")
    mechanism["change"] = mechanism["percent_change"].map(lambda x: f"{100*x:+.1f}\\%")
    _write_tex_table(mechanism, "mechanism_endpoints.tex", ["series label", "1982", "2012", "change"])

    passthrough = pd.read_csv(TABLE_DIR / "ppi_core_pass_through.csv")
    passthrough["window"] = passthrough["window_years"].map(lambda x: f"{int(x)}年")
    passthrough["PPI specification"] = passthrough["specification"].map(
        {"current PPI": "当期", "current plus four lags": "当期+4ラグの和"}
    )
    passthrough["first"] = passthrough["first_estimate"].map(lambda x: f"{x:+.3f}")
    passthrough["2012Q4"] = passthrough["last_estimate"].map(lambda x: f"{x:+.3f}")
    passthrough["corr"] = passthrough["corr_with_posterior_Nbar"].map(lambda x: f"{x:+.3f}")
    _write_tex_table(passthrough, "ppi_core_pass_through.tex", ["window", "PPI specification", "first", "2012Q4", "corr"])


def write_economic_magnitude(sample: pd.DataFrame, kappa_t: np.ndarray) -> None:
    slope_decline = kappa_t[:, 0] - kappa_t[:, -1]
    rows = []
    for gap_size, label in [
        (1.0, "1 percentage point"),
        (2.0, "2 percentage points"),
        (4.0, "4 percentage points"),
        (float(sample["unemp_gap"].std(ddof=1)), "one sample standard deviation"),
    ]:
        mean, lower, upper = _summary(gap_size * slope_decline)
        rows.append(
            {
                "unemployment_gap_change": label,
                "gap_size_pp": gap_size,
                "inflation_response_difference_mean_pp": mean,
                "lower_95": lower,
                "upper_95": upper,
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "economic_magnitude.csv", index=False)

    x = sample["unemp_gap"].to_numpy()
    contribution_difference = (kappa_t - kappa_t[:, [0]]) * x[None, :]
    rms = np.sqrt(np.mean(contribution_difference**2, axis=1))
    maximum = np.max(np.abs(contribution_difference), axis=1)
    counterfactual = pd.DataFrame(
        [
            ("RMS", *_summary(rms)),
            ("maximum absolute", *_summary(maximum)),
        ],
        columns=["measure", "mean_pp", "lower_95", "upper_95"],
    )
    counterfactual.to_csv(TABLE_DIR / "fixed_slope_counterfactual.csv", index=False)


def write_trend_identification(
    sample: pd.DataFrame, nbar: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbar_mean = nbar.mean(axis=0)
    observed_log100 = 100.0 * np.log(sample["N_Gustavo"].to_numpy())
    center = observed_log100.mean()
    observed_n = (observed_log100 - center) / 10.0
    source_bn = (
        100.0 * np.log(sample["N_Gustavo_BN_trend"].to_numpy()) - center
    ) / 10.0

    y = (sample["pi_cpi_core"] - sample["Epi"]).to_numpy()
    inertia = (sample["pi_cpi_core_prev"] - sample["Epi"]).to_numpy()
    activity = sample["unemp_gap"].to_numpy()
    time = np.linspace(-1.0, 1.0, len(sample))
    controls = np.column_stack([np.ones(len(sample)), inertia, activity])
    interaction_n = activity * nbar_mean
    interaction_time = activity * time
    y_resid = _residualize(y, controls)
    n_resid = _residualize(interaction_n, controls)
    time_resid = _residualize(interaction_time, controls)

    groups = [
        ("1982--1987", "1982Q1", "1987Q4"),
        ("1988--1999", "1988Q1", "1999Q4"),
        ("2000--2007", "2000Q1", "2007Q4"),
        ("2008--2012", "2008Q1", "2012Q4"),
    ]
    total_design = float(n_resid @ n_resid)
    total_numerator = float(n_resid @ y_resid)
    identification_rows = []
    for label, start, end in groups:
        mask = sample["period"].between(pd.Period(start), pd.Period(end)).to_numpy()
        identification_rows.append(
            {
                "period": label,
                "n_obs": int(mask.sum()),
                "design_share": float(n_resid[mask] @ n_resid[mask] / total_design),
                "numerator_share": float(
                    n_resid[mask] @ y_resid[mask] / total_numerator
                ),
            }
        )
    pd.DataFrame(identification_rows).to_csv(
        TABLE_DIR / "identification_by_period.csv", index=False
    )

    horse_x = np.column_stack(
        [np.ones(len(sample)), inertia, activity, interaction_n, interaction_time]
    )
    horse_beta, horse_se, _ = _ols(y, horse_x)
    diagnostics = pd.DataFrame(
        [
            ("corr(Nbar, time)", np.corrcoef(nbar_mean, time)[0, 1], np.nan, np.nan),
            (
                "corr(x*Nbar, x*time)",
                np.corrcoef(interaction_n, interaction_time)[0, 1],
                np.nan,
                np.nan,
            ),
            (
                "corr(residual x*Nbar, residual x*time)",
                np.corrcoef(n_resid, time_resid)[0, 1],
                np.nan,
                np.nan,
            ),
            (
                "horse-race x*Nbar",
                horse_beta[3],
                horse_se[3],
                horse_beta[3] / horse_se[3],
            ),
            (
                "horse-race x*time",
                horse_beta[4],
                horse_se[4],
                horse_beta[4] / horse_se[4],
            ),
        ],
        columns=["diagnostic", "estimate", "standard_error", "t_stat"],
    )
    diagnostics.to_csv(TABLE_DIR / "trend_collinearity.csv", index=False)

    trend_rows = []
    for label, trend in [
        ("posterior Nbar", nbar_mean),
        ("observed N", observed_n),
        ("source BN trend", source_bn),
    ]:
        design = np.column_stack(
            [np.ones(len(sample)), inertia, activity, activity * trend]
        )
        beta, se, resid = _ols(y, design)
        trend_rows.append(
            {
                "competition_trend": label,
                "trend_start": trend[0],
                "trend_end": trend[-1],
                "trend_change": trend[-1] - trend[0],
                "kappa_0": beta[2],
                "kappa_0_se": se[2],
                "delta": beta[3],
                "delta_se": se[3],
                "kappa_start": beta[2] + beta[3] * trend[0],
                "kappa_end": beta[2] + beta[3] * trend[-1],
                "rmse": float(np.sqrt(np.mean(resid**2))),
            }
        )
    pd.DataFrame(trend_rows).to_csv(
        TABLE_DIR / "competition_trend_sensitivity.csv", index=False
    )
    return nbar_mean, observed_n, source_bn


def write_mechanism_check(sample: pd.DataFrame, kappa_t: np.ndarray) -> None:
    start = sample.iloc[:4]
    end = sample.iloc[-4:]
    n_start, n_end = start["N_Gustavo"].mean(), end["N_Gustavo"].mean()
    mu_start, mu_end = start["markup"].mean(), end["markup"].mean()
    inv_start, inv_end = start["markup_inv"].mean(), end["markup_inv"].mean()

    n = sample["N_Gustavo"].to_numpy()
    markup = sample["markup"].to_numpy()
    time = np.arange(len(sample), dtype=float)
    n_detrended = _residualize(n, np.column_stack([np.ones(len(n)), time]))
    markup_detrended = _residualize(
        markup, np.column_stack([np.ones(len(markup)), time])
    )
    kappa_start = float(kappa_t[:, 0].mean())
    kappa_end = float(kappa_t[:, -1].mean())
    markup_implied_end = kappa_start * (mu_start - 1.0) / (mu_end - 1.0)
    explained_fraction = (kappa_start - markup_implied_end) / (
        kappa_start - kappa_end
    )
    table = pd.DataFrame(
        [
            ("listed firm count", n_start, n_end, n_end / n_start - 1.0),
            ("aggregate markup", mu_start, mu_end, mu_end / mu_start - 1.0),
            ("inverse markup", inv_start, inv_end, inv_end / inv_start - 1.0),
        ],
        columns=["series", "1982_average", "2012_average", "percent_change"],
    )
    table.to_csv(TABLE_DIR / "mechanism_endpoints.csv", index=False)
    pd.DataFrame(
        [
            ("corr(N, markup), level", np.corrcoef(n, markup)[0, 1]),
            ("corr(change N, change markup)", np.corrcoef(np.diff(n), np.diff(markup))[0, 1]),
            (
                "corr(detrended N, detrended markup)",
                np.corrcoef(n_detrended, markup_detrended)[0, 1],
            ),
            ("estimated kappa start", kappa_start),
            ("estimated kappa end", kappa_end),
            ("markup-implied kappa end", markup_implied_end),
            ("fraction of estimated flattening matched by markup", explained_fraction),
        ],
        columns=["diagnostic", "value"],
    ).to_csv(TABLE_DIR / "mechanism_diagnostics.csv", index=False)


def write_pass_through(sample: pd.DataFrame, nbar_mean: np.ndarray) -> None:
    working = sample.copy()

    rows = []
    for window_years in [10, 15, 20]:
        window = 4 * window_years
        for specification in ["current PPI", "current plus four lags"]:
            estimates = []
            nbar_endpoints = []
            for end in range(window - 1, len(working)):
                block = working.iloc[end - window + 1 : end + 1]
                if specification == "current PPI":
                    regressors = ["pi_ppi"]
                else:
                    regressors = [f"ppi_lag_{lag}" for lag in range(5)]
                block = block.dropna(subset=["pi_cpi_core", *regressors])
                x = np.column_stack(
                    [np.ones(len(block)), *[block[col].to_numpy() for col in regressors]]
                )
                beta, _, _ = _ols(block["pi_cpi_core"].to_numpy(), x)
                estimates.append(float(beta[1:].sum()))
                nbar_endpoints.append(float(nbar_mean[end]))
            estimates_array = np.asarray(estimates)
            rows.append(
                {
                    "specification": specification,
                    "window_years": window_years,
                    "first_estimate": estimates_array[0],
                    "last_estimate": estimates_array[-1],
                    "corr_with_posterior_Nbar": np.corrcoef(
                        estimates_array, np.asarray(nbar_endpoints)
                    )[0, 1],
                }
            )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "ppi_core_pass_through.csv", index=False)


def save_identification_figure(
    sample: pd.DataFrame,
    nbar_mean: np.ndarray,
    observed_n: np.ndarray,
    source_bn: np.ndarray,
) -> None:
    shares = pd.read_csv(TABLE_DIR / "identification_by_period.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    dates = sample["DATE"]
    axes[0].plot(dates, observed_n, color="#2f5597", lw=1.8, label="Observed listed-firm count")
    axes[0].plot(dates, source_bn, color="#70ad47", lw=1.8, ls="--", label="Source BN trend")
    axes[0].plot(dates, nbar_mean, color="#c00000", lw=2.0, label="Posterior latent trend")
    axes[0].axhline(0.0, color="0.75", lw=0.8)
    axes[0].set_title("(a) Alternative firm-count trends")
    axes[0].set_ylabel("10-log-point transformed units")
    axes[0].legend(frameon=False, fontsize=8)

    bars = axes[1].bar(
        shares["period"],
        100.0 * shares["design_share"],
        color=["#4472c4", "#a5a5a5", "#a5a5a5", "#ed7d31"],
    )
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    axes[1].set_ylim(0.0, 70.0)
    axes[1].set_ylabel("Share of residual interaction variation (%)")
    axes[1].set_title("(b) Where the interaction is identified")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "competition_trend_identification.png", dpi=220)
    plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sample = _sample_data()
    nbar, kappa_t = _posterior_arrays()
    write_economic_magnitude(sample, kappa_t)
    nbar_mean, observed_n, source_bn = write_trend_identification(sample, nbar)
    write_mechanism_check(sample, kappa_t)
    write_pass_through(sample, nbar_mean)
    save_identification_figure(sample, nbar_mean, observed_n, source_bn)
    write_report_tex_inputs()
    print(f"Saved report additions to {TABLE_DIR} and {FIGURE_DIR}")


if __name__ == "__main__":
    main()
