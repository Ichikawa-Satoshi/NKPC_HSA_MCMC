from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.gibbs.func_gibbs.common.competition import finite_N_residuals
from nkpc_hsa.data.competition import build_competition_observation, pchip_interpolate_annual_q4
from nkpc_hsa.inference.wrappers import run_model
from nkpc_hsa.report.data_model_report import write_data_model_report
from nkpc_hsa.report.figures import plot_competition_path_comparison


def test_annual_q4_alignment() -> None:
    annual_N = pd.Series([1.0, 2.0, 3.0], index=[2000, 2001, 2002])
    quarterly_index = pd.period_range("2000Q1", "2002Q4", freq="Q")

    obs = build_competition_observation(
        annual_N,
        quarterly_index,
        frequency="annual_q4",
    )

    finite = np.isfinite(obs.N_obs)
    assert np.all(quarterly_index[finite].quarter == 4)
    assert obs.N_obs[quarterly_index.get_loc("2000Q4")] == 1.0
    assert obs.N_obs[quarterly_index.get_loc("2001Q4")] == 2.0
    assert obs.N_obs[quarterly_index.get_loc("2002Q4")] == 3.0
    assert np.all(~np.isfinite(obs.N_obs[quarterly_index.quarter != 4]))


def test_quarterly_interpolated_observation_is_finite() -> None:
    annual_N = pd.Series([1.0, 2.0, 3.0], index=[2000, 2001, 2002])
    quarterly_index = pd.period_range("2000Q1", "2002Q4", freq="Q")

    obs = build_competition_observation(
        annual_N,
        quarterly_index,
        frequency="quarterly_interpolated",
    )

    assert obs.N_obs.shape == (len(quarterly_index),)
    assert np.all(np.isfinite(obs.N_obs))
    assert obs.interpolation_method == "PCHIP"


def test_q4_anchored_pchip_passes_through_annual_q4_points() -> None:
    annual_N = pd.Series([1.0, 2.0, 3.0], index=[2000, 2001, 2002])
    quarterly_index = pd.period_range("2000Q1", "2002Q4", freq="Q")

    comparison = pchip_interpolate_annual_q4(annual_N, quarterly_index)

    assert comparison[quarterly_index.get_loc("2000Q4")] == 1.0
    assert comparison[quarterly_index.get_loc("2001Q4")] == 2.0
    assert comparison[quarterly_index.get_loc("2002Q4")] == 3.0


def test_sigma_N_residuals_use_observed_quarters_only() -> None:
    N_obs = np.array([np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 2.0])
    Nhat = np.zeros_like(N_obs)
    Nbar = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.7])

    resid = finite_N_residuals(N_obs, Nhat, Nbar)

    assert np.allclose(resid, np.array([0.2, 0.3]))
    assert resid.size == 2


def _annual_q4_fake_data(T: int = 16) -> dict[str, np.ndarray]:
    t = np.arange(T, dtype=float)
    N = np.full(T, np.nan, dtype=float)
    N[3::4] = np.linspace(-0.2, 0.2, T // 4)
    return {
        "pi": 2.0 + 0.05 * np.sin(t / 3.0),
        "pi_prev": 2.0 + 0.05 * np.sin((t - 1.0) / 3.0),
        "pi_expect": np.full(T, 2.0),
        "x": 0.2 * np.sin(t / 4.0),
        "x_prev": 0.2 * np.sin((t - 1.0) / 4.0),
        "N": N,
    }


def _assert_no_nan_headline_draws(idata, names: list[str]) -> None:
    for name in names:
        if name in idata.posterior:
            values = np.asarray(idata.posterior[name], dtype=float)
            assert np.all(np.isfinite(values))


def test_hsa_steady_annual_q4_smoke(tmp_path) -> None:
    run_dir = tmp_path / "steady"
    idata = run_model(
        "hsa_steady",
        data=_annual_q4_fake_data(),
        n_transform="identity",
        competition_measurement={"frequency": "annual_q4"},
        prior_specs={"a_N": 2.0, "b_N": 0.01},
        n_iter=8,
        burn=4,
        thin=1,
        chains=1,
        seed=1,
        run_dir=run_dir,
    )

    assert idata.posterior["Nbar"].shape[-1] == 16
    assert idata.posterior["Nhat"].shape[-1] == 16
    assert "sigma_N" in idata.posterior
    _assert_no_nan_headline_draws(idata, ["alpha", "kappa_0", "delta", "sigma_N"])
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["competition_measurement"]["frequency"] == "annual_q4"
    assert metadata["competition_measurement"]["finite_N_obs_count"] == 4
    assert (run_dir / "data_model_report.md").exists()
    assert (run_dir / "report" / "data_model_report.md").exists()
    assert (run_dir / "estimation_results_report.md").exists()
    assert (run_dir / "report" / "estimation_results_report.md").exists()
    assert (run_dir / "posterior_summary.csv").exists()
    assert (run_dir / "tables" / "posterior_summary.csv").exists()
    assert (run_dir / "competition_decomposition_summary.csv").exists()
    assert (run_dir / "tables" / "competition_decomposition_summary.csv").exists()
    decomp = pd.read_csv(run_dir / "tables" / "competition_decomposition_summary.csv")
    assert {"N_total_mean", "Nbar_mean", "Nhat_mean"}.issubset(decomp.columns)


def test_hsa_full_annual_q4_smoke(tmp_path) -> None:
    idata = run_model(
        "hsa_full",
        data=_annual_q4_fake_data(),
        n_transform="identity",
        competition_measurement={"frequency": "annual_q4"},
        prior_specs={"a_N": 2.0, "b_N": 0.01},
        n_iter=8,
        burn=4,
        thin=1,
        chains=1,
        seed=2,
        run_dir=tmp_path / "full",
    )

    assert idata.posterior["Nbar"].shape[-1] == 16
    assert idata.posterior["Nhat"].shape[-1] == 16
    assert "sigma_N" in idata.posterior
    _assert_no_nan_headline_draws(idata, ["alpha", "kappa_0", "delta", "theta_0", "gamma", "sigma_N"])


def test_hsa_dynamic_annual_q4_smoke(tmp_path) -> None:
    idata = run_model(
        "hsa_dynamic",
        data=_annual_q4_fake_data(),
        n_transform="identity",
        competition_measurement={"frequency": "annual_q4"},
        prior_specs={"a_N": 2.0, "b_N": 0.01},
        n_iter=8,
        burn=4,
        thin=1,
        chains=1,
        seed=3,
        run_dir=tmp_path / "dynamic",
    )

    assert idata.posterior["Nbar"].shape[-1] == 16
    assert idata.posterior["Nhat"].shape[-1] == 16
    assert "sigma_N" in idata.posterior
    _assert_no_nan_headline_draws(idata, ["alpha", "kappa", "theta", "sigma_N"])


def test_data_model_report_documents_competition_mode(tmp_path) -> None:
    path = write_data_model_report(
        tmp_path,
        run_or_batch_metadata={"run_id": "fake", "model": "hsa_steady", "n_obs": 8},
        sample_metadata={"sample_start": "2000Q1", "sample_end": "2001Q4", "T": 8},
        data_metadata={"pi_col": "pi", "pi_expect_col": "Epi", "pi_prev_col": "pi_prev", "x_col": "x", "n_col": "N_Gustavo"},
        competition_metadata={
            "frequency": "annual_q4",
            "annual_timing": "q4",
            "finite_N_obs_count": 2,
            "missing_N_obs_count": 6,
            "interpolation_method": "none",
        },
        model_variant_metadata=[{"model": "hsa_steady"}],
        priors_metadata={"alpha": [0.5, 0.2]},
    )

    text = path.read_text(encoding="utf-8")
    assert "annual_q4" in text
    assert "finite_N_obs_count:** 2" in text
    assert "Model Variants" in text
    assert not (tmp_path / "data_model_report.json").exists()


def test_competition_path_plot_smoke(tmp_path) -> None:
    quarterly_index = pd.period_range("2000Q1", "2002Q4", freq="Q")
    rng = np.random.default_rng(10)
    nbar = rng.normal(0.0, 0.05, size=(2, 4, len(quarterly_index)))
    nhat = rng.normal(0.0, 0.02, size=(2, 4, len(quarterly_index)))
    annual = np.full(len(quarterly_index), np.nan)
    annual[quarterly_index.quarter == 4] = np.linspace(-0.1, 0.1, 3)
    comparison = np.linspace(-0.12, 0.12, len(quarterly_index))

    path = plot_competition_path_comparison(
        quarterly_index,
        nbar,
        nhat,
        annual,
        annual,
        comparison,
        "hsa_steady",
        "unemployment_gap",
        "annual_q4",
        tmp_path,
    )

    assert path.exists()
    assert path.stat().st_size > 0
