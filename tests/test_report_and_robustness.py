from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from nkpc_hsa.inference.period_robustness import apply_period, run_period_robustness
from nkpc_hsa.inference.model_comparison import model_comparison_table
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.report.latex import write_default_report
from nkpc_hsa.report.tables import write_latex_fragment
from nkpc_hsa.report.cpi_ppi_spec import annual_q4_run_keys, report_run_keys


def test_latex_table_generation(tmp_path) -> None:
    out = tmp_path / "table.tex"
    write_latex_fragment(pd.DataFrame({"parameter": ["alpha"], "mean": [0.5]}), out)
    text = out.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in text
    assert "alpha" in text


def test_report_source_has_expected_paths(tmp_path) -> None:
    tex = write_default_report(tmp_path / "main.tex")
    text = tex.read_text(encoding="utf-8")
    assert "../tables/result_blocks.tex" in text
    assert "prior specification, and sample period" in text


def test_output_gap_hp_core_is_configured() -> None:
    config = load_model_config()
    assert "output_gap_hp_core" in config["run_data_specs"]
    spec = configured_data_specs(config, ["output_gap_hp_core"])["output_gap_hp_core"]
    assert spec["label"] == "Output gap HP, core CPI"
    assert spec["pi_col"] == "pi_cpi_core"
    assert spec["pi_prev_col"] == "pi_cpi_core_prev"
    assert spec["x_col"] == "output_gap_HP"
    assert spec["x_prev_col"] == "output_gap_HP_prev"


def test_cpi_ppi_report_requires_77_unique_estimation_cells() -> None:
    keys = report_run_keys()
    assert len(keys) == 77
    assert len(set(keys)) == 77
    assert ("ces", "inv_markup", "baseline") in keys
    assert ("hsa_dynamic", "inv_markup", "baseline") in keys


def test_annual_q4_report_reestimates_only_the_61_hsa_cells() -> None:
    keys = annual_q4_run_keys()
    assert len(keys) == 61
    assert all(model != "ces" for model, _, _ in keys)
    assert ("hsa_dynamic", "inv_markup", "baseline") in keys


def test_period_filter_excludes_covid() -> None:
    dates = pd.date_range("2019-01-01", periods=12, freq="QE")
    df = pd.DataFrame({"x": range(12)}, index=dates)
    out = apply_period(df, {"exclude": [["2020-01-01", "2020-12-31"]]})
    assert not ((out.index >= "2020-01-01") & (out.index <= "2020-12-31")).any()


def test_period_filter_uses_inclusive_quarter_boundaries() -> None:
    dates = pd.DatetimeIndex(
        [pd.Timestamp("2007-12-31 23:59:59.999999"), pd.Timestamp("2008-03-31 23:59:59.999999")]
    )
    out = apply_period(pd.DataFrame({"x": [1.0, 2.0]}, index=dates), {"end": "2007-12-31"})
    assert out.index.tolist() == [dates[0]]


def test_period_robustness_skips_an_interior_gap() -> None:
    dates = pd.date_range("2018-01-01", periods=20, freq="QE")
    values = np.arange(20, dtype=float)
    df = pd.DataFrame(
        {"pi": values, "pi_prev": values, "pi_expect": values, "x": values, "x_prev": values, "N": values + 1.0},
        index=dates,
    )
    outputs, table = run_period_robustness(
        "ces",
        data=df,
        periods={"exclude_2020": {"exclude": [["2020-01-01", "2020-12-31"]]}},
        data_spec={},
        min_obs=4,
    )
    assert outputs == {}
    assert table.loc[0, "status"] == "skipped"
    assert "Non-contiguous quarterly sample" in table.loc[0, "warning"]


def test_period_robustness_counts_complete_case_estimation_sample() -> None:
    dates = pd.date_range("2000-01-01", periods=8, freq="QE")
    df = pd.DataFrame(
        {
            "pi": np.arange(8, dtype=float),
            "pi_prev": np.arange(8, dtype=float),
            "pi_expect": np.arange(8, dtype=float),
            "x": np.arange(8, dtype=float),
            "x_prev": np.arange(8, dtype=float),
            "N": [np.nan] * 5 + [1.0, 1.1, 1.2],
        },
        index=dates,
    )

    outputs, table = run_period_robustness(
        "ces",
        data=df,
        periods={"full": {}},
        data_spec={},
        min_obs=4,
    )

    assert outputs == {}
    row = table.iloc[0]
    assert row["status"] == "skipped"
    assert row["n_obs"] == 3
    assert row["start"] == dates[5].date().isoformat()
    assert row["end"] == dates[7].date().isoformat()


def test_hsa_full_conditional_chib_is_computed() -> None:
    from nkpc_hsa.gibbs.gibbs_marginal_likelihood import chib_conditional_marginal_likelihood

    chains, draws, time = 2, 20, 8
    rng = np.random.default_rng(123)
    dims = ("chain", "draw")
    state_dims = ("chain", "draw", "time")
    coords = {"chain": np.arange(chains), "draw": np.arange(draws), "time": np.arange(time)}
    nbar = np.linspace(-0.5, 0.5, time)
    nhat = np.sin(np.linspace(0.0, 1.0, time))
    posterior = xr.Dataset(
        {
            "alpha": (dims, 0.5 + 0.01 * rng.normal(size=(chains, draws))),
            "kappa_0": (dims, 0.1 + 0.01 * rng.normal(size=(chains, draws))),
            "delta": (dims, 0.01 * rng.normal(size=(chains, draws))),
            "theta_0": (dims, 0.1 + 0.01 * rng.normal(size=(chains, draws))),
            "gamma": (dims, 0.01 * rng.normal(size=(chains, draws))),
            "phi_1": (dims, 0.7 + 0.01 * rng.normal(size=(chains, draws))),
            "lambda_ez": (dims, 0.01 * rng.normal(size=(chains, draws))),
            "rho_1": (dims, 0.5 + 0.01 * rng.normal(size=(chains, draws))),
            "rho_2": (dims, -0.5 + 0.01 * rng.normal(size=(chains, draws))),
            "n": (dims, 0.01 * rng.normal(size=(chains, draws))),
            "sigma_e": (dims, np.full((chains, draws), 1.0)),
            "sigma_zeta": (dims, np.full((chains, draws), 1.0)),
            "sigma_u": (dims, np.full((chains, draws), 1.0)),
            "sigma_eps": (dims, np.full((chains, draws), 1.0)),
            "Nbar": (state_dims, np.broadcast_to(nbar, (chains, draws, time))),
            "Nhat": (state_dims, np.broadcast_to(nhat, (chains, draws, time))),
        },
        coords=coords,
    )
    data = {
        "pi": np.linspace(2.0, 3.0, time),
        "pi_prev": np.linspace(1.8, 2.8, time),
        "pi_expect": np.linspace(2.1, 2.9, time),
        "x": np.linspace(-1.0, 1.0, time),
        "x_prev": np.linspace(-0.9, 0.9, time),
        "N": nbar + nhat,
    }

    result = chib_conditional_marginal_likelihood(posterior, data, family="full")
    assert np.isfinite(result.log_marginal_likelihood)
    assert np.isfinite(result.log_likelihood)
    assert "HSA full" in result.method

    idata = SimpleNamespace(
        posterior=posterior,
        attrs={
            "model": "hsa_full",
            "data_spec": "fake",
            "prior_spec": "baseline",
            "constraint_spec": "unrestricted",
            "n_transform": "log100_centered10",
            "estimation_revision": ESTIMATION_REVISION,
        },
    )
    table = model_comparison_table({"hsa_full_fake": idata}, data_by_model={"hsa_full_fake": data})
    assert np.isnan(float(table.loc[0, "log_marginal_likelihood"]))
    assert "Automated Chib marginal likelihood is not reported" in table.loc[0, "notes"]


def test_in_sample_lppd_includes_cross_equation_shock_term() -> None:
    chains, draws, time = 1, 30, 6
    dims = ("chain", "draw")
    x = np.linspace(-1.0, 1.0, time)
    posterior = xr.Dataset(
        {
            "alpha": (dims, np.zeros((chains, draws))),
            "kappa": (dims, np.zeros((chains, draws))),
            "phi_1": (dims, np.zeros((chains, draws))),
            "lambda_ez": (dims, np.full((chains, draws), 2.0)),
            "sigma_e": (dims, np.full((chains, draws), np.sqrt(5.0))),
            "sigma_zeta": (dims, np.ones((chains, draws))),
        }
    )
    idata = SimpleNamespace(
        posterior=posterior,
        attrs={"model": "ces", "estimation_revision": ESTIMATION_REVISION},
    )
    data = {
        "pi": 2.0 * x,
        "pi_prev": np.zeros(time),
        "pi_expect": np.zeros(time),
        "x": x,
        "x_prev": np.zeros(time),
    }
    table = model_comparison_table({"ces_fake": idata}, data_by_model={"ces_fake": data})
    assert float(table.loc[0, "in_sample_posterior_mean_rmse"]) < 1e-12
    assert np.isfinite(float(table.loc[0, "in_sample_conditional_lppd"]))
