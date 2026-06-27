from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from nkpc_hsa.inference.period_robustness import apply_period
from nkpc_hsa.inference.model_comparison import model_comparison_table
from nkpc_hsa.report.latex import write_default_report
from nkpc_hsa.report.tables import write_latex_fragment


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


def test_period_filter_excludes_covid() -> None:
    dates = pd.date_range("2019-01-01", periods=12, freq="QE")
    df = pd.DataFrame({"x": range(12)}, index=dates)
    out = apply_period(df, {"exclude": [["2020-01-01", "2020-12-31"]]})
    assert not ((out.index >= "2020-01-01") & (out.index <= "2020-12-31")).any()


def test_hsa_full_conditional_chib_is_computed() -> None:
    from analysis.gibbs.func_gibbs.gibbs_marginal_likelihood import chib_conditional_marginal_likelihood

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
        },
    )
    table = model_comparison_table({"hsa_full_fake": idata}, data_by_model={"hsa_full_fake": data})
    assert np.isfinite(float(table.loc[0, "log_marginal_likelihood"]))
    assert "conditional on posterior mean latent states" in table.loc[0, "notes"]
