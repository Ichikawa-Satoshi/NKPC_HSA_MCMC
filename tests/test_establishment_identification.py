from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.dataprep.build import load_quarterly_establishment_stock
from nkpc_hsa.gibbs.common.joint_ffbs import sample_joint_competition_states_ffbs
from nkpc_hsa.gibbs.hsa_const_theta.model import func_nkpc_hsa_const_theta
from nkpc_hsa.gibbs.hsa_steady.model import func_nkpc_hsa_decomp_tv_kappa_kalman
from nkpc_hsa.inference.wrappers import _coerce_model_data, model_sample_index
from nkpc_hsa.paths import data_root

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = data_root(ROOT)


def test_restored_bed_flows_produce_the_declared_quarterly_stock() -> None:
    out = load_quarterly_establishment_stock(DATA_DIR / "raw")
    assert out.index[0].to_period("Q") == pd.Period("1993Q2", freq="Q")
    assert out.index[-1].to_period("Q") == pd.Period("2023Q3", freq="Q")
    assert out.loc[out.index[0], "establishment_births"] == 181_000.0
    assert out.loc[out.index[0], "establishment_deaths"] == 160_000.0
    # 1993 BDS ESTAB (5,682,098) is the Q1 level anchor; Q2 adds its net flow.
    assert out.loc[out.index[0], "establishment_stock"] == 5_703_098.0
    np.testing.assert_allclose(
        out["establishment_stock"].diff().iloc[1:],
        out["establishment_net_entry"].iloc[1:],
    )


def test_establishment_spec_is_exactly_79_quarters_and_builds_ehat() -> None:
    data = pd.read_csv(DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]).set_index("DATE")
    config = load_model_config(ROOT / "configs" / "models.yaml")
    spec = configured_data_specs(
        config, ["unemployment_gap_core_establishment"]
    )["unemployment_gap_core_establishment"]
    index = model_sample_index(data, spec)
    model_data = _coerce_model_data(data, data_spec=spec)

    assert index is not None
    assert len(index) == 79
    assert index[0].to_period("Q") == pd.Period("1993Q2", freq="Q")
    assert index[-1].to_period("Q") == pd.Period("2012Q4", freq="Q")
    assert np.all(np.isfinite(model_data["Ehat"]))
    assert abs(float(np.mean(model_data["Ehat"]))) < 1e-10
    assert float(np.std(model_data["Ehat"], ddof=1)) > 0.01


def test_establishment_row_directly_informs_quarterly_nhat() -> None:
    T = 24
    target = 0.15 * np.sin(np.arange(T) * 2.0 * np.pi / 8.0)
    lambda_E = 1.7
    kwargs = dict(
        N_obs=np.full(T, np.nan),
        y_tilde=np.zeros(T),
        h_nhat=np.zeros(T),
        h_nbar=np.zeros(T),
        n_drift=0.0,
        rho1=0.5,
        rho2=-0.2,
        sigma_eta2=1.0,
        sigma_u2=0.05,
        sigma_eps2=0.05,
        sigma_N2=1.0,
        m0=np.zeros(3),
        P0=np.eye(3),
        Ehat_obs=lambda_E * target,
        lambda_E=lambda_E,
        sigma_E2=1e-5,
    )
    rng = np.random.default_rng(90210)
    draws = np.array([
        sample_joint_competition_states_ffbs(rng=rng, **kwargs)[1]
        for _ in range(200)
    ])
    assert float(np.max(np.abs(draws.mean(axis=0) - target))) < 0.01


def test_hsa_steady_samples_lambda_E_and_its_measurement_variance() -> None:
    rng = np.random.default_rng(8)
    T = 32
    x = rng.normal(size=T)
    nhat = 0.1 * np.sin(np.arange(T) * 2.0 * np.pi / 6.0)
    n_obs = np.where(np.arange(T) % 4 == 3, nhat, np.nan)
    result = func_nkpc_hsa_decomp_tv_kappa_kalman(
        pi_data=rng.normal(size=T),
        pi_prev_data=rng.normal(size=T),
        Epi_data=np.zeros(T),
        x_data=x,
        x_prev_data=np.r_[0.0, x[:-1]],
        N_data=n_obs,
        Ehat_data=1.5 * nhat + rng.normal(scale=0.01, size=T),
        n_burn=10,
        n_keep=20,
        priors={"mu_lambda_E": 1.0, "sigma_lambda_E": 1.0, "a_E": 2.0, "b_E": 0.01},
        opts={"seed": 81},
    )
    assert result["lambda_E"]["draws"].shape == (20,)
    assert result["sigma_E"]["draws"].shape == (20,)
    assert np.all(np.isfinite(result["lambda_E"]["draws"]))
    assert np.all(result["sigma_E"]["draws"] > 0.0)
    assert result["model"]["establishment_measurement"] is True


def test_const_theta_samples_theta_with_the_establishment_measurement() -> None:
    rng = np.random.default_rng(18)
    T = 32
    x = rng.normal(size=T)
    nhat = 0.1 * np.sin(np.arange(T) * 2.0 * np.pi / 6.0)
    n_obs = np.where(np.arange(T) % 4 == 3, nhat, np.nan)
    result = func_nkpc_hsa_const_theta(
        pi_data=rng.normal(size=T),
        pi_prev_data=rng.normal(size=T),
        Epi_data=np.zeros(T),
        x_data=x,
        x_prev_data=np.r_[0.0, x[:-1]],
        N_data=n_obs,
        Ehat_data=1.5 * nhat + rng.normal(scale=0.01, size=T),
        n_burn=10,
        n_keep=20,
        priors={
            "mu_lambda_E": 1.0,
            "sigma_lambda_E": 1.0,
            "a_E": 2.0,
            "b_E": 0.01,
        },
        opts={"seed": 181},
    )
    assert result["theta"]["draws"].shape == (20,)
    assert result["lambda_E"]["draws"].shape == (20,)
    assert result["sigma_E"]["draws"].shape == (20,)
    assert np.all(np.isfinite(result["theta"]["draws"]))
    assert result["model"]["establishment_measurement"] is True
