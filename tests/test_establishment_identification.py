from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from nkpc_hsa.dataprep.qcew import load_qcew_national_private_establishments
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.gibbs.common.joint_ffbs import build_joint_ne_system, sample_joint_ne_states_ffbs
from nkpc_hsa.gibbs.hsa_steady.model import func_nkpc_hsa_decomp_tv_kappa_kalman
from nkpc_hsa.inference.wrappers import _coerce_model_data


def test_qcew_loader_selects_published_national_private_quarterly_count(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "area_fips": ["US000"] * 4 + ["US000", "01000"],
            "own_code": [5] * 4 + [3, 5],
            "industry_code": ["10"] * 6,
            "industry_title": ["Total, all industries"] * 6,
            "year": [2000] * 6,
            "qtr": [1, 2, 3, 4, 1, 1],
            "qtrly_estabs": [5_000_001, 5_000_002, 5_000_003, 5_000_004, 99, 88],
        }
    )
    csv_path = tmp_path / "fixture.csv"
    frame.to_csv(csv_path, index=False)
    zip_path = tmp_path / "2000_qtrly_by_industry.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname="2000.q1-q4.by_industry.csv")
    csv_path.unlink()

    out = load_qcew_national_private_establishments(tmp_path, start_year=2000, end_year=2000)
    assert out["quarter"].astype(str).tolist() == ["2000Q1", "2000Q2", "2000Q3", "2000Q4"]
    assert out["qcew_establishments"].tolist() == [5_000_001, 5_000_002, 5_000_003, 5_000_004]


def test_joint_ne_state_order_transition_and_covariance() -> None:
    F, c, Q = build_joint_ne_system(
        rho_N1=0.6, rho_N2=-0.2, rho_E1=0.4, rho_E2=-0.1,
        n_N=0.01, n_E=-0.02, sigma_uN=0.2, sigma_uE=0.3,
        rho_NE=-0.5, sigma_epsN=0.04, sigma_epsE=0.05,
    )
    assert F.shape == Q.shape == (6, 6)
    np.testing.assert_array_equal(c, [0.0, 0.0, 0.01, 0.0, 0.0, -0.02])
    np.testing.assert_array_equal(F[0], [0.6, -0.2, 0, 0, 0, 0])
    np.testing.assert_array_equal(F[3], [0, 0, 0, 0.4, -0.1, 0])
    assert Q[0, 3] == Q[3, 0] == -0.5 * 0.2 * 0.3
    assert np.linalg.eigvalsh(Q[np.ix_([0, 2, 3, 5], [0, 2, 3, 5])]).min() > 0.0


def test_joint_ne_transition_generates_requested_cycle_correlation() -> None:
    _, _, Q = build_joint_ne_system(
        rho_N1=0.6, rho_N2=-0.2, rho_E1=0.4, rho_E2=-0.1,
        n_N=0.01, n_E=-0.02, sigma_uN=0.2, sigma_uE=0.3,
        rho_NE=0.65, sigma_epsN=0.04, sigma_epsE=0.05,
    )
    rng = np.random.default_rng(617)
    shocks = rng.multivariate_normal(np.zeros(6), Q, size=100_000, check_valid="ignore")
    assert abs(float(np.corrcoef(shocks[:, 0], shocks[:, 3])[0, 1]) - 0.65) < 0.01
    # The lag-copy coordinates have no structural innovations.
    assert np.all(shocks[:, [1, 4]] == 0.0)


def test_mixed_frequency_drops_only_n_row_and_keeps_quarterly_e() -> None:
    T = 16
    E_obs = 0.3 + 0.02 * np.arange(T)
    kwargs = dict(
        N_obs=np.where(np.arange(T) % 4 == 3, 0.0, np.nan),
        E_obs=E_obs,
        y_tilde=np.zeros(T), h_nhat=np.zeros(T), h_nbar=np.zeros(T),
        rho_N1=0.3, rho_N2=-0.1, rho_E1=0.5, rho_E2=-0.2,
        n_N=0.0, n_E=0.02, sigma_eta2=1.0,
        sigma_uN=0.1, sigma_uE=0.1, rho_NE=0.0,
        sigma_epsN=0.05, sigma_epsE=0.05,
        sigma_N2=1e-4, sigma_E2=1e-7,
        m0=np.zeros(6), P0=np.eye(6),
    )
    rng = np.random.default_rng(41)
    sums = []
    for _ in range(300):
        _, _, ebar, ehat, states = sample_joint_ne_states_ffbs(rng=rng, **kwargs)
        sums.append(ebar + ehat)
        assert states.shape == (T, 6)
    # Q1--Q3 have no N row, but their E level remains tightly measured.
    mean_sum = np.mean(sums, axis=0)
    assert np.max(np.abs(mean_sum - E_obs)) < 0.002


def test_joint_data_transform_uses_establishment_own_mean() -> None:
    index = pd.date_range("2000-03-31", periods=8, freq="QE")
    frame = pd.DataFrame(
        {
            "pi": np.arange(8.0), "pi_prev": np.arange(8.0), "pi_expect": np.zeros(8),
            "x": np.arange(8.0), "x_prev": np.arange(8.0),
            "N": np.linspace(100, 120, 8), "E": np.linspace(1_000_000, 1_100_000, 8),
        }, index=index,
    )
    out = _coerce_model_data(
        frame,
        data_spec={"n_col": "N", "e_col": "E", "e_transform": "log100_centered10", "establishment_model": "joint_state"},
    )
    assert "E_obs" in out and "Ehat" not in out
    assert abs(float(np.mean(out["E_obs"]))) < 1e-12
    expected = (100 * np.log(frame["E"].to_numpy()) - np.mean(100 * np.log(frame["E"].to_numpy()))) / 10
    np.testing.assert_allclose(out["E_obs"], expected)


def test_qcew_config_window_is_json_serializable() -> None:
    import json

    root = Path(__file__).resolve().parents[1]
    config = load_model_config(root / "configs" / "models.yaml")
    spec = configured_data_specs(config, ["unemployment_gap_core_qcew_joint"])[
        "unemployment_gap_core_qcew_joint"
    ]
    json.dumps(spec)
    assert spec["sample_start"] == "1982-01-01"
    assert spec["sample_end"] == "2012-12-31"


def test_joint_sampler_stores_distinct_n_e_paths_and_correlation() -> None:
    rng = np.random.default_rng(8)
    T = 28
    x = rng.normal(size=T)
    n_level = 0.1 + np.cumsum(rng.normal(scale=0.01, size=T))
    e_level = -0.3 + np.cumsum(rng.normal(scale=0.02, size=T))
    result = func_nkpc_hsa_decomp_tv_kappa_kalman(
        pi_data=rng.normal(size=T), pi_prev_data=rng.normal(size=T), Epi_data=np.zeros(T),
        x_data=x, x_prev_data=np.r_[0.0, x[:-1]],
        N_data=np.where(np.arange(T) % 4 == 3, n_level, np.nan), E_data=e_level,
        n_burn=5, n_keep=10,
        priors={"nu_NE": 5.0, "S_NE": [[0.04, 0.0], [0.0, 0.04]], "a_E": 2.0, "b_E": 0.01},
        opts={"seed": 81},
    )
    assert result["rho_NE"]["draws"].shape == (10,)
    assert np.all(np.abs(result["rho_NE"]["draws"]) < 1.0)
    assert result["state_draws"]["Nbar"].shape == result["state_draws"]["Ebar"].shape == (10, T)
    assert result["model"]["inflation_establishment_loading"] == 0.0
    assert result["model"]["state_vector"].startswith("[Nhat_t, Nhat_{t-1}, Nbar_t, Ehat_t")
