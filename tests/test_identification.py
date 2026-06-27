from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from nkpc_hsa.inference.identification import data_scale_table, identification_table, model_sample


def test_identification_table_reports_hsa_interaction_design() -> None:
    dates = pd.date_range("2000-01-01", periods=6, freq="QE")
    data = pd.DataFrame(
        {
            "DATE": dates,
            "pi": [2.0, 2.2, 2.1, 2.4, 2.5, 2.6],
            "pi_prev": [1.9, 2.0, 2.2, 2.1, 2.4, 2.5],
            "Epi": [2.0, 2.0, 2.0, 2.1, 2.2, 2.2],
            "x": [-1.0, -0.5, 0.0, 0.4, 0.7, 1.0],
            "x_prev": [-1.1, -1.0, -0.5, 0.0, 0.4, 0.7],
            "N": np.exp(np.linspace(1.0, 1.2, 6)),
        }
    )
    spec = {
        "name": "fake_gap",
        "pi_col": "pi",
        "pi_prev_col": "pi_prev",
        "pi_expect_col": "Epi",
        "x_col": "x",
        "x_prev_col": "x_prev",
        "n_col": "N",
    }
    sample = model_sample(data, spec)

    coords = {"chain": [0, 1], "draw": [0, 1], "time": np.arange(6)}
    dims = ("chain", "draw")
    state_dims = ("chain", "draw", "time")
    nbar = np.broadcast_to(np.linspace(-0.2, 0.3, 6), (2, 2, 6))
    nhat = np.broadcast_to(sample["N_model"].to_numpy() - np.linspace(-0.2, 0.3, 6), (2, 2, 6))
    posterior = xr.Dataset(
        {
            "kappa_0": (dims, np.array([[0.1, 0.2], [0.1, 0.2]])),
            "delta": (dims, np.array([[0.01, 0.02], [0.01, 0.02]])),
            "kappa_t": (state_dims, 0.1 + 0.02 * nbar),
            "Nbar": (state_dims, nbar),
            "Nhat": (state_dims, nhat),
        },
        coords=coords,
    )
    idata = SimpleNamespace(
        posterior=posterior,
        attrs={
            "model": "hsa_steady",
            "data_spec": "fake_gap",
            "prior_spec": "baseline",
            "constraint_spec": "unrestricted",
        },
    )

    scale = data_scale_table(data, {"fake_gap": spec})
    assert scale.loc[0, "n_obs"] == 6
    table = identification_table({"fake_run": idata}, {"fake_gap": sample})
    assert table.loc[0, "delta_mean"] == 0.015
    assert table.loc[0, "p_delta_gt0"] == 1.0
    assert table.loc[0, "p_all_kappa_t_gt0"] == 1.0
    assert table.loc[0, "N_decomposition_rmse"] < 1e-12
