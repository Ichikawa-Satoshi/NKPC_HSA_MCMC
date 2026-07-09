from __future__ import annotations

import json

import numpy as np

from nkpc_hsa.inference.diagnostics import ar2_nonstationary_share, compute_diagnostics
from nkpc_hsa.inference.wrappers import run_model


def test_wrapper_saves_metadata(tmp_path) -> None:
    t = np.arange(12, dtype=float)
    data = {
        "pi": 0.2 + 0.01 * t,
        "pi_prev": 0.1 + 0.01 * t,
        "pi_expect": 0.15 + 0.01 * t,
        "x": np.sin(t / 3.0),
        "x_prev": np.cos(t / 3.0),
    }
    run_dir = tmp_path / "run"
    idata = run_model(
        "ces",
        data=data,
        prior_specs={"kappa": [0.1, 0.2]},
        n_iter=10,
        burn=4,
        thin=1,
        chains=2,
        seed=10,
        run_dir=run_dir,
    )
    assert (run_dir / "posterior.nc").exists()
    assert (run_dir / "metadata.json").exists()
    assert idata.attrs["kappa_units"] == "physical"
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_transform"] == "log100_centered10"
    assert "ten log-point" in metadata["n_transform_note"]


def test_wrapper_applies_positive_coefficient_constraint(tmp_path) -> None:
    t = np.arange(14, dtype=float)
    data = {
        "pi": 0.2 + 0.01 * t,
        "pi_prev": 0.1 + 0.01 * t,
        "pi_expect": 0.15 + 0.01 * t,
        "x": np.linspace(0.1, 1.0, t.size),
        "x_prev": np.linspace(0.0, 0.9, t.size),
    }
    run_dir = tmp_path / "constrained"
    idata = run_model(
        "ces",
        data=data,
        prior_specs={"kappa": [0.1, 0.2]},
        coefficient_constraints={"enabled": True, "positive": ["kappa"], "max_tries": 2000},
        n_iter=12,
        burn=4,
        thin=1,
        chains=2,
        seed=11,
        run_dir=run_dir,
    )
    assert np.all(idata.posterior["kappa"].values >= 0.0)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["coefficient_constraints"]["enabled"] is True


def test_diagnostics_on_fake_posterior() -> None:
    import arviz as az

    idata = az.from_dict(
        {
            "posterior": {
                "alpha": np.random.default_rng(1).normal(size=(2, 20)),
                "rho_1": np.zeros((2, 20)),
                "rho_2": np.zeros((2, 20)),
            }
        }
    )
    table = compute_diagnostics(idata)
    assert "parameter" in table.columns
    alpha = table.loc[table["parameter"] == "alpha"].iloc[0]
    assert np.isfinite(alpha["r_hat"])
    assert np.isfinite(alpha["ess_bulk"])
    assert ar2_nonstationary_share(idata) == 0.0


def test_hsa_full_uses_sigma_N_measurement_error(tmp_path) -> None:
    rng = np.random.default_rng(3)
    t = np.arange(30, dtype=float)
    data = {
        "pi": 2.0 + 0.3 * np.sin(t / 4.0) + 0.1 * rng.standard_normal(t.size),
        "pi_prev": 2.0 + 0.3 * np.sin((t - 1) / 4.0),
        "pi_expect": np.full(t.size, 2.0),
        "x": np.sin(t / 5.0),
        "x_prev": np.sin((t - 1) / 5.0),
        "N": np.exp(2.0 + 0.02 * np.sin(t / 6.0)),
    }
    run_dir = tmp_path / "full"
    idata = run_model(
        "hsa_full",
        data=data,
        prior_specs={"a_N": 2.0, "b_N": 0.01},
        n_iter=12,
        burn=4,
        thin=1,
        chains=2,
        seed=7,
        run_dir=run_dir,
    )
    assert "sigma_N" in idata.posterior
    assert np.all(idata.posterior["sigma_N"].values > 0.0)


def test_chib_priors_are_threaded() -> None:
    from nkpc_hsa.gibbs.gibbs_marginal_likelihood import _resolve_priors

    pri = _resolve_priors({"delta": [0.0, 0.02], "rho_1": [0.5, 0.2], "a_N": 2.0})
    assert pri["delta"] == (0.0, 0.02)
    assert pri["rho_1"] == (0.5, 0.2)
    assert pri["rho_2"] == (-0.5, 0.2)
    assert pri["a_N"] == 2.0
