"""Every configured prior field must reach the sampler, in the right units.

The failure this guards against is silent: a prior mapper that drops a key does
not raise, it just lets the sampler fall back to a hard-coded default. That is
how ``b_u``/``b_eps``/``b_N`` could end up at 2.0 instead of the configured
0.02/0.01/0.01 -- two orders of magnitude off for state variances that live in
the 0.01 decade -- with no error anywhere.

Covered here:
  * every key of every shipped priors_*.yaml reaches the sampler's resolved
    prior dict, for every model;
  * KAPPA_SCALE-treated parameters are converted, and only those;
  * the deprecated legacy wrapper agrees with the authoritative mapper.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from nkpc_hsa.models.common import KAPPA_SCALE, prior_specs_to_internal

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PRIOR_FILES = sorted(CONFIG_DIR.glob("priors_*.yaml"))

# config key -> (sampler mean key, sampler sd key, scaled by KAPPA_SCALE?)
NORMAL_PRIORS = {
    "alpha": ("mu_alpha", "sigma_alpha", False),
    "kappa": ("mu_kappa", "sigma_kappa", True),
    "kappa_0": ("mu_kappa_0", "sigma_kappa_0", True),
    "delta": ("mu_delta", "sigma_delta", True),
    "theta": ("mu_theta", "sigma_theta", False),
    "theta_0": ("mu_theta", "sigma_theta", False),
    "gamma": ("mu_gamma", "sigma_gamma", False),
    "phi_1": ("mu_phi_1", "sigma_phi_1", False),
    "rho_1": ("mu_rho1", "sigma_rho1", False),
    "rho_2": ("mu_rho2", "sigma_rho2", False),
    "rho_E1": ("mu_rho_E1", "sigma_rho_E1", False),
    "rho_E2": ("mu_rho_E2", "sigma_rho_E2", False),
    "n": ("mu_n", "sigma_n", False),
    "n_E": ("mu_n_E", "sigma_n_E", False),
    "lambda_E": ("mu_lambda_E", "sigma_lambda_E", False),
}
SCALAR_PRIORS = [
    "a_e", "b_e", "a_z", "b_z", "a_u", "b_u", "a_eps", "b_eps",
    "a_N", "b_N", "a_E", "b_E", "a_epsE", "b_epsE", "b_uE",
    "nu_NE", "nu_Sigma",
]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_prior_files_exist():
    assert PRIOR_FILES, "no priors_*.yaml found"


@pytest.mark.parametrize("path", PRIOR_FILES, ids=lambda p: p.stem)
def test_every_configured_key_reaches_the_sampler(path):
    spec = _load(path)
    internal = prior_specs_to_internal(spec)

    unmapped = []
    for key, value in spec.items():
        if key in NORMAL_PRIORS:
            mean_key, sd_key, _ = NORMAL_PRIORS[key]
            if mean_key not in internal or sd_key not in internal:
                unmapped.append(key)
        elif key in SCALAR_PRIORS:
            if key not in internal:
                unmapped.append(key)
        elif key in {"S_NE", "S_Sigma"}:
            if key not in internal:
                unmapped.append(key)
        else:
            pytest.fail(f"{path.name}: key {key!r} is not covered by this test; extend the map")
    assert not unmapped, f"{path.name}: prior keys dropped before reaching the sampler: {unmapped}"


@pytest.mark.parametrize("path", PRIOR_FILES, ids=lambda p: p.stem)
def test_kappa_scale_conversion_is_applied_exactly_where_expected(path):
    spec = _load(path)
    internal = prior_specs_to_internal(spec)
    for key, (mean_key, sd_key, scaled) in NORMAL_PRIORS.items():
        if key not in spec:
            continue
        mean, sd = float(spec[key][0]), float(spec[key][1])
        factor = KAPPA_SCALE if scaled else 1.0
        assert internal[mean_key] == pytest.approx(mean * factor), f"{path.name}:{key} mean"
        assert internal[sd_key] == pytest.approx(sd * factor), f"{path.name}:{key} sd"


@pytest.mark.parametrize("path", PRIOR_FILES, ids=lambda p: p.stem)
def test_state_variance_scales_stay_in_the_transformed_n_decade(path):
    """b_u / b_eps / b_N must not drift back to O(1).

    The transformed N series has quarterly variance around 0.01, so an
    inverse-gamma scale of order 1 forces the state variances two orders of
    magnitude too large and makes the Nbar/Nhat decomposition spuriously noisy.
    """
    internal = prior_specs_to_internal(_load(path))
    for key in ("b_u", "b_eps", "b_N"):
        assert internal[key] < 0.5, f"{path.name}: {key}={internal[key]} is outside the 0.01 decade"


@pytest.mark.parametrize("path", PRIOR_FILES, ids=lambda p: p.stem)
def test_all_models_receive_the_configured_hyperparameters(path):
    """Resolve the internal dict through each sampler's own prior resolver."""
    from nkpc_hsa.gibbs.hsa_dynamic.model import _common_priors as dynamic_priors
    from nkpc_hsa.gibbs.hsa_full.model import _common_priors as full_priors
    from nkpc_hsa.gibbs.hsa_steady.model import _common_priors as steady_priors

    spec = _load(path)
    internal = prior_specs_to_internal(spec)

    # hsa_dynamic draws the u / epsilon variances from the Sigma block
    # (nu_Sigma, S_Sigma rows [e, zeta, u, eps]) rather than from the
    # inverse-gamma scales, so a_u/b_u/a_eps/b_eps are deliberately absent from
    # its resolver. That is a modelling choice, not a dropped prior.
    for resolver, label, keys in (
        (
            steady_priors,
            "hsa_steady",
            [
                "mu_n", "sigma_n", "a_u", "b_u", "a_eps", "b_eps", "a_N", "b_N",
                "mu_lambda_E", "sigma_lambda_E", "a_E", "b_E",
            ],
        ),
        (
            full_priors,
            "hsa_full/hsa_const_theta",
            ["mu_theta", "sigma_theta", "mu_gamma", "sigma_gamma", "mu_n", "sigma_n", "b_u", "b_eps", "b_N"],
        ),
        (dynamic_priors, "hsa_dynamic", ["mu_theta", "sigma_theta", "mu_n", "sigma_n", "a_N", "b_N", "nu_Sigma"]),
    ):
        resolved = resolver(internal)
        for key in keys:
            expected = internal.get(key)
            if expected is None:
                continue
            assert resolved[key] == pytest.approx(expected), f"{path.name}/{label}: {key} not delivered"

    # The dynamic model's state-variance prior must still carry the configured
    # S_Sigma, in the transformed-N decade for the u and eps rows.
    import numpy as np

    dynamic = dynamic_priors(internal)
    s_sigma = np.asarray(dynamic["S_Sigma"], dtype=float)
    assert s_sigma.shape == (4, 4)
    assert np.allclose(s_sigma, np.asarray(spec["S_Sigma"], dtype=float))
    assert s_sigma[2, 2] < 0.5 and s_sigma[3, 3] < 0.5, "S_Sigma u/eps rows left the 0.01 decade"


def test_gamma_prior_is_not_kappa_scaled():
    """gamma multiplies an unscaled regressor, so it must NOT be rescaled."""
    internal = prior_specs_to_internal({"gamma": [0.0, 0.02], "delta": [0.0, 0.02]})
    assert internal["sigma_gamma"] == pytest.approx(0.02)
    assert internal["sigma_delta"] == pytest.approx(2.0)


def test_legacy_wrapper_delegates_and_warns():
    """The deprecated wrapper must not reimplement the mapping."""
    from nkpc_hsa.gibbs.gibbs_wrappers import _prior_specs_to_dict

    spec = _load(CONFIG_DIR / "priors_baseline.yaml")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = _prior_specs_to_dict(spec)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert legacy == prior_specs_to_internal(spec)


def test_legacy_wrapper_is_not_publicly_re_exported():
    """gibbs/__init__ must not hand out the deprecated wrappers."""
    import nkpc_hsa.gibbs as gibbs

    for name in ("run_ces", "run_hsa_steady", "run_hsa_full", "draws_to_idata"):
        assert not hasattr(gibbs, name), f"{name} is still re-exported from nkpc_hsa.gibbs"
