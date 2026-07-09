from __future__ import annotations

import numpy as np

from nkpc_hsa.gibbs.common.constraints import draw_with_constraints
from nkpc_hsa.models.common import (
    KAPPA_SCALE,
    coefficient_constraints_to_internal,
    force_symmetric_positive_definite,
    kappa_internal_to_physical,
    kappa_physical_to_internal,
    prior_specs_to_internal,
)


def test_positive_definite_repair_is_symmetric_pd() -> None:
    repaired = force_symmetric_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert np.allclose(repaired, repaired.T)
    assert np.all(np.linalg.eigvalsh(repaired) > 0.0)


def test_kappa_scaling_conversion() -> None:
    assert float(kappa_physical_to_internal(0.1)) == 0.1 * KAPPA_SCALE
    assert float(kappa_internal_to_physical(10.0)) == 10.0 / KAPPA_SCALE


def test_prior_conversion_uses_physical_units() -> None:
    out = prior_specs_to_internal({"kappa": [0.1, 0.2], "kappa_0": [0.1, 0.2], "delta": [0.0, 0.5]})
    assert out["mu_kappa"] == 10.0
    assert out["sigma_kappa"] == 20.0
    assert out["mu_kappa_0"] == 10.0
    assert out["sigma_delta"] == 50.0


def test_coefficient_constraints_convert_kappa_like_bounds() -> None:
    out = coefficient_constraints_to_internal(
        {
            "enabled": True,
            "positive": ["kappa", "kappa_t", "theta"],
            "bounds": {"alpha": [0.0, 1.0], "delta": [-0.1, 0.2]},
        }
    )
    assert out["enabled"] is True
    assert out["bounds"]["kappa"] == [0.0, None]
    assert out["bounds"]["kappa_t"] == [0.0, None]
    assert out["bounds"]["theta"] == [0.0, None]
    assert out["bounds"]["alpha"] == [0.0, 1.0]
    assert out["bounds"]["delta"] == [-0.1 * KAPPA_SCALE, 0.2 * KAPPA_SCALE]


def test_draw_with_constraints_applies_path_validator() -> None:
    draws = iter(
        [
            np.array([0.5, -1.0, 0.0]),
            np.array([0.5, 1.0, 0.0]),
        ]
    )
    stats: dict[str, int] = {}
    out = draw_with_constraints(
        lambda: next(draws),
        ("alpha", "kappa_0", "delta"),
        {"enabled": True, "max_tries": 5, "bounds": {"kappa_t": [0.0, None]}},
        validators=[lambda beta: np.all(beta[1] + beta[2] * np.array([-1.0, 0.0, 1.0]) >= 0.0)],
        stats=stats,
    )
    assert out[1] == 1.0
    assert stats["rejections"] == 1
