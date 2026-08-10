from __future__ import annotations

import numpy as np
import pytest
import arviz as az
from pathlib import Path

from nkpc_hsa.config import load_model_config
from nkpc_hsa.gibbs.hsa_full.model import _sample_beta_gaussian_linear_constraints
from nkpc_hsa.inference.diagnostics import compute_diagnostics
from nkpc_hsa.inference.wrappers import _extract_draws_from_result
from nkpc_hsa.models.common import KAPPA_SCALE
from nkpc_hsa.provenance import StaleArtifactError, stamp_artifact_metadata, validate_artifact_metadata
from nkpc_hsa.theory_models import (
    HISTORICAL_MODELS,
    THEORY_MODELS,
    mu_from_zeta,
    restricted_kappa_mapping,
    validate_theory_run_design,
)


def _spec(**updates):
    base = {
        "activity_mapping": "real_marginal_cost_proxy",
        "marginal_cost_loading": 1.0,
    }
    base.update(updates)
    return base


def test_namespaces_and_nesting_are_explicit() -> None:
    assert set(HISTORICAL_MODELS) == {"ces", "hsa_steady", "hsa_dynamic", "hsa_const_theta", "hsa_full"}
    assert set(THEORY_MODELS) == {"hsa_f0", "hsa_u", "hsa_r1", "hsa_r2", "hsa_r3"}
    assert THEORY_MODELS["hsa_f0"].moving_reference is False
    assert all(THEORY_MODELS[name].moving_reference for name in ("hsa_u", "hsa_r1", "hsa_r2", "hsa_r3"))
    assert THEORY_MODELS["hsa_r2"].gamma_restriction == "gamma <= 0"
    assert THEORY_MODELS["hsa_r3"].gamma_restriction == "gamma = 0"


def test_cross_restriction_matches_production_kappa_scaling() -> None:
    theta0, zeta0, b_x = 0.2, 6.0, 0.75
    mapping = restricted_kappa_mapping(theta0, zeta0, marginal_cost_loading=b_x)
    assert mapping["kappa_N_empirical"] == pytest.approx(b_x * zeta0 * theta0 / 100.0)
    assert mapping["d_kappa_d_logN"] == pytest.approx(10.0 * mapping["kappa_N_empirical"])
    assert mapping["kappa_N_internal"] == pytest.approx(KAPPA_SCALE * mapping["kappa_N_empirical"])
    x, nbar = 3.2, -0.4
    stored_equation_term = mapping["kappa_N_empirical"] * nbar * x
    sampler_equation_term = mapping["kappa_N_internal"] * (nbar * x / KAPPA_SCALE)
    assert sampler_equation_term == pytest.approx(stored_equation_term)
    assert mu_from_zeta(zeta0) == pytest.approx(1.2)


def test_truncated_gaussian_coefficient_update_preserves_path_positivity() -> None:
    rng = np.random.default_rng(123)
    x = np.column_stack([np.ones(24), np.linspace(-2.0, 2.0, 24)])
    y = np.linspace(-0.5, 0.5, 24)
    n_extremes = (-1.5, 2.0)
    constraints = np.array([[1.0, n] for n in n_extremes])
    beta = np.array([0.1, 0.0])
    draws = []
    for _ in range(100):
        beta = _sample_beta_gaussian_linear_constraints(
            y,
            x,
            sigma2=0.5,
            prior_mean=np.zeros(2),
            prior_var=np.ones(2),
            rng=rng,
            constraint_matrix=constraints,
            lower_bounds=np.zeros(2),
            initial=beta,
            sweeps=2,
        )
        draws.append(beta.copy())
    draws_array = np.asarray(draws)
    assert np.all(draws_array @ constraints.T >= -1e-10)
    assert np.std(draws_array[:, 0]) > 0.01


def test_fixed_theory_constants_and_non_applicable_pg_fields_do_not_fail_diagnostics() -> None:
    idata = az.from_dict(
        {"posterior": {"mu0": np.full((2, 100), 1.2), "alpha": np.arange(200).reshape(2, 100)}}
    )
    summary = compute_diagnostics(idata)
    assert "mu0" not in set(summary["parameter"])
    extracted = _extract_draws_from_result(
        {"pg_diagnostics": {"n_particles": 0, "ess_mean": np.full(20, np.nan)}}
    )
    assert "pg_ess_mean" not in extracted


def test_production_sampling_expands_slow_moving_reference_chains() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_model_config(root / "configs" / "models.yaml")
    sampling = config["theory_defaults"]["sampling_by_model"]
    assert sampling["hsa_f0"]["n_iter"] == 12000
    assert sampling["hsa_u"]["n_iter"] >= 120000
    assert sampling["hsa_r1"]["chains"] >= 4
    assert sampling["hsa_r2"]["chains"] >= 4


def test_restricted_gap_proxy_needs_explicit_bx() -> None:
    with pytest.raises(ValueError, match="marginal_cost_loading"):
        validate_theory_run_design(
            "hsa_r1", {"activity_mapping": "unemployment_gap"},
            zeta0=6.0, inflation_observation="qoq",
        )


def test_quarterly_restriction_is_not_put_on_direct_yoy_regression() -> None:
    with pytest.raises(ValueError, match="4Q-YoY"):
        validate_theory_run_design(
            "hsa_r1", _spec(), zeta0=6.0, inflation_observation="yoy_4q",
        )


def test_artifact_signature_detects_stale_field() -> None:
    metadata = stamp_artifact_metadata(
        {
            "code_revision": "abc",
            "estimation_revision": "rev",
            "model": "hsa_r1",
            "model_hierarchy": "R1",
            "model_definition": {},
            "restriction_taxonomy": {},
            "exact_restrictions": [],
            "data_transformation": {},
            "inflation_observation": "qoq",
            "structural_frequency": "quarterly",
            "sample_start": "2000Q1",
            "sample_end": "2010Q4",
            "n_obs": 44,
            "competition_proxy": "N",
            "activity_proxy": "mc",
            "expectation_series": "Epi",
            "expectation_horizon": "one_year",
            "expectation_information_date": "vintage",
        }
    )
    validate_artifact_metadata(metadata, metadata)
    stale = dict(metadata)
    stale["inflation_observation"] = "yoy_4q"
    with pytest.raises(StaleArtifactError, match="STALE / HISTORICAL"):
        validate_artifact_metadata(stale, stale)


def test_historical_and_restriction_reports_are_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    historical = (root / "report" / "nkpc_hsa_report.tex").read_text(encoding="utf-8")
    restriction = (root / "report" / "nkpc_hsa_restriction_report.tex").read_text(encoding="utf-8")
    historical_builder = (root / "scripts" / "build_report.py").read_text(encoding="utf-8")
    production_driver = (root / "scripts" / "run_restriction_production.py").read_text(encoding="utf-8")
    assert "tables/theory" not in historical
    assert "19_build_theory_report.py" not in historical_builder
    assert "tables/theory/current_design.tex" in restriction
    assert "tables/annual_q4" not in restriction
    assert "admissibility_diagnostics.tex" in (
        root / "src" / "nkpc_hsa" / "reporting" / "theory_report.py"
    ).read_text(encoding="utf-8")
    assert "report_builder_revision" in (
        root / "scripts" / "build_restriction_report.py"
    ).read_text(encoding="utf-8")
    assert 'add_argument("--quick"' not in production_driver
    assert "10_estimate_theory_models.py" in production_driver
    assert "11_run_theory_diagnostics.py" in production_driver
    assert "build_restriction_report.py" in production_driver
