"""Deterministic structural checks for the nested-validation implementation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402
from nkpc_hsa.config import load_yaml  # noqa:E402
from tests.hsa_nested_validation.functions import (  # noqa:E402
    ModelSpec, _band2_cholesky, _cycle_coefficients, _design, _state_precision,
    build_model_specs, load_experiment,
)

BUNDLE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mock", "quick", "full"), default="mock")
    args = parser.parse_args()
    cfg = load_yaml(BUNDLE / "config.yaml")
    experiment = load_experiment(cfg)
    cell = experiment.cells["ppi_inverse_markup"]
    q = experiment.mean_q(cell)
    # A non-degenerate deterministic cycle is sufficient to check every column.
    h = 0.3 * np.sin(np.linspace(0.0, 8.0, cell.n_periods))
    bar = q - h
    base = np.column_stack((np.ones(cell.n_periods), cell.pi_lag, cell.epi, cell.x))

    primary_specs, specs = build_model_specs(cfg)
    by_id = {s.model_id: s for s in specs}
    free = _design(by_id["free_static_combined"], cell, q, h)

    beta_base = np.array([0.2, 0.3, 0.4, 0.1])
    delta_s, theta, lam = 0.07, -0.05, 6.0

    # Each fixed-lambda HSA model is the free combined model at delta_s=lambda*theta.
    mu_free = free @ np.r_[beta_base, delta_s, theta]
    b6_spec = ModelSpec("check_hsa", "benchmark", tuple((*by_id["ces"].coefficient_names, "theta")), lam)
    mu_b6 = _design(b6_spec, cell, q, h) @ np.r_[beta_base, theta]
    mu_free_at_b6 = free @ np.r_[beta_base, lam * theta, theta]

    damping, period = 0.8, 12.0
    phi1, phi2 = _cycle_coefficients(damping, period)
    roots = np.linalg.eigvals(np.array([[phi1, phi2], [1.0, 0.0]]))
    diag, off1, off2, _ = _state_precision(q[:20], damping, period, 0.04, 0.2)
    ld, l1, l2 = _band2_cholesky(diag, off1, off2)
    precision = np.diag(diag) + np.diag(off1, 1) + np.diag(off1, -1)
    precision += np.diag(off2, 2) + np.diag(off2, -2)
    factor = np.diag(ld) + np.diag(l1, -1) + np.diag(l2, -2)

    checks = {
        "free_design_reconstruction": float(np.max(np.abs(
            mu_free - np.column_stack((base, bar * cell.x, -h)) @ np.r_[beta_base, delta_s, theta]
        ))),
        "HSA_in_free_combined": float(np.max(np.abs(mu_b6 - mu_free_at_b6))),
        "AR2_root_modulus_error": float(np.max(np.abs(np.abs(roots) - damping))),
        "AR2_precision_factorization": float(np.max(np.abs(precision - factor @ factor.T))),
    }
    manifest_path = BUNDLE / "results" / args.mode / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks["saved_exact_identity"] = float(manifest["gate"]["max_identity_error"])
        checks["model_count"] = len(manifest["models"])
        n_prices = len(cfg["data"]["prices"])
        checks["expected_model_count"] = n_prices * (len(primary_specs) + len(specs))
        checks["joint_only"] = manifest["treatments"] == ["joint_state_split"]
        checks["removed_models_absent"] = not any(
            model_id in key
            for key in manifest["models"]
            for model_id in ("encompassing_two_coordinate", "single_coordinate_free")
        )
        checks["gate_passed"] = bool(manifest["gate"]["passed"])
        sample_path = BUNDLE / "results" / args.mode / "draws" / "joint_state_split" / "ppi_inverse_markup" / "ces.npz"
        with np.load(sample_path, allow_pickle=False) as saved:
            checks["AR2_fields_saved"] = bool(
                "cycle_damping" in saved.files and "cycle_period" in saved.files and "rho" not in saved.files
            )
    tolerance = float(cfg["gates"]["max_exact_identity_error"])
    algebra_ok = all(checks[name] <= tolerance for name in (
        "free_design_reconstruction", "HSA_in_free_combined",
        "AR2_root_modulus_error", "AR2_precision_factorization",
    ))
    identity_ok = checks.get("saved_exact_identity", 0.0) <= tolerance
    count_ok = checks.get("model_count", 24) == checks.get("expected_model_count", 24)
    removed_ok = checks.get("removed_models_absent", True)
    treatment_ok = checks.get("joint_only", True)
    ar2_saved_ok = checks.get("AR2_fields_saved", True)
    print(json.dumps(checks, indent=2))
    if not (algebra_ok and identity_ok and count_ok and removed_ok and treatment_ok and ar2_saved_ok):
        raise SystemExit("Structural validation failed")


if __name__ == "__main__":
    main()
