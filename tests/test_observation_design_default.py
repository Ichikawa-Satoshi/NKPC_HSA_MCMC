"""The declared observation design must agree everywhere it is defaulted.

The firm count is annual, so the project's main design is mixed-frequency
(``annual_q4``). That choice is declared in ``configs/models.yaml`` and has to be
honoured by the library default and by the estimation driver, or a caller that
omits the argument silently estimates the other design.

The tests also pin the distinction that makes this safe: code that *reads* a saved
run's metadata must keep falling back to ``quarterly_interpolated``, because runs
written before the field existed were interpolated. Changing those readers would
misclassify history.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from nkpc_hsa.dataprep.competition import (
    DEFAULT_COMPETITION_MEASUREMENT,
    normalize_competition_measurement,
)

ROOT = Path(__file__).resolve().parents[1]


def _config_frequency() -> str:
    config = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    return str(config["defaults"]["competition_measurement"]["frequency"])


def test_config_declares_the_mixed_frequency_design():
    assert _config_frequency() == "annual_q4"


def test_library_default_matches_the_config():
    assert DEFAULT_COMPETITION_MEASUREMENT["frequency"] == _config_frequency()
    assert normalize_competition_measurement(None)["frequency"] == _config_frequency()
    assert normalize_competition_measurement({})["frequency"] == _config_frequency()


def test_explicit_argument_still_wins():
    spec = normalize_competition_measurement({"frequency": "quarterly_interpolated"})
    assert spec["frequency"] == "quarterly_interpolated"


def test_run_metadata_default_matches_the_config():
    from nkpc_hsa.inference.wrappers import RunMetadata

    meta = RunMetadata(
        model="hsa_steady", data_spec="x", prior_spec="baseline", run_id="r",
        n_iter=1, burn=0, thin=1, chains=1, seed=0, n_transform="log100_centered10",
    )
    assert meta.competition_measurement_frequency == _config_frequency()


def test_estimation_driver_defaults_to_the_config():
    """scripts/13 must resolve its CLI default from the config, not a literal."""
    src = (ROOT / "scripts" / "13_estimate_cpi_ppi_report.py").read_text(encoding="utf-8")
    assert 'default=None' in src, "the --competition-frequency default must be resolved at runtime"
    assert 'competition_measurement' in src and 'get("frequency"' in src


def test_run_directory_names_always_carry_the_design():
    """A conditional suffix would invert meaning now that the default changed."""
    from nkpc_hsa.inference.wrappers import _default_run_dir

    for freq in ("annual_q4", "quarterly_interpolated"):
        name = _default_run_dir("hsa_steady", "spec", "baseline", "unrestricted", "rid",
                                competition_frequency=freq).name
        assert freq in name, f"{freq} missing from {name!r}"


def test_metadata_readers_still_fall_back_to_interpolated():
    """Runs saved before the field existed were interpolated; readers must say so."""
    # reporting/tables.py used to belong here. Every one of its metadata-reading
    # functions fed the superseded main.tex report and was removed with it in
    # August 2026; the module no longer touches run metadata at all. These two are
    # the readers that remain.
    readers = [
        ROOT / "scripts" / "12_build_cpi_ppi_report.py",
        ROOT / "src" / "nkpc_hsa" / "reporting" / "data_model_report.py",
    ]
    pattern = re.compile(
        r'get\(\s*["\']competition_measurement_frequency["\']\s*,\s*["\']quarterly_interpolated["\']'
    )
    for path in readers:
        assert path.exists(), f"{path} moved; this test's path list is stale"
        assert pattern.search(path.read_text(encoding="utf-8")), (
            f"{path.name} no longer falls back to quarterly_interpolated when the field is absent"
        )


def test_alpha_zero_runs_are_excluded_from_the_report_run_set():
    """The purely forward-looking restriction is a different estimating equation.

    It must not be selectable as an ``hsa_steady`` cell by the report builder, which
    keeps the newest run per (model, data spec, prior) key and would otherwise pick a
    restricted run whose run-id happens to sort later.
    """
    import json

    runs = ROOT / "results" / "runs"
    if not runs.exists():
        return
    restricted = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in runs.glob("*alpha_zero*/metadata.json")
    ]
    for meta in restricted:
        assert meta["constraint_spec"] != "unrestricted", (
            f"{meta.get('run_id')} would enter the report run-set"
        )
        assert meta.get("no_inertia") is True


def test_no_inertia_is_rejected_for_models_that_do_not_implement_it():
    import numpy as np
    import pytest

    from nkpc_hsa.inference.wrappers import run_model

    data = {k: np.zeros(8) for k in ["pi", "pi_prev", "pi_expect", "x", "x_prev"]}
    data["N"] = np.ones(8)
    with pytest.raises(ValueError, match="only implemented for hsa_steady"):
        run_model("hsa_full", data=data, n_iter=4, burn=2, thin=1, chains=1,
                  no_inertia=True, save=False)
