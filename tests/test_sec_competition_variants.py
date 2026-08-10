"""The SEC competition aggregate must be a declared choice, not a hardcoded one.

Within a SIC3 market the HHI is always a revenue-share HHI; the variants differ
only in how the markets are collapsed to one number per quarter. That choice
moves ``delta``, so it belongs in ``configs/models.yaml`` where a run records
it, and the config and the builder must not drift apart.

The firm-weighted aggregate has to stay the default with an empty spec suffix:
the August 2026 extension grid ran on it, and changing either would orphan those
run directories.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from nkpc_hsa.dataprep.sec_hhi import DEFAULT_SEC_INVERSE_COLUMNS, SEC_INVERSE_HHI_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def _load_extension_driver():
    path = ROOT / "scripts" / "16_estimate_extensions.py"
    spec = importlib.util.spec_from_file_location("extension_driver", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())


def test_every_configured_variant_is_a_column_the_builder_writes(config) -> None:
    variants = config["sec_competition_variants"]["variants"]
    built = set(DEFAULT_SEC_INVERSE_COLUMNS.values())

    for name, block in variants.items():
        assert block["column"] in built, f"{name} names a column 15_build_extension_data.py never writes"


def test_every_source_aggregate_is_validated(config) -> None:
    # A column merged into the model-ready frame but absent from the validator's
    # list would skip the "this really is an inverse HHI" check.
    for source in DEFAULT_SEC_INVERSE_COLUMNS:
        assert source in SEC_INVERSE_HHI_COLUMNS


def test_firm_weighted_stays_the_default_with_an_unchanged_spec_name(config) -> None:
    driver = _load_extension_driver()

    name, block = driver.sec_variant(config, None)

    assert name == "firm_weighted"
    assert block["column"] == "N_SEC_inverse_HHI"
    assert block["spec_suffix"] == ""
    spec = driver._extension_spec({"n_col": "N_Gustavo"}, "unemployment_gap_core", "sec_inverse_hhi", block)
    assert spec["name"] == "unemployment_gap_core__sec_inverse_hhi"


def test_selecting_a_variant_changes_both_the_column_and_the_spec_name(config) -> None:
    driver = _load_extension_driver()

    _, block = driver.sec_variant(config, "revenue_log_weighted")
    spec = driver._extension_spec({"n_col": "N_Gustavo"}, "unemployment_gap_core", "sec_inverse_hhi", block)

    assert spec["n_col"] == "N_SEC_inverse_HHI_logrevw"
    assert spec["name"] == "unemployment_gap_core__sec_inverse_hhi_logrevw"
    assert spec["sample_start"] == "2012Q1"


def test_spec_suffixes_are_unique_so_run_directories_cannot_collide(config) -> None:
    suffixes = [block["spec_suffix"] for block in config["sec_competition_variants"]["variants"].values()]

    assert len(suffixes) == len(set(suffixes))


def test_unknown_variant_is_rejected(config) -> None:
    driver = _load_extension_driver()

    with pytest.raises(ValueError, match="Unknown SEC variant"):
        driver.sec_variant(config, "not_a_variant")


def test_recommended_variant_is_declared_and_real(config) -> None:
    block = config["sec_competition_variants"]

    assert block["recommended"] in block["variants"]
