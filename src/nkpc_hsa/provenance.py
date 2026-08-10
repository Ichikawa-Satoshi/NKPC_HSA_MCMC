"""Content-addressed provenance and stale-artifact protection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ARTIFACT_SIGNATURE_FIELDS = (
    "code_revision",
    "estimation_revision",
    "model",
    "model_hierarchy",
    "model_definition",
    "restriction_taxonomy",
    "exact_restrictions",
    "data_transformation",
    "inflation_observation",
    "structural_frequency",
    "sample_start",
    "sample_end",
    "n_obs",
    "competition_proxy",
    "activity_proxy",
    "expectation_series",
    "expectation_horizon",
    "expectation_information_date",
)


class StaleArtifactError(RuntimeError):
    pass


def canonical_payload(metadata: Mapping[str, Any], fields: Sequence[str] = ARTIFACT_SIGNATURE_FIELDS) -> dict[str, Any]:
    return {field: metadata.get(field) for field in fields}


def artifact_signature(metadata: Mapping[str, Any], fields: Sequence[str] = ARTIFACT_SIGNATURE_FIELDS) -> str:
    encoded = json.dumps(canonical_payload(metadata, fields), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stamp_artifact_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(metadata)
    out["artifact_signature_fields"] = list(ARTIFACT_SIGNATURE_FIELDS)
    out["artifact_signature"] = artifact_signature(out)
    return out


def validate_artifact_metadata(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    artifact: str | Path = "artifact",
    fail_stale: bool = True,
) -> bool:
    mismatches = {
        field: {"expected": expected.get(field), "actual": actual.get(field)}
        for field in ARTIFACT_SIGNATURE_FIELDS
        if actual.get(field) != expected.get(field)
    }
    actual_signature = actual.get("artifact_signature")
    if actual_signature != artifact_signature(actual):
        mismatches["artifact_signature"] = {
            "expected": artifact_signature(actual),
            "actual": actual_signature,
        }
    if not mismatches:
        return True
    if fail_stale:
        rendered = "; ".join(
            f"{key}: expected={value['expected']!r}, actual={value['actual']!r}"
            for key, value in mismatches.items()
        )
        raise StaleArtifactError(f"STALE / HISTORICAL {artifact}: {rendered}")
    return False
