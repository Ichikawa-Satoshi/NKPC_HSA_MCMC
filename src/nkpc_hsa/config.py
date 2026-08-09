from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from nkpc_hsa.paths import project_path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or project_path("configs", "models.yaml"))


def configured_data_specs(
    config: Mapping[str, Any],
    requested: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return named data specs to run.

    If ``requested`` is omitted, use ``run_data_specs`` from the config. This lets
    the canonical pipeline estimate the three x-variable variants in one pass
    while keeping ``data_specs.default`` available for backward compatibility.
    """
    all_specs = dict(config.get("data_specs", {}) or {})
    names = list(requested or config.get("run_data_specs", []) or ["default"])
    # The estimation window is a property of the study, not of a cell, so it
    # lives in ``defaults`` and is injected here. Injecting rather than reading
    # it at the call site means every script that resolves specs through this
    # function gets the same window, and the window is recorded in each run's
    # saved data_spec.json.
    defaults = dict(config.get("defaults", {}) or {})
    # PyYAML resolves ISO dates to ``datetime.date``. Keep the public data-spec
    # payload JSON-serializable because it is saved verbatim as data_spec.json.
    window = {
        k: str(defaults[k])
        for k in ("sample_start", "sample_end")
        if defaults.get(k) is not None
    }
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in all_specs:
            raise KeyError(f"Unknown data spec {name!r}. Available: {sorted(all_specs)}")
        out[name] = {"name": name, **window, **dict(all_specs[name] or {})}
    return out


def data_spec_label(data_spec: Mapping[str, Any] | str) -> str:
    if isinstance(data_spec, str):
        return data_spec
    return str(data_spec.get("label") or data_spec.get("name") or "")


def coefficient_constraints_from_config(
    defaults: Mapping[str, Any],
    *,
    positive: list[str] | None = None,
    disabled: bool = False,
) -> dict[str, Any]:
    constraints = dict(defaults.get("coefficient_constraints", {}) or {})
    if disabled:
        constraints["enabled"] = False
        return constraints
    extra_positive: list[str] = []
    for item in positive or []:
        extra_positive.extend(part.strip() for part in str(item).split(",") if part.strip())
    if extra_positive:
        constraints["enabled"] = True
        existing = list(constraints.get("positive", []) or [])
        constraints["positive"] = sorted(set(existing).union(extra_positive))
    return constraints
