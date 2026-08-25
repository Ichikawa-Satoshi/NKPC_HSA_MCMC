from __future__ import annotations

import numpy as np

from nkpc_hsa.config import load_yaml
from tests.mixed_frequency_gustavo_capitaliq.functions import (
    _encode,
    _initial_parameters,
    benchmark_path,
    kalman_filter,
    load_measurement_data,
)


def _objects():
    config = load_yaml("tests/mixed_frequency_gustavo_capitaliq/config.yaml")
    return config, load_measurement_data(config)


def test_measurements_use_only_consecutive_quarter_capital_iq_changes():
    _, data = _objects()
    assert len(data.periods) == 157
    assert np.isfinite(data.gustavo).sum() == 40
    assert all(np.isfinite(values).sum() == 84 for values in data.ciq_growth.values())
    assert np.isclose(data.average_weights.sum(), 1.0)


def test_mechanical_benchmarks_hit_every_exact_q4_anchor():
    _, data = _objects()
    anchors = np.isfinite(data.gustavo)
    for method in ("equal_allocation", "average_allocation"):
        path = benchmark_path(data, method)
        assert np.max(np.abs(path[anchors] - data.gustavo[anchors])) < 1e-12


def test_exact_gustavo_update_conditions_without_adding_anchor_density():
    config, data = _objects()
    z = _encode(_initial_parameters(data, config), config)
    result = kalman_filter(z, data, config)
    anchors = np.isfinite(data.gustavo)
    total = result["filt_mean"][anchors, 0] + result["filt_mean"][anchors, 1]
    assert np.isfinite(result["loglik"])
    assert np.max(np.abs(total - data.gustavo[anchors])) < 1e-8
