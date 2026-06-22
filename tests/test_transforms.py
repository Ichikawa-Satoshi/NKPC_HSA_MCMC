from __future__ import annotations

import numpy as np
import pytest

from nkpc_hsa.data.transforms import transform_competition_series


def test_n_transform_log100() -> None:
    values = np.array([1.0, np.e])
    out = transform_competition_series(values, transform="log100")
    assert np.allclose(out, [0.0, 100.0])


def test_n_transform_rejects_nonpositive_levels() -> None:
    with pytest.raises(ValueError):
        transform_competition_series(np.array([1.0, 0.0]), transform="log100")
