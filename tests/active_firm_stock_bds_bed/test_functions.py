import numpy as np

from tests.active_firm_stock_bds_bed.functions import (
    ar2_coefficients, detection_indicator, transform_parameters,
)


def test_ar2_roots_are_stationary():
    phi1,phi2=ar2_coefficients(.8,12.0)
    roots=np.roots([1.0,-phi1,-phi2])
    assert np.max(np.abs(roots)) < 1.0


def test_parameter_transform_respects_bounds():
    p=transform_parameters(np.zeros(8),.08,.10,{"damping_bounds":[.2,.95],"period_bounds":[6,24],"bds_error_fixed":.005})
    assert 0 < p["omega"] < 1
    assert .2 < p["damping"] < .95
    assert 6 < p["period"] < 24
    assert p["tau"] > 0 and p["sigma_f"] == .005 and p["sigma_bed"] > 0


def test_detection_requires_all_gates():
    assert detection_indicator(.01,.20,.99,.70,positive=True)
    assert not detection_indicator(-.01,.20,.99,.70,positive=True)
    assert not detection_indicator(.01,.20,.96,.70,positive=True)
    assert not detection_indicator(.01,.20,.99,.90,positive=True)
