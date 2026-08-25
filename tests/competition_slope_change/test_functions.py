from __future__ import annotations

import numpy as np
import pandas as pd

from tests.competition_slope_change.functions import (
    CompetitionStateFit,
    NKPCFit,
    _timing_indices,
    conditional_omega_likelihood,
    economic_quantities,
)
from tests.hsa_nested_validation.functions import CellData


def test_timing_alignment():
    now, current = _timing_indices(5, "current")
    np.testing.assert_array_equal(now, current)
    now, lag = _timing_indices(5, "lag1")
    np.testing.assert_array_equal(now, [1, 2, 3, 4])
    np.testing.assert_array_equal(lag, [0, 1, 2, 3])
    now, lead = _timing_indices(5, "lead1")
    np.testing.assert_array_equal(now, [0, 1, 2, 3])
    np.testing.assert_array_equal(lead, [1, 2, 3, 4])


def test_economic_quantity_is_delta_times_state_change():
    periods=pd.period_range("2000Q1",periods=4,freq="Q")
    cell=CellData("x","x","ppi","inverse_markup",periods,np.zeros(4),np.zeros(4),np.zeros(4),
                  np.zeros(4),np.arange(4),1.0,2.0,1.0)
    draws=np.zeros((2,3,5));draws[:,:,3]=0.5;draws[:,:,4]=2.0
    bar=np.broadcast_to(np.arange(4,dtype=float),(2,3,4)).copy()
    kappa=draws[:,:,3,None]+draws[:,:,4,None]*bar
    fit=NKPCFit("x","slope_only","none",tuple(map(str,periods)),
                ("intercept","alpha_b","alpha_f","kappa_0","delta"),draws,
                np.ones((2,3)),np.zeros((2,3,3)),bar,np.zeros_like(bar),kappa,
                {"intercept":0,"alpha_b":0,"alpha_f":0,"kappa_0":0,"delta":0},
                {"intercept":1,"alpha_b":1,"alpha_f":1,"kappa_0":1,"delta":1},{})
    config={"sample":{"endpoint_windows":{"test":["2000Q1","2000Q4"]},
                      "counterfactual_reference_start":"2000Q1","counterfactual_reference_end":"2000Q2"}}
    rows=economic_quantities(fit,cell,config)
    delta_k=next(row for row in rows if row["quantity"]=="delta_kappa_comp")
    assert delta_k["mean"]==6.0
    inflation=next(row for row in rows if row["quantity"]=="inflation_effect_at_one_sd_x")
    assert inflation["mean"]==12.0


def test_conditional_omega_slice_is_finite_and_normalized():
    shape=(2,5,8);q=np.zeros(shape);h=np.zeros(shape)
    for t in range(8):
        q[:,:,t]=0.1*t;h[:,:,t]=0.02*np.sin(t)
    fit=CompetitionStateFit("ar1","test",tuple(map(str,pd.period_range("2000Q1",periods=8,freq="Q"))),
        q,q-h,h,np.full((2,5),.2),np.full((2,5),.3),np.full((2,5),.6),
        np.full((2,5),np.nan),np.zeros(8),{})
    profile=conditional_omega_likelihood(fit,np.linspace(.05,.95,19))
    assert np.isfinite(profile.relative_conditional_loglik).all()
    assert abs(profile.relative_conditional_loglik.max())<1e-12
