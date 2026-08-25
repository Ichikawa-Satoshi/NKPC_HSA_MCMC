import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

from nkpc_hsa.config import load_yaml
from tests.active_firm_stock_bds_bed.functions import ThetaCell
from tests.gustavo_state_capitaliq_cycle.functions import QoqFit,_ar1_transform,ar2_coefficients,build_qoq_design,draw_bridge_path,load_nkpc_cells,load_oil_controls,qoq_pointwise_loglik
from tests.gustavo_state_capitaliq_cycle.dynamic_functions import centered_states,dynamic_design


def test_ar2_parameterization_is_stationary():
    phi1,phi2=ar2_coefficients(.8,12)
    assert np.max(np.abs(np.roots([1,-phi1,-phi2])))<1


def test_bridge_hits_both_annual_anchors_exactly():
    periods=pd.period_range("2000Q4","2002Q4",freq="Q")
    anchors=pd.Series([0.,1.,-.5],index=pd.PeriodIndex(["2000Q4","2001Q4","2002Q4"],freq="Q"))
    path=draw_bridge_path(np.random.default_rng(7),periods,anchors,.2)
    for period,value in anchors.items():assert path[periods.get_loc(period)]==value


def test_bridge_is_reproducible_for_a_fixed_seed():
    periods=pd.period_range("2000Q4","2001Q4",freq="Q")
    anchors=pd.Series([0.,1.],index=pd.PeriodIndex(["2000Q4","2001Q4"],freq="Q"))
    a=draw_bridge_path(np.random.default_rng(11),periods,anchors,.2)
    b=draw_bridge_path(np.random.default_rng(11),periods,anchors,.2)
    assert np.allclose(a,b)


def test_ar1_transform_is_identity_at_zero_persistence():
    y=np.array([1.,2.,4.]);X=np.column_stack([np.ones(3),np.arange(3.)])
    yt,Xt=_ar1_transform(y,X,0.)
    assert np.allclose(yt,y) and np.allclose(Xt,X)


def test_ar1_transform_uses_stationary_initial_observation():
    y=np.array([2.,3.]);X=np.ones((2,1));rho=.6;yt,Xt=_ar1_transform(y,X,rho)
    assert np.allclose(yt,[1.6,1.8])
    assert np.allclose(Xt[:,0],[.8,.4])


def test_combined_design_centers_slow_state_and_pairs_channels():
    periods=pd.period_range("2000Q1",periods=3,freq="Q")
    cell=ThetaCell("test",periods,np.zeros(3),np.array([1.,2.,3.]),np.array([4.,5.,6.]),np.array([2.,3.,4.]),1.,1.)
    hat=np.array([.1,.2,.3]);bar=np.array([1.,2.,4.]);X,centered=build_qoq_design(cell,hat,bar)
    assert X.shape==(3,6)
    assert np.isclose(centered.mean(),0.)
    assert np.allclose(X[:,4],centered*cell.x)
    assert np.allclose(X[:,5],-hat)


def test_pointwise_loglik_uses_original_scale_for_stationary_ar1():
    periods=pd.period_range("2000Q1",periods=3,freq="Q");cell=ThetaCell("test",periods,np.array([1.,2.,3.]),np.zeros(3),np.zeros(3),np.zeros(3),1.,1.);names=("intercept","alpha_b","alpha_f","kappa_0","theta_CIQ");draws=np.zeros((1,1,5));zeros=np.zeros((1,1,3));fit=QoqFit("test","test","persistent_ar1",tuple(map(str,periods)),names,draws,np.ones((1,1)),np.full((1,1),.5),zeros,zeros,np.zeros((1,1),int),np.zeros((1,1),int),dict.fromkeys(names,0.),dict.fromkeys(names,1.),{})
    actual=qoq_pointwise_loglik(cell,fit)[0,0];expected=np.r_[norm.logpdf(1.,0,1/np.sqrt(.75)),norm.logpdf(np.array([2.,3.]),.5*np.array([1.,2.]),1.)]
    assert np.allclose(actual,expected)


def test_hsa_dynamic_design_is_nested_in_free_dynamic_design():
    periods=pd.period_range("2000Q1",periods=4,freq="Q");cell=ThetaCell("test",periods,np.zeros(4),np.arange(4.),np.arange(4.)+.5,np.array([1.,-1.,2.,-2.]),1.,1.);bar=np.array([1.,2.,4.,3.]);hat=np.array([.2,-.1,.3,-.2]);lam=1.7;theta=.4;gamma=-.2;common=np.array([.1,.3,.6,.8]);restricted=np.r_[common,theta,gamma];free=np.r_[common,lam*theta,.5*lam*gamma,theta,gamma]
    assert np.allclose(dynamic_design("hsa_restricted_dynamic",cell,hat,bar,lam)@restricted,dynamic_design("free_dynamic",cell,hat,bar)@free)
    barc,q2=centered_states(bar);assert np.isclose(barc.mean(),0.) and np.isclose(q2.mean(),0.)


def test_core_cpi_qoq_cell_uses_matched_one_quarter_ahead_expectation():
    bundle=Path(__file__).resolve().parent
    config=load_yaml(bundle/"config.yaml");extension=load_yaml(bundle/"core_cpi_config.yaml")
    price=extension["price"];config["data"]["prices"]={price["name"]:{k:price[k] for k in ("inflation","inflation_lag","expectation")}}
    cells=load_nkpc_cells(config);cell=cells["core_cpi_negative_unemployment_gap"]
    assert len(cell.periods)==83 and str(cell.periods[0])=="1993Q2" and str(cell.periods[-1])=="2013Q4"
    assert np.isfinite(cell.pi).all() and np.isfinite(cell.epi).all()
    assert not np.allclose(cell.pi,cell.epi)


def test_oil_controls_are_current_and_lagged_annualized_qoq_changes():
    periods=pd.period_range("1993Q2","2013Q4",freq="Q")
    controls,meta=load_oil_controls(periods)
    assert controls.shape==(83,2) and np.isfinite(controls).all()
    assert meta["transformation"]=="400_log_quarterly_difference"
    assert np.allclose(controls[1:,1],controls[:-1,0])


def test_oil_control_design_preserves_hsa_nesting():
    periods=pd.period_range("2000Q1",periods=4,freq="Q")
    cell=ThetaCell("test",periods,np.zeros(4),np.arange(4.),np.arange(4.)+.5,np.array([1.,-1.,2.,-2.]),1.,1.)
    bar=np.array([1.,2.,4.,3.]);hat=np.array([.2,-.1,.3,-.2]);oil=np.column_stack([np.arange(4.),np.arange(4.)-.5]);lam=1.7;theta=.4;gamma=-.2
    common=np.array([.1,.3,.6,.8,.02,-.01]);restricted=np.r_[common,theta,gamma];free=np.r_[common,lam*theta,.5*lam*gamma,theta,gamma]
    assert np.allclose(dynamic_design("hsa_restricted_dynamic",cell,hat,bar,lam,oil)@restricted,dynamic_design("free_dynamic",cell,hat,bar,controls=oil)@free)
