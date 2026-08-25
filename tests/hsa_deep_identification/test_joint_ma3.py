"""Numerical checks for the local exact-N MA(3) joint sampler."""
from __future__ import annotations

import numpy as np

from nkpc_hsa.error_robustness.ma_error import MAWeighting
from tests.hsa_deep_identification.joint_ma3 import _state_draw
from tests.hsa_nested_validation.functions import _cycle_coefficients, _cycle_unit_cov


def _dense_conditional(q, drift, y, constant, loading, psi, sigma2, damping, period, tau2, omega):
    T = len(q); vb = omega*tau2; vh = (1-omega)*tau2
    precision = np.zeros((T,T)); rhs = np.zeros(T)
    precision[:2,:2] += np.linalg.inv(vh*_cycle_unit_cov(damping,period))
    phi1,phi2 = _cycle_coefficients(damping,period)
    for t in range(2,T):
        row=np.zeros(T); row[t]=1; row[t-1]=-phi1; row[t-2]=-phi2
        precision += np.outer(row,row)/vh
    H=np.zeros((T-1,T))
    for t in range(1,T): H[t-1,t]=1; H[t-1,t-1]=-1
    slow=np.diff(q)-drift[1:]
    precision += H.T@H/vb; rhs += H.T@slow/vb
    W=MAWeighting(psi,T).solve(np.eye(T))/sigma2
    G=np.diag(loading); target=y-constant
    precision += G.T@W@G; rhs += G.T@W@target
    covariance=np.linalg.inv(precision); mean=covariance@rhs
    return mean,covariance


def test_state_ffbs_matches_dense_gaussian_conditional():
    rng=np.random.default_rng(90210); T=14
    q=np.cumsum(rng.normal(.02,.15,T)); drift=np.r_[0,np.tile([.01,.03,.04,.02],4)[:T-1]]
    y=rng.normal(size=T); constant=rng.normal(size=T); loading=np.linspace(-.3,.2,T)
    psi=np.array([.35,.15,.08]); sigma2=.7; damping=.72; period=11.; tau2=.09; omega=.22
    mean,cov=_dense_conditional(q,drift,y,constant,loading,psi,sigma2,damping,period,tau2,omega)
    draws=np.array([_state_draw(rng,q,drift,y,constant,loading,psi,sigma2,damping,period,tau2,omega) for _ in range(6000)])
    mc_se=np.sqrt(np.diag(cov)/len(draws))
    assert np.max(np.abs(draws.mean(0)-mean)/mc_se) < 4.5
    ratio=np.diag(np.cov(draws,rowvar=False))/np.diag(cov)
    assert np.all((ratio>.92)&(ratio<1.08))


def test_exact_identity_is_algebraic():
    rng=np.random.default_rng(7); q=rng.normal(size=20); h=rng.normal(size=20); bar=q-h
    assert np.max(np.abs(q-bar-h)) < 1e-14
