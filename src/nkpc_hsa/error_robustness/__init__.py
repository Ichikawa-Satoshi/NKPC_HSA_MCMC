"""Error-structure robustness: the NKPC disturbance as an MA(3) process.

Why this package exists
-----------------------
Inflation in this project is year-over-year: ``pi_t = 100 * (log P_t - log P_{t-4})``
(``dataprep/func_data_build.py``, ``log_yoy`` / ``pct_yoy``). Four-quarter
overlap means that even an i.i.d. quarterly structural shock shows up in the
estimated equation as a moving average of order three. Every production sampler
nonetheless treats the inflation disturbance as i.i.d., and the residuals say
that is wrong: across all ten baseline CES cells the residual autocorrelation is
positive and significant at lags 1-3 and turns negative at lag 4, with
Ljung-Box(4) between 24 and 56 against a 5% critical value of 9.49.

This package is **purely additive**. It does not modify, monkey-patch, or import
mutable state from the production samplers; it carries its own copies of the
blocks it has to change. Production results stay bit-for-bit reproducible from
the untouched code in ``nkpc_hsa.gibbs``.

What changes relative to the baseline
-------------------------------------
Exactly one thing. The baseline inflation disturbance

    e_t = lambda_ez * zeta_t + v_t,        v_t ~ iid N(0, sigma_v^2)

becomes

    e_t = lambda_ez * zeta_t + xi_t,       xi_t = psi(L) v_t
    psi(L) = 1 + psi_1 L + psi_2 L^2 + psi_3 L^3

with ``psi`` restricted to the invertible region. The cross-equation loading on
the activity innovation stays contemporaneous, so ``psi = 0`` reproduces the
production model exactly -- that identity is what
``tests/test_error_robustness_ma3.py`` pins.

Why not AR(1), and why not psi = (1,1,1)
----------------------------------------
Both were fitted to the plug-in NKPC residuals and both lose. MA(3) beats AR(1)
on BIC in all ten baseline cells (main cell: 3.3 against 18.1). AR(1) matches
the lag-1 autocorrelation and then collapses -- 0.14 and 0.05 at lags 2 and 3
against 0.24 and 0.22 in the data -- because one parameter cannot carry three
lags. The exact equal-weight overlap restriction ``psi = (1,1,1)`` is rejected
even harder (BIC 124.6 in the main cell, worse than i.i.d. at 32.0): it implies
a lag-1 autocorrelation of 0.75 where the data show 0.38. ``scripts/error_robustness``
regenerates both comparisons.
"""

from __future__ import annotations

__all__ = ["ma_error"]
