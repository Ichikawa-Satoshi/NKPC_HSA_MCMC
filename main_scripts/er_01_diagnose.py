"""Evidence that the i.i.d. NKPC disturbance is wrong, and that MA(3) is the fix.

Three self-contained pieces, all written to ``results/error_robustness/``:

1. **Residual autocorrelation.** Plug-in NKPC residuals from every baseline CES
   run, their autocorrelations at lags 1-8, and Ljung-Box(4). This is the fact
   the section opens with: the disturbance is not serially independent, and the
   shape (positive at lags 1-3, sign change at lag 4) is the signature of
   four-quarter overlap rather than of generic persistence.

2. **Order selection.** Exact stationary Gaussian likelihoods for i.i.d.,
   AR(1), AR(2), MA(3), the equal-weight overlap restriction psi = (1,1,1), and
   ARMA(1,3), compared on BIC. This is what rules out AR(1) as the alternative
   specification: one parameter can match the lag-1 autocorrelation and nothing
   past it.

3. **Bias Monte Carlo.** A recursive NKPC design in which ``pi_{t-1}`` genuinely
   carries past disturbances, estimated both ways. This is the only piece that
   speaks to *why* the misspecification matters rather than *that* it exists.

Plug-in caveat, which the report must carry: pieces 1 and 2 evaluate residuals
at posterior-mean coefficients from the i.i.d. runs, so they are indicative
rather than a joint posterior comparison. The order conclusion is not close
enough for that to matter; the magnitudes are.

    python main_scripts/er_01_diagnose.py                # all three
    python main_scripts/er_01_diagnose.py --skip-monte-carlo
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz, cho_factor, cho_solve
from scipy.optimize import minimize

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.error_robustness.ma_error import is_invertible
from nkpc_hsa.inference.wrappers import model_sample_index

OUT = RESULTS_DIR / "error_robustness"
MAX_LAG = 8
LB_LAG = 4


# ------------------------------------------------------------------
# Exact stationary Gaussian likelihood for a given autocovariance
# ------------------------------------------------------------------

def _gaussian_nll(gamma: np.ndarray, e: np.ndarray) -> float:
    n = e.size
    padded = np.r_[gamma, np.zeros(max(0, n - gamma.size))][:n]
    try:
        factor = cho_factor(toeplitz(padded), lower=True)
    except Exception:
        return 1e10
    log_det = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    return 0.5 * (n * np.log(2.0 * np.pi) + log_det + float(e @ cho_solve(factor, e)))


def _ar_autocov(phi, sigma2: float, n: int) -> np.ndarray:
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = phi.size
    horizon = max(n, 600)
    psi = np.zeros(horizon)
    psi[0] = 1.0
    for i in range(1, horizon):
        psi[i] = sum(phi[j] * psi[i - j - 1] for j in range(min(p, i)))
    return np.array([sigma2 * float(np.dot(psi[: horizon - k], psi[k:])) for k in range(n)])


def _ma_autocov(psi, sigma2: float, n: int) -> np.ndarray:
    coef = np.r_[1.0, np.asarray(psi, dtype=float)]
    q = coef.size
    return np.array(
        [sigma2 * float(np.dot(coef[: q - k], coef[k:])) if k < q else 0.0 for k in range(n)]
    )


def _arma13_autocov(phi: float, psi, sigma2: float, n: int) -> np.ndarray:
    horizon = max(n, 600)
    theta = np.r_[1.0, np.asarray(psi, dtype=float)]
    weights = np.zeros(horizon)
    for i in range(horizon):
        weights[i] = (theta[i] if i < theta.size else 0.0) + (phi * weights[i - 1] if i else 0.0)
    return np.array([sigma2 * float(np.dot(weights[: horizon - k], weights[k:])) for k in range(n)])


def _stationary_ar(phi) -> bool:
    return bool(np.all(np.abs(np.roots(np.r_[1.0, -np.atleast_1d(phi)])) < 1.0))


def _fit(e: np.ndarray, kind: str) -> dict:
    """Maximum likelihood for one error specification. Returns loglik and BIC."""
    n = e.size
    log_var = float(np.log(np.var(e)))

    if kind == "iid":
        nll = _gaussian_nll(np.r_[np.var(e), np.zeros(n - 1)], e)
        return {"ll": -nll, "bic": 2 * nll + np.log(n), "k": 1, "par": np.array([np.var(e)])}

    objectives = {
        "AR(1)": (lambda th: _gaussian_nll(_ar_autocov([np.tanh(th[0])], np.exp(th[1]), n), e),
                  [0.3, log_var]),
        "AR(2)": (lambda th: _gaussian_nll(_ar_autocov(th[:2], np.exp(th[2]), n), e)
                  if _stationary_ar(th[:2]) else 1e10, [0.3, 0.05, log_var]),
        "MA(3)": (lambda th: _gaussian_nll(_ma_autocov(th[:3], np.exp(th[3]), n), e)
                  if is_invertible(th[:3]) else 1e10, [0.35, 0.15, 0.15, log_var]),
        # psi = (1,1,1) has no free MA parameter, only a variance. It sits
        # exactly on the unit circle (roots -1, +/-i) so it is not invertible;
        # the likelihood is still well defined, which is why it can be scored.
        "MA(3) equal": (lambda th: _gaussian_nll(_ma_autocov([1.0, 1.0, 1.0], np.exp(th[0]), n), e),
                        [log_var - 1.4]),
        "ARMA(1,3)": (lambda th: _gaussian_nll(
            _arma13_autocov(np.tanh(th[0]), th[1:4], np.exp(th[4]), n), e)
            if is_invertible(th[1:4]) else 1e10, [0.2, 0.3, 0.1, 0.1, log_var]),
    }
    if kind not in objectives:
        raise ValueError(f"unknown error specification {kind!r}")
    objective, start = objectives[kind]

    best = None
    for jitter in range(6):
        x0 = np.array(start, dtype=float)
        if jitter:
            x0 = x0 + np.random.default_rng(jitter).normal(0.0, 0.15, x0.size)
        candidate = minimize(objective, x0, method="Nelder-Mead",
                             options=dict(maxiter=40000, maxfev=40000, xatol=1e-9, fatol=1e-9))
        if best is None or candidate.fun < best.fun:
            best = candidate
    k = len(start)
    return {"ll": -best.fun, "bic": 2 * best.fun + k * np.log(n), "k": k, "par": best.x}


# ------------------------------------------------------------------
# 1 + 2. Residual diagnostics and order selection on the baseline CES runs
# ------------------------------------------------------------------

SPECS = ["iid", "AR(1)", "AR(2)", "MA(3)", "MA(3) equal", "ARMA(1,3)"]


def residual_evidence(runs_glob: str, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import arviz as az

    acf_rows, bic_rows = [], []
    for run in sorted(glob.glob(runs_glob)):
        posterior = az.from_netcdf(Path(run) / "posterior.nc").posterior
        spec = json.loads((Path(run) / "data_spec.json").read_text())
        cols = [spec["pi_col"], spec["pi_prev_col"], spec["pi_expect_col"], spec["x_col"]]
        sample_index = model_sample_index(data, spec)
        if sample_index is None:
            raise ValueError(f"Could not reconstruct the sample for {spec.get('name', run)!r}.")
        frame = data.loc[sample_index, cols + [spec["n_col"]]]

        alpha = float(posterior["alpha"].mean())
        kappa = float(posterior["kappa"].mean())
        pi, pi_prev, pi_expect, x = (frame[c].to_numpy() for c in cols)
        resid = pi - alpha * pi_prev - (1.0 - alpha) * pi_expect - kappa * x
        resid = resid - resid.mean()
        n = resid.size

        acf = [float(np.corrcoef(resid[k:], resid[:-k])[0, 1]) for k in range(1, MAX_LAG + 1)]
        ljung = n * (n + 2) * sum(acf[k - 1] ** 2 / (n - k) for k in range(1, LB_LAG + 1))
        acf_rows.append({
            "data_spec": spec["name"], "n_obs": n, "alpha": round(alpha, 4),
            **{f"acf_lag{k}": round(acf[k - 1], 4) for k in range(1, MAX_LAG + 1)},
            "ljung_box_4": round(float(ljung), 2),
            "acf_se": round(float(2.0 / np.sqrt(n)), 4),
        })

        fits = {kind: _fit(resid, kind) for kind in SPECS}
        row = {"data_spec": spec["name"], "n_obs": n}
        row.update({f"bic_{k}": round(v["bic"], 2) for k, v in fits.items()})
        row["best"] = min(fits, key=lambda k: fits[k]["bic"])
        row["psi_ma3"] = np.round(fits["MA(3)"]["par"][:3], 4).tolist()
        row["rho_ar1"] = round(float(np.tanh(fits["AR(1)"]["par"][0])), 4)
        bic_rows.append(row)

    return pd.DataFrame(acf_rows), pd.DataFrame(bic_rows)


# ------------------------------------------------------------------
# 3. Bias Monte Carlo on a recursive NKPC design
# ------------------------------------------------------------------

def bias_monte_carlo(n_reps: int, n_obs: int = 124) -> pd.DataFrame:
    from nkpc_hsa.error_robustness.ces_ma3 import func_nkpc_ces_ma3
    from nkpc_hsa.gibbs.gibbs_ces import func_nkpc_ces

    true_alpha, true_kappa, sigma_v = 0.60, 0.15, 0.8
    true_psi = np.array([0.45, 0.30, 0.55])
    priors = dict(
        mu_alpha=0.5, sigma_alpha=0.2, mu_kappa=0.1, sigma_kappa=0.2,
        mu_phi_1=0.7, sigma_phi_1=0.2, mu_lambda=0.0, sigma_lambda=0.5,
        a_e=2.0, b_e=2.0, a_z=0.001, b_z=0.001,
    )

    rows = []
    for rep in range(n_reps):
        rng = np.random.default_rng(1000 + rep)
        x = np.zeros(n_obs + 1)
        for t in range(1, n_obs + 1):
            x[t] = 0.8 * x[t - 1] + rng.standard_normal()
        x_t, x_prev = x[1:], x[:-1]
        pi_expect = 2.0 + 0.3 * rng.standard_normal(n_obs)
        v = rng.standard_normal(n_obs + 3) * sigma_v
        xi = np.array([
            v[t + 3] + true_psi[0] * v[t + 2] + true_psi[1] * v[t + 1] + true_psi[2] * v[t]
            for t in range(n_obs)
        ])
        # Recursive, so pi_{t-1} genuinely carries v_{t-1..t-4}: this is the
        # channel that biases alpha and it is absent from a design that draws
        # the lagged regressor independently.
        pi = np.zeros(n_obs + 1)
        pi[0] = 2.0
        for t in range(n_obs):
            pi[t + 1] = (
                true_alpha * pi[t] + (1 - true_alpha) * pi_expect[t] + true_kappa * x_t[t] + xi[t]
            )
        args = (pi[1:], pi[:-1], pi_expect, x_t, x_prev, 1500, 5000)
        iid = func_nkpc_ces(*args, priors=priors, opts=dict(seed=77 + rep))
        ma3 = func_nkpc_ces_ma3(*args, priors=priors, opts=dict(seed=77 + rep, ma_order=3))
        rows.append({
            "rep": rep,
            "alpha_iid": float(iid["alpha"]["draws"].mean()),
            "alpha_ma3": float(ma3["alpha"]["draws"].mean()),
            "kappa_iid": float(iid["kappa"]["draws"].mean()),
            "kappa_ma3": float(ma3["kappa"]["draws"].mean()),
            "alpha_true": true_alpha,
            "kappa_true": true_kappa,
        })
        print(f"  replication {rep + 1}/{n_reps}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_DIR / "processed" / "model_ready.csv"))
    parser.add_argument(
        "--runs-glob",
        default=str(RESULTS_DIR / "runs" / "ces_*_baseline_quarterly_interpolated"),
        help="Production CES runs to take plug-in coefficients from. Read only.",
    )
    parser.add_argument("--monte-carlo-reps", type=int, default=12)
    parser.add_argument("--skip-monte-carlo", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, parse_dates=["DATE"]).set_index("DATE")

    if not glob.glob(args.runs_glob):
        raise SystemExit(
            f"no runs matched {args.runs_glob}; run main_scripts/02_estimate_models.py first"
        )

    print("residual autocorrelation and order selection...")
    acf, bic = residual_evidence(args.runs_glob, data)
    acf.to_csv(OUT / "residual_acf.csv", index=False)
    bic.to_csv(OUT / "error_order_bic.csv", index=False)
    print(acf[["data_spec", "acf_lag1", "acf_lag2", "acf_lag3", "acf_lag4", "ljung_box_4"]]
          .to_string(index=False))
    print()
    print(bic[["data_spec"] + [f"bic_{k}" for k in SPECS] + ["best"]].to_string(index=False))
    print(f"\n  MA(3) beats AR(1) in {int((bic['bic_MA(3)'] < bic['bic_AR(1)']).sum())}"
          f"/{len(bic)} cells")

    if not args.skip_monte_carlo:
        print(f"\nbias Monte Carlo, {args.monte_carlo_reps} replications...")
        mc = bias_monte_carlo(args.monte_carlo_reps)
        mc.to_csv(OUT / "bias_monte_carlo.csv", index=False)
        for name, truth in (("alpha", 0.60), ("kappa", 0.15)):
            iid_bias = mc[f"{name}_iid"].mean() - truth
            ma3_bias = mc[f"{name}_ma3"].mean() - truth
            better = int((abs(mc[f"{name}_ma3"] - truth) < abs(mc[f"{name}_iid"] - truth)).sum())
            print(f"  {name}: iid bias {iid_bias:+.4f}, MA(3) bias {ma3_bias:+.4f}, "
                  f"MA(3) closer in {better}/{len(mc)} replications")

    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
