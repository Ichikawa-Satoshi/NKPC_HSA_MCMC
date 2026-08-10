"""Full-posterior forward-filtered fit comparison.

Complements the Chib marginal likelihoods. For every model we score the SAME target
series -- inflation -- by its one-step-ahead predictive density,

    LPD = sum_t log p(pi_t | pi_{1:t-1}, x, [N_{1:t-1}], theta),

then averages over parameter draws from the *full-sample* posterior. States are
forward-filtered, but the parameter draws have seen every observation.  The result is
therefore an in-sample, full-posterior diagnostic, not a genuine prequential or
out-of-sample predictive score.

The HSA models may use the firm count N as extra information; CES cannot. That
asymmetry is the economic question itself ("does the firm count help explain
inflation?"), and unlike the joint marginal likelihood it does not mix different
target variables: every model is scored on p(pi).

The legacy WAIC/PSIS-LOO transforms are retained for historical comparison only. They
are not standard WAIC/LOO because their input is a forward-filtered conditional density,
not the pointwise likelihood used to fit the full posterior.

    python scripts/predictive_comparison.py [--draws 300]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

import _bootstrap  # noqa: F401
from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import (
    ESTIMATION_REVISION,
    _coerce_model_data,
    _prepare_competition_measurement,
    model_sample_index,
)
from nkpc_hsa.dataprep.transforms import DEFAULT_N_TRANSFORM

TAB = RESULTS_DIR / "evidence" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

MODELS = ["ces", "hsa_steady", "hsa_dynamic", "hsa_full"]
UNEMP = {"Headline CPI": "unemployment_gap",
         "Core CPI": "unemployment_gap_core",
         "PPI": "unemployment_gap_ppi"}


def find_run(model, spec, freq="quarterly_interpolated"):
    best = None
    for d in sorted(glob.glob(str(RESULTS_DIR / "runs" / f"{model}_{spec}_baseline_*"))):
        mp = Path(d) / "metadata.json"
        if not mp.exists():
            continue
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        if (m.get("model") == model and m.get("data_spec") == spec
                and m.get("competition_measurement_frequency") == freq
                and m.get("period", "full") == "full"
                and m.get("constraint_spec", "unrestricted") == "unrestricted"
                and str(m.get("estimation_revision", "")) == ESTIMATION_REVISION):
            best = Path(d)
    return best


def _flat(post, name):
    if name not in post:
        return None
    return np.asarray(post[name], dtype=float).reshape(-1)


def _log_norm(x, mu, var):
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


def build_scoring_data(frame, spec, frequency):
    """Use the estimator's exact sample and competition-observation construction."""
    if "DATE" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame["DATE"] = pd.to_datetime(frame["DATE"])
        frame = frame.set_index("DATE")
    model_data = _coerce_model_data(frame, data_spec=spec)
    sample_index = model_sample_index(frame, spec)
    context = _prepare_competition_measurement(
        model="hsa_steady",
        data=frame,
        data_spec=spec,
        model_data=model_data,
        sample_index=sample_index,
        n_transform=DEFAULT_N_TRANSFORM,
        competition_measurement={"frequency": frequency, "annual_timing": "q4"},
    )
    n_obs = context.get("N_obs_used")
    if n_obs is None:
        raise ValueError(f"Could not construct competition observations for {frequency!r}.")
    out = {key: np.asarray(model_data[key], float) for key in ["pi", "pi_prev", "pi_expect", "x", "x_prev"]}
    out["N"] = np.asarray(n_obs, float)
    return out


def _linearized_state_observation(*, mean, x_t, delta, theta0, gamma):
    """First-order representation whose value is exact at ``mean``."""
    nhat, nbar = float(mean[0]), float(mean[2])
    h = np.array([-(theta0 + gamma * nbar), 0.0, delta * x_t - gamma * nhat])
    # f(mean) - grad f(mean)' mean for
    # f = delta*x*Nbar - theta0*Nhat - gamma*Nbar*Nhat.
    intercept = gamma * nhat * nbar
    return h, intercept


def prequential_ces(post, d, idx):
    """CES: inflation is a static regression given params (no latent state)."""
    a = _flat(post, "alpha")[idx]
    k = _flat(post, "kappa")[idx]
    lam = _flat(post, "lambda_ez")
    lam = np.zeros_like(a) if lam is None else lam[idx]
    se = _flat(post, "sigma_e")[idx]
    sz = _flat(post, "sigma_zeta")[idx]
    eta2 = np.maximum(se ** 2 - (lam ** 2) * (sz ** 2), 1e-10)
    zeta = d["x"][None, :] - _flat(post, "phi_1")[idx][:, None] * d["x_prev"][None, :]
    mu = (a[:, None] * d["pi_prev"][None, :] + (1 - a[:, None]) * d["pi_expect"][None, :]
          + k[:, None] * d["x"][None, :] + lam[:, None] * zeta)
    return _log_norm(d["pi"][None, :], mu, eta2[:, None])   # (S,T)


def prequential_hsa(post, d, idx, kind):
    """HSA: Kalman filter over s=(Nhat, Nhat_lag, Nbar); score the inflation row's
    one-step-ahead predictive density, then update with (N, pi) both observed."""
    S, T = idx.size, d["pi"].size
    a = _flat(post, "alpha")[idx]
    lam = _flat(post, "lambda_ez"); lam = np.zeros(S) if lam is None else lam[idx]
    ph = _flat(post, "phi_1")[idx]
    se, sz = _flat(post, "sigma_e")[idx], _flat(post, "sigma_zeta")[idx]
    eta2 = np.maximum(se ** 2 - (lam ** 2) * (sz ** 2), 1e-10)
    r1, r2 = _flat(post, "rho_1")[idx], _flat(post, "rho_2")[idx]
    nd = _flat(post, "n")[idx]
    su2, sp2 = _flat(post, "sigma_u")[idx] ** 2, _flat(post, "sigma_eps")[idx] ** 2
    sN = _flat(post, "sigma_N"); sN2 = (np.full(S, 1e-6) if sN is None else sN[idx] ** 2)
    if kind == "steady":
        k0, dl = _flat(post, "kappa_0")[idx], _flat(post, "delta")[idx]
        th0 = np.zeros(S); gm = np.zeros(S)
    elif kind == "dynamic":
        k0 = _flat(post, "kappa")[idx]; dl = np.zeros(S)
        th0 = _flat(post, "theta")[idx]; gm = np.zeros(S)
    else:  # full / const_theta
        k0, dl = _flat(post, "kappa_0")[idx], _flat(post, "delta")[idx]
        t0 = _flat(post, "theta_0"); th0 = (_flat(post, "theta") if t0 is None else t0)[idx]
        g = _flat(post, "gamma"); gm = np.zeros(S) if g is None else g[idx]

    out = np.empty((S, T))
    for s in range(S):
        F = np.array([[r1[s], r2[s], 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        c = np.array([0.0, 0.0, nd[s]])
        Q = np.diag([su2[s], 1e-10, sp2[s]])
        # Match the estimating model's declared initial prior.  Substituting N[0]
        # here both changed the model and used the first competition observation
        # before its measurement update.
        m = np.zeros(3); P = np.eye(3) * 10.0
        zeta = d["x"] - ph[s] * d["x_prev"]
        for t in range(T):
            if t > 0:
                m = F @ m + c
                P = F @ P @ F.T + Q
            det = (a[s] * d["pi_prev"][t] + (1 - a[s]) * d["pi_expect"][t]
                   + k0[s] * d["x"][t] + lam[s] * zeta[t])
            # Extended-Kalman linearisation for the bilinear full model.  The
            # intercept is essential: without it the approximation does not even
            # reproduce the nonlinear observation function at its expansion point.
            h, state_intercept = _linearized_state_observation(
                mean=m, x_t=d["x"][t], delta=dl[s], theta0=th0[s], gamma=gm[s]
            )
            mu = det + state_intercept + h @ m
            var = float(h @ P @ h + eta2[s])
            out[s, t] = _log_norm(d["pi"][t], mu, max(var, 1e-12))
            # Match the mixed-frequency likelihood: a missing annual N observation
            # drops only that row, while the inflation row remains.
            if np.isfinite(d["N"][t]):
                H = np.vstack([np.array([1.0, 0.0, 1.0]), h])
                y = np.array([d["N"][t], d["pi"][t] - det - state_intercept])
                R = np.diag([sN2[s], eta2[s]])
            else:
                H = h[None, :]
                y = np.array([d["pi"][t] - det - state_intercept])
                R = np.array([[eta2[s]]])
            Sm = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(Sm)
            m = m + K @ (y - H @ m)
            P = (np.eye(3) - K @ H) @ P
            P = (P + P.T) / 2.0
    return out


def plugin_score(post, d, model):
    """The in-sample plug-in fit score the report has always described.

    Evaluate the inflation equation at posterior-MEAN parameters and states and score
    the residuals with a Gaussian log-likelihood at the posterior-mean shock variance.
    It reuses the estimation sample and collapses the posterior to a point, so it is
    neither out-of-sample nor a posterior predictive density; it is reported next to the
    forward-filtered scores so the reader can compare two explicitly in-sample diagnostics.
    """
    mean = lambda name: float(np.mean(_flat(post, name)))
    alpha = mean("alpha")
    Nhat = None
    theta_t = None
    if model == "ces":
        kappa_t = np.full(d["pi"].size, mean("kappa"))
    else:
        Nbar = np.asarray(post["Nbar"], float).reshape(-1, d["pi"].size).mean(axis=0)
        if model == "hsa_dynamic":
            kappa_t = np.full(d["pi"].size, mean("kappa"))
        else:
            kappa_t = mean("kappa_0") + mean("delta") * Nbar
        if model in {"hsa_dynamic", "hsa_full"}:
            Nhat = np.asarray(post["Nhat"], float).reshape(-1, d["pi"].size).mean(axis=0)
            if model == "hsa_dynamic":
                theta_t = np.full(d["pi"].size, mean("theta"))
            else:
                theta_name = "theta_0" if "theta_0" in post else "theta"
                theta_t = np.full(d["pi"].size, mean(theta_name))
                if "gamma" in post:
                    theta_t = theta_t + mean("gamma") * Nbar
    phi = mean("phi_1")
    lam = 0.0 if "lambda_ez" not in post else mean("lambda_ez")
    zeta = d["x"] - phi * d["x_prev"]
    fitted = alpha * d["pi_prev"] + (1.0 - alpha) * d["pi_expect"] + kappa_t * d["x"]
    fitted = fitted + lam * zeta
    if Nhat is not None and theta_t is not None:
        fitted = fitted - theta_t * Nhat
    resid = d["pi"] - fitted
    if "sigma_eta" in post:
        sigma2 = mean("sigma_eta") ** 2
    else:
        sigma2 = max(mean("sigma_e") ** 2 - lam**2 * mean("sigma_zeta") ** 2, 1e-12)
    T = resid.size
    return float(-0.5 * T * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(resid**2) / sigma2)


def scores(ll):
    """ll: (S,T) pointwise log-lik. Returns LPD, WAIC, PSIS-LOO, max Pareto k."""
    import arviz as az
    S, T = ll.shape
    lpd_t = logsumexp(ll, axis=0) - np.log(S)
    lpd = float(lpd_t.sum())
    p_waic = float(np.var(ll, axis=0, ddof=1).sum())
    waic = float(lpd - p_waic)
    try:
        dt = az.from_dict({"posterior": {"_d": np.zeros((1, S))},
                           "log_likelihood": {"pi": ll[None, :, :]}})
        r = az.loo(dt, var_name="pi")
        loo = float(np.asarray(r.elpd if hasattr(r, "elpd") else r.elpd_loo))
        kmax = float(np.nanmax(np.asarray(r.pareto_k))) if hasattr(r, "pareto_k") else float("nan")
    except Exception:
        loo, kmax = float("nan"), float("nan")
    return lpd, waic, loo, kmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--frequency", default=None,
                    choices=["annual_q4", "quarterly_interpolated"],
                    help="Observation design; default runs both.")
    args = ap.parse_args()
    designs = [args.frequency] if args.frequency else ["annual_q4", "quarterly_interpolated"]
    specs = configured_data_specs(load_model_config())
    df = pd.read_csv(
        DATA_DIR / "processed" / "model_ready.csv", parse_dates=["DATE"]
    ).set_index("DATE")
    import arviz as az

    rows = []
    for freq in designs:
      for price, spec in UNEMP.items():
          d = build_scoring_data(df, specs[spec], freq)
          base = None
          for model in MODELS:
              # CES has no latent firm-count state, so its likelihood and posterior are
              # invariant to the observation design; the same 16 cells are shared,
              # exactly as in scripts/12_build_cpi_ppi_report.py.
              run = find_run(model, spec, "quarterly_interpolated" if model == "ces" else freq)
              if run is None:
                  continue
              post = az.from_netcdf(run / "posterior.nc").posterior
              n_all = _flat(post, "alpha").size
              idx = np.linspace(0, n_all - 1, min(args.draws, n_all)).astype(int)
              if model == "ces":
                  ll = prequential_ces(post, d, idx)
              else:
                  kind = {"hsa_steady": "steady", "hsa_dynamic": "dynamic"}.get(model, "full")
                  ll = prequential_hsa(post, d, idx, kind)
              lpd, waic, loo, kmax = scores(ll)
              if model == "ces":
                  base = lpd
              rows.append({"design": freq, "price": price, "model": model,
                           # Which vintage of the runs these scores came from. The
                           # scores are already filtered on it; recording it means a
                           # consumer of this CSV can tell whether it is current
                           # instead of silently reformatting a stale file.
                           "estimation_revision": ESTIMATION_REVISION,
                           "plugin_score": round(plugin_score(post, d, model), 2),
                           "LPD_1step": round(lpd, 2), "WAIC": round(waic, 2),
                           "LOO": round(loo, 2), "max_pareto_k": round(kmax, 2),
                           "dLPD_vs_ces": None if base is None else round(lpd - base, 2)})
              print(f"{freq[:6]:6s} {price:13s} {model:12s} LPD={lpd:9.2f}  WAIC={waic:9.2f}  LOO={loo:9.2f}  k={kmax:4.2f}"
                    + ("" if base is None else f"  dLPD vs CES={lpd-base:+7.2f}"), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "predictive_comparison.csv", index=False)
    json.dump(rows, open(TAB / "predictive_comparison.json", "w"), indent=2)
    print(f"\nsaved -> {TAB/'predictive_comparison.csv'}")


if __name__ == "__main__":
    main()
