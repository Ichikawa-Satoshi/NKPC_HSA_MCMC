"""Attribute every changed report number to the fix that caused it.

`scripts/report_artifact_diff.py` shows *that* an artifact changed, against the
snapshot taken before any of the fixes. It cannot say *why*, because three
independent changes moved numbers at once:

  T1  hsa_full            alternating FFBS  ->  Particle Gibbs
  T2  convergence rule    9 scalars         ->  18 scalars + state + derived paths
  T3  hsa_const_theta     alternating FFBS  ->  exact joint FFBS

This script separates them. T1 and T3 are read directly off disk, because the
superseded runs were never deleted; T2 is recomputed in-process by applying both
rules to the *same* posteriors, so no re-estimation is involved.

    python scripts/fix_attribution.py
"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

warnings.filterwarnings("ignore")
import arviz as az  # noqa: E402

REV = "2026-08-state-initial-covariance-v2"
OLD_SCALARS = ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma", "rho_1", "rho_2"]


def _load12():
    spec = importlib.util.spec_from_file_location("b12", ROOT / "scripts" / "12_build_cpi_ppi_report.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b12"] = mod
    spec.loader.exec_module(mod)
    return mod


def _group(idata, names):
    """max Rhat / min bulk ESS over a set of stored variables."""
    rhat, ess = -np.inf, np.inf
    for name in names:
        if name not in idata.posterior:
            continue
        arr = idata.posterior[name]
        if float(np.nanstd(np.asarray(arr))) <= 0.0:
            continue
        rhat = max(rhat, float(np.nanmax(np.asarray(az.rhat(arr), dtype=float))))
        ess = min(ess, float(np.nanmin(np.asarray(az.ess(arr, method="bulk"), dtype=float))))
    return (rhat, ess) if np.isfinite(rhat) else (np.nan, np.nan)


def _summ(idata, name):
    if name not in idata.posterior:
        return None
    v = np.asarray(idata.posterior[name], dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    return v.mean(), np.quantile(v, 0.025), np.quantile(v, 0.975)


def _fmt(s):
    return "--" if s is None else f"{s[0]:+.4f} [{s[1]:+.4f},{s[2]:+.4f}]"


def _index_runs(model, marker):
    """Newest run per (spec, prior, freq), split by a run-id marker."""
    out = {}
    for path in sorted((ROOT / "results" / "runs").glob("*/posterior.nc")):
        idata = az.from_netcdf(path)
        a = idata.attrs
        if str(a.get("model")) != model or str(a.get("estimation_revision", "")) != REV:
            continue
        if str(a.get("period", "full")) != "full":
            continue
        if str(a.get("constraint_spec", "unrestricted")) != "unrestricted":
            continue
        key = (
            str(a.get("data_spec")),
            str(a.get("prior_spec", "baseline")),
            str(a.get("competition_measurement_frequency", "")),
        )
        rid = str(a.get("run_id", ""))
        bucket = "new" if marker in rid else "old"
        slot = out.setdefault(key, {})
        if bucket not in slot or rid >= slot[bucket][0]:
            slot[bucket] = (rid, idata)
    return out


def sampler_change(model, marker, params, title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    runs = _index_runs(model, marker)
    rows, flips = [], {"watch->OK": 0, "OK->watch": 0, "same": 0, "n/a": 0}
    for key in sorted(runs):
        slot = runs[key]
        if "old" not in slot or "new" not in slot:
            flips["n/a"] += 1
            continue
        old, new = slot["old"][1], slot["new"][1]
        ro, eo = _group(old, [p for p in params if p in old.posterior] or params)
        rn, en = _group(new, [p for p in params if p in new.posterior] or params)
        co = np.isfinite(ro) and ro <= 1.01 and eo >= 400
        cn = np.isfinite(rn) and rn <= 1.01 and en >= 400
        flips["watch->OK" if (cn and not co) else "OK->watch" if (co and not cn) else "same"] += 1
        rows.append((key, old, new, ro, eo, rn, en, co, cn))

    print(f"{'spec / prior / freq':52s} {'旧 R^/ESS':>16s} {'新 R^/ESS':>16s}  判定")
    for key, _, _, ro, eo, rn, en, co, cn in rows:
        lab = f"{key[0]} / {key[1]} / {'PCHIP' if key[2].startswith('quarter') else 'annual'}"
        mark = "  watch->OK" if (cn and not co) else "  OK->watch" if (co and not cn) else ""
        print(f"{lab:52s} {ro:7.3f}/{eo:<8.0f} {rn:7.3f}/{en:<8.0f}{mark}")
    print(f"\n  収束判定の変化: {flips}")
    return rows


def coefficient_shifts(rows, names):
    print(f"\n  係数の変化（baseline のみ）")
    print(f"  {'cell':44s} {'param':9s} {'旧':28s} {'新':28s}")
    for key, old, new, *_ in rows:
        if key[1] != "baseline":
            continue
        lab = f"{key[0]} / {'PCHIP' if key[2].startswith('quarter') else 'annual'}"
        for nm in names:
            so, sn = _summ(old, nm), _summ(new, nm)
            if so is None and sn is None:
                continue
            print(f"  {lab:44s} {nm:9s} {_fmt(so):28s} {_fmt(sn):28s}")


def diagnostics_rule_change(m):
    """T2: same posteriors, old 9-scalar rule vs the new grouped rule."""
    print(f"\n{'=' * 100}\nT2  収束判定ルールの拡張（同一 posterior に両ルールを適用）\n{'=' * 100}")
    print("  旧ルール: 9係数のみ  /  新ルール: 18スカラー（n・分散を含む）+ 状態パス + 派生パス\n")
    for freq, label in [("quarterly_interpolated", "PCHIP"), ("annual_q4", "annual-Q4")]:
        runs = m.load_report_runs(min_iter=1, competition_frequency=freq)
        changed = []
        for key in sorted(runs):
            idata = runs[key][1]
            ro, eo = _group(idata, OLD_SCALARS)
            old_ok = np.isfinite(ro) and ro <= 1.01 and eo >= 400
            d = m._diagnostics(idata)
            new_ok = bool(d["converged"])
            if old_ok != new_ok:
                worst = d["scalar"]["worst_ess"]
                changed.append((key, ro, eo, d["max_rhat"], d["min_ess"], worst))
        print(f"  {label}: {len(runs)}セル中 {len(changed)}セルで判定が変化")
        for key, ro, eo, rn, en, worst in changed:
            print(f"    {'/'.join(key):52s} 旧 {ro:.3f}/{eo:<7.0f} -> 新 {rn:.3f}/{en:<7.0f}  原因={worst}")
        # cells where coefficients pass but paths do not
        coef_only = [
            key for key in sorted(runs)
            if (d := m._diagnostics(runs[key][1]))["converged"] and d["has_states"]
            and not d["joint_converged"]
        ]
        print(f"    うち「係数OK・状態パス不合格」= {len(coef_only)}セル"
              f"{'  ' + ', '.join('/'.join(k) for k in coef_only[:4]) if coef_only else ''}")


def main() -> None:
    m = _load12()
    rows_ct = sampler_change(
        "hsa_const_theta", "jointffbs",
        ["alpha", "kappa_0", "delta", "theta", "rho_1", "rho_2", "n", "phi_1", "lambda_ez",
         "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps", "sigma_N"],
        "T3  hsa_const_theta : 交互FFBS -> 厳密 joint FFBS",
    )
    coefficient_shifts(rows_ct, ["delta", "theta", "kappa_0"])

    rows_full = sampler_change(
        "hsa_full", "_pg",
        ["alpha", "kappa_0", "delta", "theta_0", "gamma", "rho_1", "rho_2", "n", "phi_1",
         "lambda_ez", "sigma_e", "sigma_zeta", "sigma_u", "sigma_eps", "sigma_N"],
        "T1  hsa_full : 交互FFBS -> Particle Gibbs",
    )
    coefficient_shifts(rows_full, ["delta", "theta_0", "gamma"])

    diagnostics_rule_change(m)


if __name__ == "__main__":
    main()
