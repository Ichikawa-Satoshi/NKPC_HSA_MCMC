"""Sign check (unrestricted) and sign-restriction Bayes factor for the HSA signs.

Two complementary readings of the same posterior draws:

  (1) UNRESTRICTED SIGN CHECK -- posterior probabilities that each HSA
      coefficient lies in its theory-predicted half-line (delta>0, theta_0>0,
      gamma>0; theta-positive convention, inflation loads -theta_0*Nhat),
      estimated with symmetric mean-zero priors (no sign imposed).

  (2) SIGN-RESTRICTION MARGINAL-LIKELIHOOD COMPARISON -- the Bayes factor of a
      theory-sign-restricted model (same priors truncated to the theory region)
      against the unrestricted model.  For a purely inequality restriction under
      the encompassing prior this is exact (Klugkist & Hoijtink 2007):

          BF = P(theta in region | y) / P(theta in region | prior)
             = (posterior region probability) / 0.5^k

      because the constrained coefficients have symmetric mean-zero priors, so
      each half-line has prior probability 0.5 and they are a priori independent.

  BF>1 favours the theory restriction; BF<1 favours the unrestricted model;
  BF~1 means the data are uninformative about the restriction (posterior region
  probability equals the prior 0.5^k -- the weak-identification case).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]
from nkpc_hsa.paths import results_root  # noqa: E402

MODEL_LABELS = {1: "Slope", 2: "Direct", 3: "Dynamic", 4: "Joint"}
# theory-predicted region per model (theta-positive convention): all signs +1.
REGION = {
    1: [("delta", +1)],
    2: [("theta_0", +1)],
    3: [("theta_0", +1), ("gamma", +1)],
    4: [("delta", +1), ("theta_0", +1), ("gamma", +1)],
}


def _flat(d: dict, name: str) -> np.ndarray | None:
    names = list(d["coeff_names"])
    if name not in names:
        return None
    return d["coeffs"][:, :, names.index(name)].reshape(-1)


def _kass_raftery(bf: float) -> str:
    if not np.isfinite(bf):
        return "--"
    if bf < 1 / 10:
        return "against (strong)"
    if bf < 1 / 3:
        return "against (subst.)"
    if bf < 1:
        return "against (weak)"
    if bf < 3:
        return "for (weak)"
    if bf < 10:
        return "for (subst.)"
    if bf < 100:
        return "for (strong)"
    return "for (decisive)"


def analyse(results_dir: Path, variant_of_case, primary=("ppi", "inverse_markup")) -> pd.DataFrame:
    rows = []
    for case in (1, 2, 3, 4):
        var = variant_of_case(case)
        spec_dir = results_dir / f"case{case}" / f"{primary[0]}__{primary[1]}__{var}"
        for model in (1, 2, 3, 4):
            path = spec_dir / f"model{model}.npz"
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=True)
            d = {k: z[k] for k in z.files}
            region = REGION[model]
            k = len(region)
            mask = np.ones(d["coeffs"].shape[0] * d["coeffs"].shape[1], dtype=bool)
            marg = {}
            for name, sign in region:
                f = _flat(d, name)
                if f is None:
                    mask = None
                    break
                inreg = (f > 0) if sign > 0 else (f < 0)
                marg[name] = float(inreg.mean())
                mask &= inreg
            if mask is None:
                continue
            post_region = float(mask.mean())
            prior_region = 0.5 ** k
            bf = post_region / prior_region if prior_region > 0 else np.nan
            rows.append({
                "case": case, "model": model, "model_label": MODEL_LABELS[model],
                "k_constraints": k,
                "P_delta_pos": marg.get("delta", np.nan),
                "P_theta0_pos": marg.get("theta_0", np.nan),
                "P_gamma_pos": marg.get("gamma", np.nan),
                "post_region": post_region, "prior_region": prior_region,
                "bayes_factor": bf, "log10_BF": np.log10(bf) if bf > 0 else -np.inf,
                "verdict": _kass_raftery(bf),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--variant", default=None, help="fix the Case-1 competition variant")
    args = ap.parse_args()
    rdir = args.results_dir or (results_root() / "report_estimation" / "pilot_hybrid")

    def var_of(case: int) -> str:
        if case == 1:
            return args.variant or "firm_weighted"
        return "gustavo"

    df = analyse(rdir, var_of)
    if df.empty:
        print(f"no draws found under {rdir}")
        return
    pd.set_option("display.width", 200)
    show = df[["case", "model_label", "P_delta_pos", "P_theta0_pos", "P_gamma_pos",
               "post_region", "prior_region", "bayes_factor", "log10_BF", "verdict"]]
    print(f"results: {rdir}\n")
    print(show.round(3).to_string(index=False))
    out = rdir / "sign_restriction.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
