from __future__ import annotations
import os, re, json, glob
import pandas as pd

ROOT = "/Users/satoshi/GitHub/NKPC_HSA_MCMC"
TAB = os.path.join(ROOT, "results/tables")
FIGREL = "figures"   # relative to results/report.html

SPECS = [
    ("unemployment_gap", "Unemployment gap"),
    ("unemployment_gap_core", "Unemployment gap · core CPI"),
    ("output_gap_bn", "Output gap · BN filter"),
    ("output_gap_bn_core", "Output gap BN · core CPI"),
    ("output_gap_hp", "Output gap · HP filter"),
    ("inv_markup", "Inverse-markup gap"),
    ("labor_share_gap_hp", "Labor-share gap · HP"),
]
FREQ_R = {"quarterly_interpolated": 0, "annual_q4": 1}
PRIOR_R = {"baseline": 0, "weak": 1, "tight": 2}
PERIOD_R = {"full": 0, "pre_2008": 1, "post_2008": 2, "start_1988": 3, "end_2019": 4, "exclude_covid": 5}
CONSTR_R = {"unrestricted": 0, "restricted_kappa": 1, "restricted_kappa_t": 2}
PERIOD_LBL = {"full": "full sample", "pre_2008": "pre-2008", "post_2008": "post-2008",
              "start_1988": "from 1988", "end_2019": "to 2019", "exclude_covid": "ex-COVID"}
FREQ_LBL = {"quarterly_interpolated": "quarterly", "annual_q4": "annual Q4"}
CONSTR_LBL = {"unrestricted": "unrestricted", "restricted_kappa": "κ ≥ 0", "restricted_kappa_t": "κ_t ≥ 0"}
PRIOR_LBL = {"baseline": "baseline prior", "weak": "weak prior", "tight": "tight prior"}
FIG_CAP = {
    "kappa_t_path.png": "Time-varying κ_t path",
    "theta_t_path.png": "Time-varying θ_t path",
    "prior_posterior_ces.png": "Prior vs posterior · CES",
    "prior_posterior_hsa_steady.png": "Prior vs posterior · HSA steady",
    "prior_posterior_hsa_dynamic.png": "Prior vs posterior · HSA dynamic",
    "prior_posterior_hsa_const_theta.png": "Prior vs posterior · HSA const-θ",
    "prior_posterior_hsa_full.png": "Prior vs posterior · HSA full",
}
FIG_ORDER = list(FIG_CAP)
KEY = ["alpha", "kappa", "kappa_0", "delta", "theta", "theta_0", "gamma"]
DYN = ["rho_1", "rho_2", "phi_1", "n", "sigma_e", "sigma_N"]


def ts(s):
    m = re.search(r"(\d{8}_\d{6})$", str(s))
    return m.group(1) if m else ""


def load_block(bdir, spec):
    """Return coeff/sddr/mc dicts filtered to exact spec + current transform + latest run."""
    out = {"co": None, "sd": None, "mc": None}
    cpath = os.path.join(bdir, "coefficients.csv")
    if os.path.exists(cpath):
        c = pd.read_csv(cpath)
        c = c[c["data_spec"] == spec]
        if "n_transform" in c:
            c = c[c["n_transform"] == "log100_centered10"]
        if len(c):
            c = c.assign(ts=c["run"].map(ts)).sort_values("ts").drop_duplicates(["model", "parameter"], keep="last")
            cell = {}
            for _, r in c.iterrows():
                cell.setdefault(r["model"], {})[r["parameter"]] = [
                    round(float(r["posterior_mean"]), 4), round(float(r["ci_2.5"]), 4),
                    round(float(r["ci_97.5"]), 4), round(float(r["p_gt_0"]), 3)]
            models = [m for m in ["ces", "hsa_steady", "hsa_dynamic", "hsa_const_theta", "hsa_full"] if m in cell]
            out["co"] = {"models": models, "cell": cell}
    spath = os.path.join(bdir, "sddr.csv")
    if os.path.exists(spath):
        s = pd.read_csv(spath)
        s = s[s["data_spec"] == spec]
        if len(s):
            s = s.assign(ts=s["run"].map(ts)).sort_values("ts").drop_duplicates(["model", "restriction"], keep="last")
            rows = []
            for _, r in s.iterrows():
                bf01 = float(r["sddr_bf01"])
                rows.append([r["model"], r["restriction"], round(1.0 / bf01, 2) if bf01 else None])
            out["sd"] = rows
    mpath = os.path.join(bdir, "model_comparison.csv")
    if os.path.exists(mpath):
        m = pd.read_csv(mpath)
        if "data_spec" in m:
            m = m[m["data_spec"] == spec]
        if "n_transform" in m:
            m = m[m["n_transform"] == "log100_centered10"]
        if len(m):
            m = m.assign(ts=m["run"].map(ts), has=m["predictive_score"].notna().astype(int))
            m = m.sort_values(["has", "ts"]).drop_duplicates(["model"], keep="last")
            order = {"ces": 0, "hsa_steady": 1, "hsa_dynamic": 2, "hsa_const_theta": 3, "hsa_full": 4}
            m = m.assign(o=m["model"].map(order)).sort_values("o")
            rows = []
            for _, r in m.iterrows():
                def g(k):
                    v = r.get(k)
                    return None if pd.isna(v) else float(v)
                rows.append([r["model"], g("predictive_score"), g("log_marginal_likelihood"), g("bayes_factor_vs_baseline")])
            out["mc"] = rows
    return out


def block_meta(name):
    p = name.split("__")
    spec, prior, period, constr, freq = p[0], p[1], p[2], p[3], p[4]
    is_main = (prior == "baseline" and period == "full" and constr == "unrestricted" and freq == "quarterly_interpolated")
    if is_main:
        title = "Baseline"
    else:
        bits = []
        if freq != "quarterly_interpolated":
            bits.append("Annual-Q4")
        if constr != "unrestricted":
            bits.append(CONSTR_LBL[constr])
        if prior != "baseline":
            bits.append(PRIOR_LBL[prior].replace(" prior", "-prior"))
        if period != "full":
            bits.append(PERIOD_LBL[period])
        title = " · ".join(bits) if bits else "Variant"
    meta = f"{PRIOR_LBL[prior]} · {PERIOD_LBL[period]} · {CONSTR_LBL[constr]} · {FREQ_LBL[freq]}"
    rank = (0 if period == "full" else 1, PRIOR_R[prior], CONSTR_R[constr], FREQ_R[freq], PERIOD_R[period])
    return title, meta, rank, is_main


def figs_for(spec, name):
    fdir = os.path.join(ROOT, "results/figures", spec, "blocks", name)
    if not os.path.isdir(fdir):
        return []
    out = []
    for fn in FIG_ORDER:
        if os.path.exists(os.path.join(fdir, fn)):
            out.append([FIG_CAP[fn], f"{FIGREL}/{spec}/blocks/{name}/{fn}"])
    return out


def build():
    data = {"specs": [], "labels": {}, "overview": {}}
    headline = []
    for spec, label in SPECS:
        bdir_root = os.path.join(TAB, spec, "blocks")
        if not os.path.isdir(bdir_root):
            continue
        names = [n for n in os.listdir(bdir_root)
                 if n.count("__") == 4 and n.split("__")[4] in FREQ_R]
        blocks = []
        for name in names:
            title, meta, rank, is_main = block_meta(name)
            b = load_block(os.path.join(bdir_root, name), spec)
            b.update({"name": name, "t": title, "meta": meta, "fg": figs_for(spec, name), "rank": rank})
            blocks.append(b)
            if is_main and b["co"]:
                cell = b["co"]["cell"].get("hsa_steady", {})
                d = cell.get("delta"); k0 = cell.get("kappa_0")
                bf = next((r[2] for r in (b["sd"] or []) if r[0] == "hsa_steady" and r[1] == "delta=0"), None)
                headline.append([label, k0, d, bf])
        blocks.sort(key=lambda x: x["rank"])
        for b in blocks:
            del b["rank"]
        data["specs"].append(spec)
        data["labels"][spec] = label
        data.setdefault("blocks", {})[spec] = blocks
    data["overview"]["headline"] = headline
    return data


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    data = build()
    payload = json.dumps(data, separators=(",", ":"))
    tmpl = open(os.path.join(here, "_report_template.html")).read()
    out = os.path.join(ROOT, "results", "report.html")
    open(out, "w").write(tmpl.replace("__DATA__", payload))
    nb = sum(len(v) for v in data.get("blocks", {}).values())
    print("specs:", len(data["specs"]), "blocks:", nb, "json_kb:", round(len(payload) / 1024))
