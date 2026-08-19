"""Appendix: Particle Gibbs (joint conditional-SMC) state update for PCHIP hsa_full.

Stages (run via argv): validate | pilot | produce | all
Outputs go to results/evidence/ and NEVER overwrite existing runs.

  python main_scripts/appendix_particle_gibbs_hsa_full.py validate
  python main_scripts/appendix_particle_gibbs_hsa_full.py pilot
  python main_scripts/appendix_particle_gibbs_hsa_full.py produce --particles 512
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401  (sets up sys.path / ROOT)
from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT

from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import _coerce_model_data
from nkpc_hsa.dataprep.transforms import DEFAULT_N_TRANSFORM
from nkpc_hsa.dataprep import transform_competition_series
from nkpc_hsa.models.common import prior_specs_to_internal
from nkpc_hsa.gibbs.hsa_full.model import KAPPA_SCALE, _common_priors
from nkpc_hsa.gibbs.hsa_full_pg.model import (
    func_nkpc_hsa_full_pg,
    sample_states_joint_ffbs_gamma0,
    sample_states_particle_gibbs,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = RESULTS_DIR / "evidence"
FIG = OUT / "figures"
TAB = OUT / "tables"
DRAWS = OUT / "draws"
LOG = OUT / "logs"
for d in (OUT, FIG, TAB, DRAWS, LOG):
    d.mkdir(parents=True, exist_ok=True)

CELL = "unemployment_gap_core"          # default for validate / pilot
BASELINE_CELLS = [
    "unemployment_gap", "unemployment_gap_core", "unemployment_gap_ppi",
    "output_gap_hp", "output_gap_hp_core", "output_gap_hp_ppi",
    "output_gap_bn", "output_gap_bn_core", "output_gap_bn_ppi",
]
SEED = 12345
N_ITER, BURN, THIN, CHAINS = 12000, 4000, 5, 2


def find_existing_run(spec):
    """Current-revision PCHIP baseline hsa_full run dir for a data spec (for comparison)."""
    import glob
    from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION
    cands = []
    for d in sorted(glob.glob(str(RESULTS_DIR / "runs" / f"hsa_full_{spec}_baseline_*"))):
        mp = Path(d) / "metadata.json"
        if not mp.exists():
            continue
        try:
            m = json.load(open(mp))
        except Exception:  # noqa: BLE001
            continue
        if (m.get("data_spec") == spec
                and m.get("competition_measurement_frequency") == "quarterly_interpolated"
                and m.get("period", "full") == "full"
                and m.get("constraint_spec", "unrestricted") == "unrestricted"
                and str(m.get("estimation_revision", "")) == ESTIMATION_REVISION):
            cands.append(Path(d))
    return cands[-1] if cands else None


EXISTING_RUN = find_existing_run(CELL)


def log(msg: str, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


# --------------------------------------------------------------------------- #
def load_cell(spec_name=CELL, prior_name="baseline"):
    specs = configured_data_specs(load_model_config())
    spec = specs[spec_name]
    df = pd.read_csv(DATA_DIR / "processed" / "model_ready.csv")
    md = _coerce_model_data(df, data_spec=spec)
    N = transform_competition_series(md["N"], transform=DEFAULT_N_TRANSFORM)
    pri_dict = yaml.safe_load(open(ROOT / "configs" / f"priors_{prior_name}.yaml"))
    pri_int = prior_specs_to_internal(pri_dict)
    return {
        "pi": np.asarray(md["pi"], float),
        "pi_prev": np.asarray(md["pi_prev"], float),
        "pi_expect": np.asarray(md["pi_expect"], float),
        "x": np.asarray(md["x"], float),
        "x_prev": np.asarray(md["x_prev"], float),
        "N": np.asarray(N, float),
        "pri_int": pri_int,
        "pri_dict": pri_dict,
        "spec": spec,
    }


def init_geom(pri):
    return dict(
        m0_Nhat=pri["m0_Nhat"], P0_Nhat=pri["P0_Nhat"],
        m0_Nhat_lag=pri["m0_Nhat_lag"], P0_Nhat_lag=pri["P0_Nhat_lag"],
        m0_Nbar=pri["m0_Nbar"], P0_Nbar=pri["P0_Nbar"],
    )


# --------------------------------------------------------------------------- #
# STAGE 1 -- validation against exact joint FFBS in the gamma == 0 linear case
# --------------------------------------------------------------------------- #
def stage_validate(cell):
    from scipy import stats as sps

    fh = open(LOG / "validate.log", "w")
    log("=== VALIDATION: gamma=0 linear special case (PG vs exact joint FFBS) ===", fh)
    pri = _common_priors(cell["pri_int"])
    geom = init_geom(pri)
    y = cell["pi"] - cell["pi_expect"]
    a_t = cell["pi_prev"] - cell["pi_expect"]
    x_t, N_obs = cell["x"], cell["N"]

    # Fixed, realistic parameters (physical units for kappa0/delta), gamma = 0.
    fixed = dict(
        alpha=0.5, kappa0_eff=0.06, delta_eff=0.02, theta0=0.05, gamma=0.0,
        lambda_ez=0.0, rho1=0.6, rho2=-0.2, n_drift=-0.05,
        sigma_eta2=0.5, sigma_u2=0.02, sigma_eps2=0.01, sigma_N2=0.01,
    )
    zeta = x_t - 0.8 * cell["x_prev"]  # any fixed phi; only enters via lambda_ez=0 here
    T = y.size
    M, BURN_PG = 3000, 300

    rng = np.random.default_rng(20260808)
    # --- exact FFBS: M iid draws ---
    t0 = time.time()
    ffbs_Nbar = np.empty((M, T)); ffbs_Nhat = np.empty((M, T))
    for m in range(M):
        r = sample_states_joint_ffbs_gamma0(
            y=y, a_t=a_t, x_t=x_t, zeta=zeta, N_obs=N_obs,
            alpha=fixed["alpha"], kappa0_eff=fixed["kappa0_eff"], delta_eff=fixed["delta_eff"],
            theta0=fixed["theta0"], lambda_ez=fixed["lambda_ez"],
            rho1=fixed["rho1"], rho2=fixed["rho2"], n_drift=fixed["n_drift"],
            sigma_eta2=fixed["sigma_eta2"], sigma_u2=fixed["sigma_u2"],
            sigma_eps2=fixed["sigma_eps2"], sigma_N2=fixed["sigma_N2"], rng=rng, **geom,
        )
        ffbs_Nbar[m] = r["Nbar"]; ffbs_Nhat[m] = r["Nhat"]
    log(f"exact FFBS: {M} draws in {time.time()-t0:.1f}s", fh)

    def ffbs_draw():
        return sample_states_joint_ffbs_gamma0(
            y=y, a_t=a_t, x_t=x_t, zeta=zeta, N_obs=N_obs,
            alpha=fixed["alpha"], kappa0_eff=fixed["kappa0_eff"], delta_eff=fixed["delta_eff"],
            theta0=fixed["theta0"], lambda_ez=fixed["lambda_ez"], rho1=fixed["rho1"], rho2=fixed["rho2"],
            n_drift=fixed["n_drift"], sigma_eta2=fixed["sigma_eta2"], sigma_u2=fixed["sigma_u2"],
            sigma_eps2=fixed["sigma_eps2"], sigma_N2=fixed["sigma_N2"], rng=rng, **geom,
        )

    results = {}
    # ONE-STEP INVARIANCE TEST: each PG step starts from an *independent* exact
    # posterior (FFBS) draw. A correct PG kernel leaves the exact posterior
    # invariant, so one-step outputs must be distributed as the exact posterior
    # -- this isolates correctness from the PG chain's (slow) mixing.
    for P in (128, 512):
        t0 = time.time()
        pg_Nbar = np.empty((M, T)); pg_Nhat = np.empty((M, T)); moved = []
        for m in range(M):
            r = ffbs_draw()
            out = sample_states_particle_gibbs(
                y=y, a_t=a_t, x_t=x_t, zeta=zeta, N_obs=N_obs,
                alpha=fixed["alpha"], kappa0_eff=fixed["kappa0_eff"], delta_eff=fixed["delta_eff"],
                theta0=fixed["theta0"], gamma=fixed["gamma"], lambda_ez=fixed["lambda_ez"],
                rho1=fixed["rho1"], rho2=fixed["rho2"], n_drift=fixed["n_drift"],
                sigma_eta2=fixed["sigma_eta2"], sigma_u2=fixed["sigma_u2"],
                sigma_eps2=fixed["sigma_eps2"], sigma_N2=fixed["sigma_N2"],
                Nbar_ref=r["Nbar"], Nhat_ref=r["Nhat"], Nhat_ref_lag=pri["m0_Nhat_lag"],
                n_particles=P, rng=rng, **geom,
            )
            pg_Nbar[m] = out["Nbar"]; pg_Nhat[m] = out["Nhat"]; moved.append(out["moved_frac"])
        dt = time.time() - t0

        def cmp(a, b):
            ma, mb = a.mean(0), b.mean(0)
            sa, sb = a.std(0), b.std(0)
            mc_se = sa / np.sqrt(M)
            return dict(
                max_abs_mean_diff=float(np.max(np.abs(ma - mb))),
                max_mean_diff_in_MCse=float(np.max(np.abs(ma - mb) / (mc_se + 1e-12))),
                max_abs_sd_diff=float(np.max(np.abs(sa - sb))),
                max_rel_sd_diff=float(np.max(np.abs(sa - sb) / (sa + 1e-12))),
            )
        cN = cmp(ffbs_Nbar, pg_Nbar); ch = cmp(ffbs_Nhat, pg_Nhat)
        ks = {f"Nbar_t{t}": float(sps.ks_2samp(ffbs_Nbar[:, t], pg_Nbar[:, t]).pvalue) for t in (0, T // 2, T - 1)}
        ks.update({f"Nhat_t{t}": float(sps.ks_2samp(ffbs_Nhat[:, t], pg_Nhat[:, t]).pvalue) for t in (0, T // 2, T - 1)})
        results[P] = dict(seconds=dt, moved_frac=float(np.mean(moved)), Nbar=cN, Nhat=ch, ks_pvalues=ks)
        log(f"[P={P}] {dt:.1f}s moved={np.mean(moved):.3f} | "
            f"Nbar max|dmean|={cN['max_abs_mean_diff']:.4f} ({cN['max_mean_diff_in_MCse']:.1f} MCse), "
            f"max|dSD|={cN['max_abs_sd_diff']:.4f} | Nhat max|dmean|={ch['max_abs_mean_diff']:.4f} "
            f"({ch['max_mean_diff_in_MCse']:.1f} MCse) | KS pvals min={min(ks.values()):.3f}", fh)

        if P == 512:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            tt = np.arange(T)
            for k, (a_f, a_p, name) in enumerate([(ffbs_Nbar, pg_Nbar, "Nbar"), (ffbs_Nhat, pg_Nhat, "Nhat")]):
                ax[k].plot(tt, a_f.mean(0), "k-", lw=2, label="exact FFBS mean")
                ax[k].fill_between(tt, a_f.mean(0) - a_f.std(0), a_f.mean(0) + a_f.std(0), color="k", alpha=0.15)
                ax[k].plot(tt, a_p.mean(0), "r--", lw=1.5, label=f"PG mean (P={P})")
                ax[k].fill_between(tt, a_p.mean(0) - a_p.std(0), a_p.mean(0) + a_p.std(0), color="r", alpha=0.15)
                ax[k].set_title(f"{name}: PG vs exact FFBS (gamma=0)"); ax[k].legend(fontsize=8)
            fig.tight_layout(); fig.savefig(FIG / "validation_gamma0_pg_vs_ffbs.png", dpi=130); plt.close(fig)

    json.dump({"fixed_params": fixed, "M": M, "results": results}, open(TAB / "validation.json", "w"), indent=2)
    log(f"Saved {FIG/'validation_gamma0_pg_vs_ffbs.png'} and {TAB/'validation.json'}", fh)
    fh.close()
    return results


# --------------------------------------------------------------------------- #
# STAGE 2 -- pilots at 128 / 256 / 512 particles
# --------------------------------------------------------------------------- #
def _ess_1d(x):
    import arviz as az
    return float(az.ess(np.asarray(x)[None, :]))


def stage_pilot(cell):
    fh = open(LOG / "pilot.log", "w")
    log("=== PILOTS (short chains) ===", fh)
    rows = []
    nb, nk = 800, 1500
    for P in (128, 256, 512):
        t0 = time.time()
        res = func_nkpc_hsa_full_pg(
            cell["pi"], cell["pi_prev"], cell["pi_expect"], cell["x"], cell["x_prev"], cell["N"],
            n_burn=nb, n_keep=nk, priors=cell["pri_int"],
            opts={"seed": 7, "store_every": 1, "n_particles": P, "verbose": False},
        )
        dt = time.time() - t0
        pg = res["pg_diagnostics"]
        gq = res["gamma"]["quantiles"]  # [0.025,0.05,0.25,0.5,0.75,0.95,0.975]
        rows.append(dict(
            particles=P, seconds=round(dt, 1), sec_per_iter=round(dt / (nb + nk), 4),
            pg_ess_mean=round(float(np.mean(pg["ess_mean"])), 1),
            pg_ess_min=round(float(np.min(pg["ess_min"])), 1),
            ess_per_particle=round(float(np.mean(pg["ess_mean"])) / P, 3),
            moved_frac=round(float(np.mean(pg["moved_frac"])), 3),
            delta_mean=round(float(res["delta"]["mean"]), 4),
            gamma_mean=round(float(res["gamma"]["mean"]), 4),
            gamma_q=[round(float(gq[0]), 4), round(float(gq[6]), 4)],
        ))
        log(f"[P={P}] {dt:.1f}s ({dt/(nb+nk)*1000:.1f} ms/iter) | pg_ESS mean={rows[-1]['pg_ess_mean']} "
            f"min={rows[-1]['pg_ess_min']} ({rows[-1]['ess_per_particle']:.2f}/particle) | "
            f"moved={rows[-1]['moved_frac']} | delta={rows[-1]['delta_mean']} gamma={rows[-1]['gamma_mean']}", fh)
    pd.DataFrame(rows).to_csv(TAB / "pilot.csv", index=False)
    json.dump(rows, open(TAB / "pilot.json", "w"), indent=2)
    # choose: smallest P with pg_ess_min >= ~30 and moved_frac reasonable, else largest
    choice = 512
    for r in rows:
        if r["pg_ess_min"] >= 30 and r["moved_frac"] >= 0.15:
            choice = r["particles"]; break
    log(f"Chosen particle count: {choice}", fh)
    json.dump({"chosen_particles": choice, "rows": rows}, open(TAB / "pilot_choice.json", "w"), indent=2)
    fh.close()
    return choice, rows


# --------------------------------------------------------------------------- #
# STAGE 3 -- production run + diagnostics + comparison
# --------------------------------------------------------------------------- #
SCAL_KEYS = ["alpha", "kappa_0", "delta", "theta_0", "gamma", "phi_1",
             "rho1", "rho2", "n", "sigma_e", "sigma_u", "sigma_eps", "sigma_N"]
PATH_KEYS = ["Nbar", "Nhat", "kappa_t", "theta_t"]


def stage_produce(cell, particles, from_cache=False, spec=CELL, existing_run=None):
    import arviz as az
    existing_run = existing_run if existing_run is not None else EXISTING_RUN
    fh = open(LOG / f"produce_{spec}.log", "w")
    log(f"=== PRODUCTION: Particle Gibbs hsa_full, {spec}, P={particles} ===", fh)
    log(f"MCMC: n_iter={N_ITER} burn={BURN} thin={THIN} chains={CHAINS} seed={SEED}", fh)
    npz_path = DRAWS / f"pg_hsa_full_{spec}_P{particles}.npz"

    if from_cache and npz_path.exists():
        log(f"Loading cached draws from {npz_path}", fh)
        d = np.load(npz_path)
        scal = {k: d[f"scal_{k}"] for k in SCAL_KEYS}
        paths = {k: d[f"path_{k}"] for k in PATH_KEYS}
        pg_ess_all = list(d["pg_ess"]); moved_all = list(d["moved"])
        runtime = float("nan")
    else:
        child_seeds = np.random.SeedSequence(SEED).spawn(CHAINS)
        scal = {k: [] for k in SCAL_KEYS}
        paths = {k: [] for k in PATH_KEYS}
        pg_ess_all, moved_all = [], []
        t0 = time.time()
        for ci, child in enumerate(child_seeds):
            cseed = int(child.generate_state(1)[0])
            res = func_nkpc_hsa_full_pg(
                cell["pi"], cell["pi_prev"], cell["pi_expect"], cell["x"], cell["x_prev"], cell["N"],
                n_burn=BURN, n_keep=N_ITER - BURN, priors=cell["pri_int"],
                opts={"seed": cseed, "store_every": THIN, "n_particles": particles, "verbose": True},
            )
            for k in scal:
                scal[k].append(res[k]["draws"])
            for k in paths:
                paths[k].append(res["state_draws"][k])
            pg_ess_all.append(res["pg_diagnostics"]["ess_mean"])
            moved_all.append(res["pg_diagnostics"]["moved_frac"])
            log(f"chain {ci} done ({time.time()-t0:.0f}s cumulative)", fh)
        runtime = time.time() - t0
        log(f"Total production runtime: {runtime:.0f}s", fh)
        scal = {k: np.array(v) for k, v in scal.items()}          # (chain, draw)
        paths = {k: np.array(v) for k, v in paths.items()}         # (chain, draw, T)
        np.savez_compressed(npz_path,
                            **{f"scal_{k}": v for k, v in scal.items()},
                            **{f"path_{k}": v for k, v in paths.items()},
                            pg_ess=np.array(pg_ess_all), moved=np.array(moved_all))

    def _rhat(a):
        return float(np.asarray(az.rhat(a)))

    def _ess(a, m):
        if m == "tail":
            return float(np.asarray(az.ess(a, method="tail", prob=(0.025, 0.975))))
        return float(np.asarray(az.ess(a, method=m)))

    diag = {}
    for k in scal:
        diag[k] = dict(
            mean=float(scal[k].mean()),
            q025=float(np.quantile(scal[k], 0.025)),
            q975=float(np.quantile(scal[k], 0.975)),
            rhat=_rhat(scal[k]),
            ess_bulk=_ess(scal[k], "bulk"),
            ess_tail=_ess(scal[k], "tail"),
        )
    path_diag = {}
    for k, arr in paths.items():
        rh = np.asarray(az.rhat(arr))
        eb = np.asarray(az.ess(arr, method="bulk"))
        path_diag[k] = dict(max_rhat=float(np.nanmax(rh)), min_ess_bulk=float(np.nanmin(eb)))

    pg_diag = dict(
        particles=particles,
        ess_mean_overall=float(np.mean(pg_ess_all)),
        ess_mean_min=float(np.min(pg_ess_all)),
        ess_per_particle=float(np.mean(pg_ess_all) / particles),
        moved_frac=float(np.mean(moved_all)),
    )

    # Corr(Nhat_t, Nbar_t*Nhat_t) and condition number of [Nhat, Nbar*Nhat] per draw
    Nhat = paths["Nhat"].reshape(-1, paths["Nhat"].shape[-1])
    Nbar = paths["Nbar"].reshape(-1, paths["Nbar"].shape[-1])
    corrs, conds = [], []
    for i in range(Nhat.shape[0]):
        h = Nhat[i]; hb = Nbar[i] * Nhat[i]
        if h.std() > 1e-9 and hb.std() > 1e-9:
            corrs.append(float(np.corrcoef(h, hb)[0, 1]))
        Xc = np.column_stack([h - h.mean(), hb - hb.mean()])
        s = np.linalg.svd(Xc, compute_uv=False)
        conds.append(float(s[0] / max(s[-1], 1e-12)))
    corrs = np.array(corrs); conds = np.array(conds)
    collin = dict(
        corr_Nhat_NbarNhat=dict(mean=float(corrs.mean()), q025=float(np.quantile(corrs, .025)), q975=float(np.quantile(corrs, .975))),
        condition_number=dict(mean=float(conds.mean()), median=float(np.median(conds)), q975=float(np.quantile(conds, .975))),
    )

    # comparison vs existing alternating-FFBS hsa_full
    comp = {}
    try:
        if existing_run is None:
            raise FileNotFoundError("no matching existing FFBS run")
        ex = az.from_netcdf(existing_run / "posterior.nc")
        post = ex.posterior
        for k in ["delta", "theta_0", "gamma"]:
            if k in post:
                v = post[k]
                comp[k] = dict(
                    existing_mean=float(v.mean()),
                    existing_q025=float(v.quantile(0.025)),
                    existing_q975=float(v.quantile(0.975)),
                    existing_rhat=_rhat(post[k]),
                    existing_ess_bulk=_ess(post[k], "bulk"),
                    pg_mean=diag[k]["mean"], pg_q025=diag[k]["q025"], pg_q975=diag[k]["q975"],
                    pg_rhat=diag[k]["rhat"], pg_ess_bulk=diag[k]["ess_bulk"],
                )
    except Exception as e:  # noqa: BLE001
        comp["error"] = str(e)

    summary = dict(
        cell=spec, particles=particles, runtime_seconds=round(runtime, 1),
        mcmc=dict(n_iter=N_ITER, burn=BURN, thin=THIN, chains=CHAINS, seed=SEED),
        scalar_diagnostics=diag, path_diagnostics=path_diag,
        particle_diagnostics=pg_diag, collinearity=collin,
        comparison_vs_existing_ffbs=comp,
        existing_run=str(existing_run.relative_to(ROOT)) if existing_run else None,
    )
    json.dump(summary, open(TAB / f"production_summary_{spec}_P{particles}.json", "w"), indent=2)
    log("delta:  PG mean=%.4f [%.4f,%.4f] Rhat=%.3f ESSbulk=%.0f" %
        (diag["delta"]["mean"], diag["delta"]["q025"], diag["delta"]["q975"], diag["delta"]["rhat"], diag["delta"]["ess_bulk"]), fh)
    log("gamma:  PG mean=%.4f [%.4f,%.4f] Rhat=%.3f ESSbulk=%.0f" %
        (diag["gamma"]["mean"], diag["gamma"]["q025"], diag["gamma"]["q975"], diag["gamma"]["rhat"], diag["gamma"]["ess_bulk"]), fh)
    log("theta0: PG mean=%.4f [%.4f,%.4f] Rhat=%.3f ESSbulk=%.0f" %
        (diag["theta_0"]["mean"], diag["theta_0"]["q025"], diag["theta_0"]["q975"], diag["theta_0"]["rhat"], diag["theta_0"]["ess_bulk"]), fh)
    log("path Rhat/ESS: " + json.dumps(path_diag), fh)
    log("particle: " + json.dumps(pg_diag), fh)
    log("collinearity: " + json.dumps(collin), fh)
    log("comparison vs existing FFBS: " + json.dumps(comp, indent=2), fh)

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for j, k in enumerate(["delta", "gamma", "theta_0"]):
        for c in range(CHAINS):
            ax[j].plot(scal[k][c], lw=0.6, alpha=0.8)
        ax[j].set_title(f"{k}  (Rhat={diag[k]['rhat']:.3f}, ESS={diag[k]['ess_bulk']:.0f})")
    fig.suptitle(f"Particle Gibbs traces ({spec}, P={particles})"); fig.tight_layout()
    fig.savefig(FIG / f"pg_traces_{spec}_P{particles}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    T = paths["Nbar"].shape[-1]; tt = np.arange(T)
    for k, axk in zip(["Nbar", "kappa_t"], ax):
        a = paths[k].reshape(-1, T)
        axk.plot(tt, a.mean(0), "b-", lw=2)
        axk.fill_between(tt, np.quantile(a, .025, 0), np.quantile(a, .975, 0), color="b", alpha=0.15)
        axk.set_title(f"{k} posterior (PG)")
    fig.tight_layout(); fig.savefig(FIG / f"pg_state_paths_{spec}_P{particles}.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(np.concatenate(pg_ess_all)); ax[0].set_title(f"particle ESS per iter (P={particles})")
    ax[0].axhline(np.mean(pg_ess_all), color="r", ls="--")
    ax[1].plot(np.concatenate(moved_all)); ax[1].set_title("state-path movement fraction per iter")
    fig.tight_layout(); fig.savefig(FIG / f"pg_particle_diag_{spec}_P{particles}.png", dpi=130); plt.close(fig)
    log("Saved figures + tables + draws to results/evidence/", fh)
    fh.close()
    return summary


def _chosen_particles(default=512):
    cj = TAB / "pilot_choice.json"
    return json.load(open(cj))["chosen_particles"] if cj.exists() else default


def stage_produce_all(particles, from_cache=False):
    """Run every baseline PCHIP hsa_full cell and build a cross-cell gamma table."""
    fh = open(LOG / "produce_all.log", "w")
    log(f"=== PRODUCE ALL baseline hsa_full cells (P={particles}) ===", fh)
    rows = []
    for spec in BASELINE_CELLS:
        log(f"\n##### CELL: {spec} #####", fh)
        cell = load_cell(spec)
        summ = stage_produce(cell, particles, from_cache=from_cache, spec=spec, existing_run=find_existing_run(spec))
        d = summ["scalar_diagnostics"]; c = summ["collinearity"]; pth = summ["path_diagnostics"]
        cmp = summ["comparison_vs_existing_ffbs"]

        def exval(k, f):
            v = cmp.get(k)
            return v.get(f) if isinstance(v, dict) else None

        gci = [d["gamma"]["q025"], d["gamma"]["q975"]]
        rows.append(dict(
            cell=spec,
            delta_mean=round(d["delta"]["mean"], 4), delta_lo=round(d["delta"]["q025"], 4), delta_hi=round(d["delta"]["q975"], 4),
            delta_ess_pg=round(d["delta"]["ess_bulk"], 0),
            delta_ess_ffbs=(round(exval("delta", "existing_ess_bulk"), 1) if exval("delta", "existing_ess_bulk") is not None else None),
            theta0_mean=round(d["theta_0"]["mean"], 4), theta0_lo=round(d["theta_0"]["q025"], 4), theta0_hi=round(d["theta_0"]["q975"], 4),
            gamma_mean=round(d["gamma"]["mean"], 4), gamma_lo=round(gci[0], 4), gamma_hi=round(gci[1], 4),
            gamma_excludes_zero=bool(gci[0] > 0 or gci[1] < 0),
            gamma_ess_pg=round(d["gamma"]["ess_bulk"], 0),
            gamma_ess_ffbs=(round(exval("gamma", "existing_ess_bulk"), 1) if exval("gamma", "existing_ess_bulk") is not None else None),
            gamma_rhat_pg=round(d["gamma"]["rhat"], 3),
            corr_Nhat_NbarNhat=round(c["corr_Nhat_NbarNhat"]["mean"], 3),
            cond_number=round(c["condition_number"]["mean"], 2),
            Nbar_path_max_rhat=round(pth["Nbar"]["max_rhat"], 3),
            Nbar_path_min_ess=round(pth["Nbar"]["min_ess_bulk"], 1),
        ))
        pd.DataFrame(rows).to_csv(TAB / "cross_cell_summary.csv", index=False)  # incremental
        json.dump(rows, open(TAB / "cross_cell_summary.json", "w"), indent=2)
        log(f"[{spec}] delta={rows[-1]['delta_mean']} [{rows[-1]['delta_lo']},{rows[-1]['delta_hi']}] "
            f"gamma={rows[-1]['gamma_mean']} [{rows[-1]['gamma_lo']},{rows[-1]['gamma_hi']}] "
            f"excl0={rows[-1]['gamma_excludes_zero']} corr={rows[-1]['corr_Nhat_NbarNhat']} "
            f"deltaESS pg/ffbs={rows[-1]['delta_ess_pg']}/{rows[-1]['delta_ess_ffbs']}", fh)
    log("\nCross-cell summary -> tables/cross_cell_summary.{csv,json}", fh)
    fh.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["validate", "pilot", "produce", "produce-all", "all"])
    ap.add_argument("--particles", type=int, default=None)
    ap.add_argument("--cell", default=None, help="data spec for single produce (default core-CPI unemployment)")
    ap.add_argument("--from-cache", action="store_true", help="reuse saved draws; skip MCMC")
    args = ap.parse_args()

    if args.stage == "produce-all":
        stage_produce_all(int(args.particles or _chosen_particles()), from_cache=args.from_cache)
        return

    cell = load_cell()
    if args.stage in ("validate", "all"):
        stage_validate(cell)
    choice = args.particles
    if args.stage in ("pilot", "all"):
        choice, _ = stage_pilot(cell)
    if args.stage in ("produce", "all"):
        choice = int(choice) if choice is not None else _chosen_particles()
        spec = args.cell or CELL
        c = load_cell(spec) if spec != CELL else cell
        stage_produce(c, choice, from_cache=args.from_cache, spec=spec, existing_run=find_existing_run(spec))


if __name__ == "__main__":
    main()
