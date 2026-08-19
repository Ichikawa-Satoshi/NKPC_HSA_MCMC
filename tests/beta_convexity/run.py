"""Estimate the hybrid-NKPC (beta_b, beta_f) restriction and compare to baseline.

For each cell and each of {baseline, convexity, adding_up} it fits the modular-cut
QoQ E2 equation and reports how the backward/forward split and the HSA slope delta
move under the restriction (review item §4.2). The unconstrained default estimator
is untouched; this bundle only adds the constrained specifications.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys as _sys, pathlib as _pathlib  # noqa: E402  (bootstrap: importable at any depth)
_ROOT = next(_p for _p in _pathlib.Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
_sys.path[:0] = [str(_ROOT), str(_ROOT / "src"), str(_ROOT / "tests")]
from tests import _bootstrap  # noqa: F401,E402

from nkpc_hsa.config import load_yaml
from nkpc_hsa.phillips.data import load_design_data, robust_scale
from nkpc_hsa.phillips.estimation import _save_fit
from nkpc_hsa.phillips.inflation import reference_draws
from nkpc_hsa.phillips.markup_measurement import sample_markup_measurement_posterior
from tests.beta_convexity.functions import CONSTRAINTS, delta_summary, fit_hybrid_restricted

BUNDLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BUNDLE_DIR / "results"


def _plot(summary: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    cells = summary["label"].unique()
    fig, ax = plt.subplots(figsize=(1.6 + 1.9 * len(cells), 4.0))
    offsets = {"baseline": -0.25, "convexity": 0.0, "adding_up": 0.25}
    colors = {"baseline": "#6B7280", "convexity": "#0072B2", "adding_up": "#D55E00"}
    for constraint, dx in offsets.items():
        rows = summary[summary["constraint"] == constraint]
        xs = np.arange(len(cells)) + dx
        ax.errorbar(
            xs, rows["delta_mean"],
            yerr=[rows["delta_mean"] - rows["delta_ci_2_5"], rows["delta_ci_97_5"] - rows["delta_mean"]],
            fmt="o", capsize=3, color=colors[constraint], label=constraint,
        )
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(cells)), cells, rotation=30, ha="right")
    ax.set_ylabel(r"HSA slope $\delta$ (posterior mean, 95% CI)")
    ax.set_title("Hybrid restriction and the HSA slope")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "delta_by_restriction.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE_DIR / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-draws", dest="save_draws", action="store_false")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = args.output_dir
    tables, figures, draws_dir = out / "tables", out / "figures", out / "draws"
    for directory in (tables, figures, draws_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sampling = dict(cfg["sampling"])
    if args.quick:
        sampling.update(iterations=400, warmup=150, thin=2, chains=2)

    data = load_design_data(
        include_qcew=False,
        sample_start=str(cfg["sample"]["start"]),
        sample_end=str(cfg["sample"]["end"]),
    )
    inverse_markup = data.quarterly["x_inverse_markup"].to_numpy(float)
    proxy = inverse_markup - float(np.mean(inverse_markup))
    posterior = sample_markup_measurement_posterior(
        data.annual_observation,
        proxy,
        q_scale=data.q_scale,
        proxy_scale=robust_scale(proxy),
        periods=tuple(map(str, data.periods)),
        markup_error="iid",
        iterations=int(sampling["iterations"]),
        warmup=int(sampling["warmup"]),
        thin=int(sampling["thin"]),
        chains=int(sampling["chains"]),
        seed=int(sampling["seed"]),
    )
    _, q0, _ = reference_draws(data, posterior)
    if args.save_draws:
        # Save the competition-state draws so the report can draw blocks 3-4
        # (time-varying kappa_t and the qbar/qhat decomposition). Shapes align
        # with the fit coefficient draws (chain x draw [x T]).
        np.savez_compressed(
            draws_dir / "state.npz",
            qbar=posterior.draws["qbar"],
            qhat=posterior.draws["qhat"],
            periods=np.asarray(tuple(map(str, data.periods))),
            q0=float(q0),
        )
    mu = float(data.config["benchmark"]["mu_reference"])
    zeta = mu / (mu - 1.0)
    b_x = float(data.config["benchmark"]["b_x"])
    model = str(cfg["inflation"]["model"])

    rows: list[dict[str, object]] = []
    for cell, price, activity in ((tuple(c) for c in cfg["cells"])):
        label = f"{price}/{activity}"
        for constraint in CONSTRAINTS:
            fit = fit_hybrid_restricted(
                data, posterior,
                cell=int(cell), model=model, q0=q0,
                seed=int(sampling["seed"]) + 500_009,
                constraint=constraint,
                price_override=str(price), activity_override=str(activity),
            )
            if args.save_draws:
                _save_fit(draws_dir / f"{price}_{activity}_{model}_{constraint}.npz", fit)
            summary = delta_summary(fit, b_x=b_x, zeta_reference=zeta)
            summary.update(label=label, cell=int(cell), price=price, activity=activity,
                           constraint=constraint, model=model)
            rows.append(summary)
            print(f"[{label}] {constraint}: beta_b+beta_f={summary['beta_sum_mean']:.3f} "
                  f"delta={summary['delta_mean']:.4f} binding={summary['restriction_binding_share']:.2%}",
                  flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "restriction_comparison.csv", index=False)
    _plot(summary, figures)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": str(args.config),
        "sampling": sampling,
        "model": model,
        "b_x": b_x, "zeta_reference": zeta,
        "constraints": list(CONSTRAINTS),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {tables / 'restriction_comparison.csv'}")


if __name__ == "__main__":
    main()
