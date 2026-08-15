"""The report's specification tables, generated from ``configs/``.

Model variants, the baseline prior block, and the three prior sets the robustness
section sweeps. These were previously typed into the .tex by hand, which is how a
prior table can silently disagree with the YAML the sampler actually loaded --
every number here is read from the config files at build time instead.

    python production/main_scripts/make_spec_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_DIR, ROOT

from nkpc_hsa.reporting.cpi_ppi_spec import MODEL_LABELS, MODEL_ORDER

TABLES = RESULTS_DIR / "tables" / "shared"

MODEL_VARIANTS = {
    "ces": ("constant", "$0$", "Textbook NKPC benchmark"),
    "hsa_steady": (r"$\kappa_0+\delta\bar N_t$", "$0$", r"$\bar N$ channel only"),
    "hsa_dynamic": ("constant", "constant", r"$\hat N$ channel only"),
    "hsa_const_theta": (r"$\kappa_0+\delta\bar N_t$", "constant", r"HSA full with $\gamma=0$"),
    "hsa_full": (r"$\kappa_0+\delta\bar N_t$", r"$\theta_0+\gamma\bar N_t$", "HSA full"),
}
SAMPLER_TEXT = {
    "ces": "Gibbs (conjugate)",
    "hsa_steady": "Gibbs + exact joint FFBS",
    "hsa_dynamic": "Gibbs + exact joint FFBS",
    "hsa_const_theta": "Gibbs + exact joint FFBS",
    "hsa_full": "Gibbs + Particle Gibbs",
}


def _tabular(colspec: str, header: list[str], rows: list[list[str]]) -> str:
    out = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"


def write_model_variants_table() -> None:
    rows = [
        [MODEL_LABELS[m], MODEL_VARIANTS[m][0], MODEL_VARIANTS[m][1], MODEL_VARIANTS[m][2], SAMPLER_TEXT[m]]
        for m in MODEL_ORDER
    ]
    (TABLES / "model_variants.tex").write_text(
        _tabular("lllll", [r"Model", r"$\kappa_t$", r"$\theta_t$", "Role", "State block"], rows),
        encoding="utf-8",
    )


def _norm(pair) -> str:
    mean, sd = float(pair[0]), float(pair[1])
    mean_s = f"{mean:g}"
    return rf"$N({mean_s},{sd:g}^2)$"


def _ig(a, b) -> str:
    return rf"$\operatorname{{IG}}({float(a):g},{float(b):g})$"


PRIOR_BLOCKS = [
    ("Inflation eq.", "alpha", r"$\alpha$", "Backward inertia"),
    (None, "kappa", r"$\kappa,\kappa_0$", "Slope level"),
    (None, "delta", r"$\delta$", "Slope dependence on firm count"),
    (None, "theta", r"$\theta,\theta_0$", "Cyclical entry effect"),
    (None, "gamma", r"$\gamma$", "Time-varying coefficient of entry"),
    ("Activity eq.", "phi_1", r"$\phi_1$", "AR(1) of the activity variable"),
    (None, "lambda", r"$\lambda_{e\zeta}$", "Simultaneous-shock correction"),
    ("Firm-count state", "rho_1", r"$\rho_1$", "Cycle AR(2), truncated to stationary region"),
    (None, "rho_2", r"$\rho_2$", "Cycle AR(2), truncated to stationary region"),
    (None, "n", "$n$", "Trend drift"),
]
# lambda is not written in the YAML; the sampler falls back to this. Stated
# explicitly so the table matches what is actually used.
LAMBDA_DEFAULT = [0.0, 0.5]


def write_prior_tables() -> None:
    sets = {name: yaml.safe_load((ROOT / "configs" / f"priors_{name}.yaml").read_text(encoding="utf-8"))
            for name in ("baseline", "weak", "tight")}
    base = sets["baseline"]

    rows = []
    for block, key, symbol, note in PRIOR_BLOCKS:
        pair = LAMBDA_DEFAULT if key == "lambda" else base.get(key)
        if pair is None:
            continue
        rows.append([block or "", symbol, _norm(pair), note])
    rows.append([r"\midrule Variances", r"$\sigma_\eta^2$", _ig(base["a_e"], base["b_e"]), "Inflation shock"])
    rows.append(["", r"$\sigma_\zeta^2$", _ig(base["a_z"], base["b_z"]), "Activity shock"])
    rows.append(["", r"$\sigma_u^2$", _ig(base["a_u"], base["b_u"]), "Cycle shock"])
    rows.append(["", r"$\sigma_\varepsilon^2$", _ig(base["a_eps"], base["b_eps"]), "Trend shock"])
    rows.append(["", r"$\sigma_N^2$", _ig(base["a_N"], base["b_N"]), "Firm-count measurement error"])
    (TABLES / "baseline_priors.tex").write_text(
        _tabular("llll", ["Block", "Parameter", "Prior", "Interpretation"], rows), encoding="utf-8"
    )

    # The three prior sets side by side. The report sweeps these, so what each
    # one actually changes has to be visible rather than described.
    def ig_mean(a, b) -> str:
        a, b = float(a), float(b)
        return "undefined" if a <= 1.0 else f"{b / (a - 1.0):g}"

    coef_rows = []
    for key, symbol in [("alpha", r"$\alpha$"), ("kappa", r"$\kappa,\kappa_0$"), ("delta", r"$\delta$"),
                        ("theta", r"$\theta,\theta_0$"), ("gamma", r"$\gamma$"), ("phi_1", r"$\phi_1$"),
                        ("rho_1", r"$\rho_1$"), ("rho_2", r"$\rho_2$"), ("n", "$n$")]:
        coef_rows.append([symbol] + [_norm(sets[s][key]) for s in ("baseline", "weak", "tight")])
    for label, a_key, b_key in [(r"$\sigma_u^2$", "a_u", "b_u"), (r"$\sigma_\varepsilon^2$", "a_eps", "b_eps"),
                                (r"$\sigma_N^2$", "a_N", "b_N"), (r"$\sigma_\eta^2$", "a_e", "b_e")]:
        coef_rows.append([label] + [
            _ig(sets[s][a_key], sets[s][b_key]) + rf" \tiny(mean {ig_mean(sets[s][a_key], sets[s][b_key])})"
            for s in ("baseline", "weak", "tight")
        ])
    (TABLES / "prior_sets.tex").write_text(
        _tabular("lccc", ["Parameter", "baseline", "weak", "tight"], coef_rows), encoding="utf-8"
    )
    print(f"wrote {(TABLES / 'model_variants.tex').relative_to(ROOT)}, "
          f"{(TABLES / 'baseline_priors.tex').relative_to(ROOT)}, "
          f"{(TABLES / 'prior_sets.tex').relative_to(ROOT)}")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    write_prior_tables()
    write_model_variants_table()


if __name__ == "__main__":
    main()
