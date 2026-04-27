from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

from func_data_build import build_dataset
from func_gibbs.gibbs_marginal_likelihood import (
    chib_conditional_marginal_likelihood,
    chib_marginal_likelihood,
    load_posterior_dataset,
)


GAP_SPECS = (
    ("output_gap_BN", "GDPC1 (BN)"),
    ("unemp_gap", "Unemployment Gap"),
    ("markup_BN_inv", "Inverse of Markup (BN)"),
    ("markup_inv", "Inverse of Markup"),
)

INFLATION_SPECS = {
    "ppi": {
        "pi": "pi_ppi",
        "pi_prev": "pi_ppi_prev",
        "pi_expect": "Epi_spf_gdp",
    },
    "cpi": {
        "pi": "pi_cpi",
        "pi_prev": "pi_cpi_prev",
        "pi_expect": "Epi_spf_cpi",
    },
}

FAMILY_SPECS = (
    ("ces", "CES", "gibbs_ces"),
    ("dynamic", "HSA dynamic", "gibbs_hsa_dynamic"),
    ("steady", "HSA steady", "gibbs_hsa_steady"),
)

PERIOD_SPEC = {
    "label": "1988_2017",
    "start": "1988-03-31",
    "end": "2017-12-31",
    "n_col": "N_TNIC",
}


def _data_for_model(data: pd.DataFrame, infl: str, gap: str) -> dict[str, np.ndarray]:
    spec = INFLATION_SPECS[infl]
    out = {
        "pi": np.asarray(data[spec["pi"]], dtype=float),
        "pi_prev": np.asarray(data[spec["pi_prev"]], dtype=float),
        "pi_expect": np.asarray(data[spec["pi_expect"]], dtype=float),
        "x": np.asarray(data[gap], dtype=float),
        "x_prev": np.asarray(data[f"{gap}_prev"], dtype=float),
    }
    if "N" in data:
        out["N"] = np.asarray(data["N"], dtype=float)
    return out


def _model_path(*, family: str, infl: str, gap: str, orth: bool, period_label: str) -> Path:
    corr = "uncorr" if orth else "corr"
    if family == "ces":
        return (
            REPO_ROOT
            / "results"
            / "idata"
            / f"gibbs_ces_{infl}"
            / f"{infl}_ces_{gap.lower()}_{period_label}_{corr}.nc"
        )
    return (
        REPO_ROOT
        / "results"
        / "idata"
        / f"gibbs_hsa_{family}_{period_label}_{infl}"
        / f"{infl}_hsa_{gap.lower()}_{period_label}_{corr}.nc"
    )


def _format_table(df: pd.DataFrame, infl: str) -> str:
    sub = df[df["inflation"] == infl].copy()
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Output gap & Uncorr. & Rank Uncorr. & Corr. & Rank Corr. \\",
        r"\midrule",
    ]
    for gap, gap_label in GAP_SPECS:
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{gap_label}}}}} \\")
        for family_label in [x[1] for x in FAMILY_SPECS]:
            line = [family_label, ""]
            for orth, label in [(True, "Uncorr."), (False, "Corr.")]:
                hit = sub[(sub["family"] == family_label) & (sub["gap"] == gap) & (sub["orthogonal"] == orth)]
                if hit.empty:
                    line.extend(["--", "--"])
                else:
                    line.extend([f"{float(hit['log_marginal_likelihood'].iloc[0]):.2f}", f"{int(hit['rank'].iloc[0])}"])
            lines.append(" & ".join(line) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = build_dataset(REPO_ROOT / "data")
    data = data.loc[PERIOD_SPEC["start"]:PERIOD_SPEC["end"]].copy()
    data["N"] = np.asarray(data[PERIOD_SPEC["n_col"]], dtype=float)

    rows = []
    conditional_rows = []
    for infl in INFLATION_SPECS:
        for family, family_label, _family_dir in FAMILY_SPECS:
            for gap, gap_label in GAP_SPECS:
                model_data = _data_for_model(data, infl, gap)
                for orth in (True, False):
                    path = _model_path(
                        family=family,
                        infl=infl,
                        gap=gap,
                        orth=orth,
                        period_label=PERIOD_SPEC["label"],
                    )
                    if not path.exists():
                        raise FileNotFoundError(path)
                    print(f"Computing Chib marginal likelihood: {path}", flush=True)
                    ds = load_posterior_dataset(path)
                    res = chib_marginal_likelihood(ds, model_data, family=family, orth=orth)
                    cond_res = chib_conditional_marginal_likelihood(ds, model_data, family=family)
                    rows.append(
                        {
                            "inflation": infl.upper(),
                            "family": family_label,
                            "gap": gap,
                            "gap_label": gap_label,
                            "orthogonal": orth,
                            "covariance": "Uncorr." if orth else "Corr.",
                            "log_marginal_likelihood": res.log_marginal_likelihood,
                            "log_likelihood": res.log_likelihood,
                            "log_prior": res.log_prior,
                            "log_posterior_ordinate": res.log_posterior_ordinate,
                            "n_draws": res.n_draws,
                            "method": res.method,
                            "idata_path": str(path.relative_to(REPO_ROOT)),
                        }
                    )
                    conditional_rows.append(
                        {
                            "inflation": infl.upper(),
                            "family": family_label,
                            "gap": gap,
                            "gap_label": gap_label,
                            "orthogonal": orth,
                            "covariance": "Uncorr." if orth else "Corr.",
                            "log_marginal_likelihood": cond_res.log_marginal_likelihood,
                            "log_likelihood": cond_res.log_likelihood,
                            "log_prior": cond_res.log_prior,
                            "log_posterior_ordinate": cond_res.log_posterior_ordinate,
                            "n_draws": cond_res.n_draws,
                            "method": cond_res.method,
                            "idata_path": str(path.relative_to(REPO_ROOT)),
                        }
                    )

    out_dir = REPO_ROOT / "results" / "tex" / "table" / "gibbs_marginal_likelihood"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gibbs_chib_marginal_likelihood.csv"
    out_tex_all = out_dir / "gibbs_chib_marginal_likelihood_long.tex"
    out_cond_csv = out_dir / "gibbs_chib_conditional_marginal_likelihood.csv"
    out_cond_tex_all = out_dir / "gibbs_chib_conditional_marginal_likelihood_long.tex"
    df = pd.DataFrame(rows)
    df["rank_global"] = (
        df.groupby("inflation")["log_marginal_likelihood"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    df["rank"] = (
        df.groupby(["inflation", "gap"])["log_marginal_likelihood"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    df = df.sort_values(["inflation", "gap", "rank", "family", "orthogonal"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    df.to_latex(out_tex_all, index=False, float_format="%.4f", escape=False)

    cond_df = pd.DataFrame(conditional_rows)
    cond_df["rank_global"] = (
        cond_df.groupby("inflation")["log_marginal_likelihood"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    cond_df["rank"] = (
        cond_df.groupby(["inflation", "gap"])["log_marginal_likelihood"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    cond_df = cond_df.sort_values(["inflation", "gap", "rank", "family", "orthogonal"]).reset_index(drop=True)
    cond_df.to_csv(out_cond_csv, index=False)
    cond_df.to_latex(out_cond_tex_all, index=False, float_format="%.4f", escape=False)

    for infl in ("CPI", "PPI"):
        (out_dir / f"gibbs_chib_marginal_likelihood_{infl.lower()}.tex").write_text(
            _format_table(df, infl),
            encoding="utf-8",
        )
        (out_dir / f"gibbs_chib_conditional_marginal_likelihood_{infl.lower()}.tex").write_text(
            _format_table(cond_df, infl),
            encoding="utf-8",
        )
    print(f"written: {out_csv}", flush=True)
    print(f"written: {out_tex_all}", flush=True)
    print(f"written: {out_cond_csv}", flush=True)
    print(f"written: {out_cond_tex_all}", flush=True)


if __name__ == "__main__":
    main()
