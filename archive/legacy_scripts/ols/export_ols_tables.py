from __future__ import annotations

from pathlib import Path
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TEX_DIR = PROJECT_ROOT / "results" / "tex" / "table" / "ols"
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(SRC_DIR))
from analysis.src.func_data_build import build_dataset  # noqa: E402


PERIODS = [
    ("1982-01-01", "2012-12-31"),
    ("1982-01-01", "1992-12-31"),
    ("1993-01-01", "2002-12-31"),
    ("2003-01-01", "2012-12-31"),
]

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

X_SPECS = {
    "unemp_gap": {
        "column": "unemp_gap",
        "label": "Unemployment Gap",
        "slug": "unempgap",
    },
    "output_gap_BN": {
        "column": "output_gap_BN",
        "label": "GDPC1 (BN)",
        "slug": "outputgapbn",
    },
    "markup_BN_inv": {
        "column": "markup_BN_inv",
        "label": "Inverse of Markup (BN)",
        "slug": "markupbninv",
    },
}


@dataclass
class OLSResult:
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    nobs: int
    rsquared: float
    rsquared_adj: float


def fit_ols(y: pd.Series, X: pd.DataFrame) -> OLSResult:
    X_mat = X.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)
    nobs, nvars = X_mat.shape

    xtx_inv = np.linalg.inv(X_mat.T @ X_mat)
    beta = xtx_inv @ (X_mat.T @ y_vec)
    fitted = X_mat @ beta
    resid = y_vec - fitted

    sse = float(resid @ resid)
    y_centered = y_vec - np.mean(y_vec)
    sst = float(y_centered @ y_centered)
    rsquared = 1.0 - sse / sst if sst > 0 else np.nan
    rsquared_adj = 1.0 - (1.0 - rsquared) * (nobs - 1) / (nobs - nvars) if nobs > nvars else np.nan

    sigma2 = sse / (nobs - nvars)
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    pvalues = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=nobs - nvars))

    return OLSResult(
        params=pd.Series(beta, index=X.columns, dtype=float),
        bse=pd.Series(se, index=X.columns, dtype=float),
        pvalues=pd.Series(pvalues, index=X.columns, dtype=float),
        nobs=nobs,
        rsquared=rsquared,
        rsquared_adj=rsquared_adj,
    )


def _stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def fit_models(sub: pd.DataFrame, y_col: str) -> tuple[OLSResult, OLSResult]:
    sub = sub.copy()
    sub["pi_prev_minus_Epi"] = sub["pi_prev"] - sub["Epi"]
    y = sub[y_col] - sub["Epi"]

    x0 = pd.DataFrame(
        {
            "const": 1.0,
            "pi_prev_minus_Epi": sub["pi_prev_minus_Epi"],
            "x2": sub["x2"],
        }
    )
    m0 = fit_ols(y, x0)

    x1 = pd.DataFrame(
        {
            "const": 1.0,
            "pi_prev_minus_Epi": sub["pi_prev_minus_Epi"],
            "x2": sub["x2"],
            "Nhat": sub["Nhat"],
        }
    )
    m1 = fit_ols(y, x1)
    return m0, m1


def make_stata_table(models_dict: dict[str, OLSResult], *, include_period_note: bool = False) -> str:
    reg_order = ["const", "pi_prev_minus_Epi", "x2", "Nhat"]
    reg_labels = {
        "const": "Constant",
        "pi_prev_minus_Epi": r"$\alpha$",
        "x2": r"$\kappa$",
        "Nhat": r"$\gamma$",
    }

    columns = list(models_dict.keys())
    rows: list[list[str]] = []

    for reg in reg_order:
        coef_row = [reg_labels[reg]]
        se_row = [""]
        for col in columns:
            result = models_dict[col]
            if reg in result.params.index:
                coef = result.params[reg]
                coef_row.append(f"{coef:.4f}{_stars(float(result.pvalues[reg]))}")
                se_row.append(f"({result.bse[reg]:.4f})")
            else:
                coef_row.append("")
                se_row.append("")
        rows.append(coef_row)
        rows.append(se_row)

    info_rows = [
        ["Adj. R$^2$"] + [f"{models_dict[col].rsquared_adj:.3f}" for col in columns],
        ["N"] + [f"{models_dict[col].nobs:d}" for col in columns],
        ["R$^2$"] + [f"{models_dict[col].rsquared:.3f}" for col in columns],
    ]

    lines = [
        r"\begin{table}",
        r"\begin{center}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * (len(columns) + 1) + "}",
        r"\hline",
        "               & " + " & ".join(columns) + r"  \\",
        r"\hline",
    ]
    for row in rows + info_rows:
        lines.append("               " + " & ".join(row) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{center}",
            r"\end{table}",
            r"\bigskip",
        ]
    )
    if include_period_note:
        lines.extend(
            [
                r"{\tiny Notes: Standard errors in parentheses. * p<.1, ** p<.05, *** p<.01.}",
                r"{\tiny Periods: P1 = 1982Q1--2012Q4, P2 = 1982Q1--1992Q4, P3 = 1993Q1--2002Q4, P4 = 2003Q1--2012Q4.}",
                r"{\tiny Columns labeled ``No $N_t^c$'' exclude the competition term; ``With $N_t^c$'' include $N_t^c$.}",
            ]
        )
    else:
        lines.extend(
            [
                r"{\tiny Notes: Standard errors in parentheses. * p<.1, ** p<.05, *** p<.01.}",
            ]
        )
    return "\n".join(lines)


def export_tables() -> None:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    data = build_dataset(DATA_DIR)

    for infl, spec in INFLATION_SPECS.items():
        for x_key, x_spec in X_SPECS.items():
            y = data[spec["pi"]].dropna()
            X = pd.DataFrame(
                {
                    "pi_prev": data[spec["pi_prev"]],
                    "Epi": data[spec["pi_expect"]],
                    "x2": data[x_spec["column"]],
                    "Nhat": data["N_BN_cycle"],
                }
            )
            df = pd.concat([y.rename(spec["pi"]), X], axis=1).dropna()

            all_models: dict[str, OLSResult] = {}
            for i, (start, end) in enumerate(PERIODS, 1):
                sub = df.loc[start:end].dropna()
                m0, m1 = fit_models(sub, spec["pi"])
                all_models[f"P{i} No $N_t^c$"] = m0
                all_models[f"P{i} With $N_t^c$"] = m1

                period_models = {
                    "Without $N_t^c$": m0,
                    "With $N_t^c$": m1,
                }
                period_table = make_stata_table(period_models, include_period_note=False)
                period_path = TEX_DIR / f"ols_{infl}_{x_spec['slug']}_period{i}_stata_style.tex"
                period_path.write_text(period_table, encoding="utf-8")
                print(f"Saved {period_path}")

            all_table = make_stata_table(all_models, include_period_note=True)
            all_path = TEX_DIR / f"ols_{infl}_{x_spec['slug']}_stata_style_all.tex"
            all_path.write_text(all_table, encoding="utf-8")
            print(f"Saved {all_path}")


if __name__ == "__main__":
    export_tables()
