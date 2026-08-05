from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PARAMETER_UNITS = {
    "alpha": "share",
    "kappa": "inflation points per x point",
    "kappa_0": "inflation points per x point at average Nbar",
    "kappa_t": "inflation points per x point",
    "delta": "change in kappa_t per +10 log-point Nbar deviation",
    "theta": "inflation effect per +10 log-point Nhat deviation",
    "theta_0": "inflation effect per +10 log-point Nhat deviation at average Nbar",
    "gamma": "change in theta_t per +10 log-point Nbar deviation",
    "rho_1": "AR(2) coefficient",
    "rho_2": "AR(2) coefficient",
    "phi_1": "AR(1) coefficient",
    "n": "Nbar drift in ten-log-point units",
    "lambda_ez": "shock loading",
}


def parameter_unit(name: str) -> str:
    return PARAMETER_UNITS.get(name, "reported physical units")


def posterior_summary_table(idata, *, var_names: Iterable[str] | None = None) -> pd.DataFrame:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return pd.DataFrame()
    names = list(var_names or posterior.data_vars)
    rows = []
    for name in names:
        if name not in posterior:
            continue
        values = np.asarray(posterior[name]).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "parameter": name,
                "unit": parameter_unit(name),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "ci_2.5": float(np.quantile(values, 0.025)),
                "ci_97.5": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


COEFFICIENT_PARAMETERS = (
    "alpha",
    "kappa",
    "kappa_0",
    "delta",
    "theta",
    "theta_0",
    "gamma",
    "phi_1",
    "rho_1",
    "rho_2",
    "n",
    "lambda_ez",
)


def write_latex_fragment(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(df.to_latex(index=index, float_format="%.4f", escape=True), encoding="utf-8")
