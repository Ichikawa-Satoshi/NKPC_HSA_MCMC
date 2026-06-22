from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from nkpc_hsa.inference.wrappers import run_model
from nkpc_hsa.paths import project_path


def run_prior_sensitivity(
    model: str,
    *,
    data=None,
    data_spec: Mapping | None = None,
    prior_files: list[str | Path] | None = None,
    n_iter: int = 12000,
    burn: int = 4000,
    thin: int = 5,
    chains: int = 2,
    seed: int = 12345,
    n_transform: str = "log100",
    covariance_structure: str = "e_zeta_only",
    coefficient_constraints: Mapping | None = None,
):
    prior_files = prior_files or [
        project_path("configs", "priors_baseline.yaml"),
        project_path("configs", "priors_weak.yaml"),
        project_path("configs", "priors_tight.yaml"),
    ]
    outputs = {}
    for i, prior_path in enumerate(prior_files):
        name = Path(prior_path).stem.replace("priors_", "")
        idata = run_model(
            model,
            data=data,
            data_spec=data_spec,
            prior_specs=prior_path,
            prior_name=name,
            n_iter=n_iter,
            burn=burn,
            thin=thin,
            chains=chains,
            seed=seed + i,
            n_transform=n_transform,
            covariance_structure=covariance_structure,
            coefficient_constraints=coefficient_constraints,
        )
        outputs[name] = idata
    return outputs


DEFAULT_PRIOR_ROBUSTNESS_PARAMETERS = (
    "alpha",
    "kappa",
    "kappa_0",
    "delta",
    "theta",
    "theta_0",
    "gamma",
    "rho_1",
    "rho_2",
    "sigma_e",
    "sigma_zeta",
    "sigma_u",
    "sigma_eps",
    "sigma_N",
)


def prior_sensitivity_table(idata_map: Mapping[str, object], var_names=DEFAULT_PRIOR_ROBUSTNESS_PARAMETERS) -> pd.DataFrame:
    rows = []
    for prior_name, idata in idata_map.items():
        posterior = getattr(idata, "posterior", None)
        if posterior is None:
            continue
        for var in var_names:
            if var not in posterior:
                continue
            values = posterior[var].values.reshape(-1)
            values = values[pd.notna(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "prior": prior_name,
                    "parameter": var,
                    "mean": float(pd.Series(values).mean()),
                    "median": float(pd.Series(values).median()),
                    "ci_2.5": float(pd.Series(values).quantile(0.025)),
                    "ci_97.5": float(pd.Series(values).quantile(0.975)),
                    "p_gt_0": float((values > 0.0).mean()),
                    "p_lt_0": float((values < 0.0).mean()),
                }
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    conclusion_rows = []
    for param in ["delta", "theta", "theta_0", "gamma"]:
        sub = table[table["parameter"] == param]
        if sub.empty:
            continue
        signs = sub["p_gt_0"].apply(lambda p: "positive" if p >= 0.95 else "negative" if p <= 0.05 else "ambiguous")
        robust = signs.nunique() == 1
        conclusion_rows.append(
            {
                "prior": "all",
                "parameter": f"{param}_robustness",
                "mean": float("nan"),
                "median": float("nan"),
                "ci_2.5": float("nan"),
                "ci_97.5": float("nan"),
                "p_gt_0": float("nan"),
                "p_lt_0": float("nan"),
                "conclusion": "robust" if robust else "sensitive",
            }
        )
    table["conclusion"] = ""
    if conclusion_rows:
        table = pd.concat([table, pd.DataFrame(conclusion_rows)], ignore_index=True)
    return table


def save_prior_sensitivity_overlays(
    idata_map: Mapping[str, object],
    out_dir: str | Path,
    *,
    var_names=DEFAULT_PRIOR_ROBUSTNESS_PARAMETERS,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for var in var_names:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        plotted = False
        for prior_name, idata in idata_map.items():
            posterior = getattr(idata, "posterior", None)
            if posterior is None or var not in posterior:
                continue
            values = posterior[var].values.reshape(-1)
            values = values[pd.notna(values)]
            if values.size < 3 or float(values.std(ddof=1)) <= 0.0:
                continue
            grid = pd.Series(values).quantile([0.005, 0.995]).to_numpy(dtype=float)
            xs = np.linspace(grid[0], grid[1], 300)
            kde = gaussian_kde(values)
            ax.plot(xs, kde(xs), label=prior_name)
            plotted = True
        if plotted:
            ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
            ax.set_title(f"Prior sensitivity: {var}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out / f"prior_sensitivity_{var}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
