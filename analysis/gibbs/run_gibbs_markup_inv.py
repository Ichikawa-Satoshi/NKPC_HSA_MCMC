from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

from func_data_build import build_dataset
from func_gibbs.gibbs_notebook_utils import format_sample_window, prepare_gibbs_sample, save_idata_map
from func_gibbs.gibbs_wrappers import draws_to_idata, run_ces, run_hsa_dynamic, run_hsa_steady


N_ITER = 12000
BURN = 4000
THIN = 5
SEED = 0
X_NAME = "markup_inv"


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

N_SPECS = {
    "gustavo": "N_Gustavo",
    "tnic": "N_TNIC",
}

CES_PRIORS = {
    "alpha": (0.5, 0.2),
    "kappa": (0.1, 0.2),
    "phi_1": (0.7, 0.2),
    "rho": (0.0, 0.5),
}

DYNAMIC_PRIORS = {
    "alpha": (0.5, 0.2),
    "kappa": (0.1, 0.2),
    "theta": (0.1, 0.2),
    "phi_1": (0.7, 0.2),
    "rho_1": (0.2, 0.2),
    "rho_2": (0.2, 0.2),
}

STEADY_PRIORS = {
    "alpha": (0.5, 0.2),
    "kappa_0": (0.1, 0.2),
    "delta": (0.1, 0.2),
    "phi_1": (0.7, 0.2),
    "rho_1": (0.2, 0.2),
    "rho_2": (0.2, 0.2),
}


def _load_data():
    data = build_dataset(REPO_ROOT / "data")
    data = data.loc["1982-01-01":"2012-12-31"].copy()
    data["DATE"] = pd.to_datetime(data.index)
    return data


def _run_pair(model_name: str, runner, *, kwargs: dict, out_dir: Path) -> None:
    idata_map = {}
    for orth in (False, True):
        prefix = "orth_" if orth else ""
        run_name = f"{model_name}_{prefix}{X_NAME}"
        print(f"Running Gibbs model: {run_name}", flush=True)
        draws = runner(
            **kwargs,
            n_iter=N_ITER,
            burn=BURN,
            thin=THIN,
            rng=np.random.default_rng(SEED),
            orth=orth,
        )
        idata_map[run_name] = draws_to_idata(draws)
    save_idata_map(idata_map, out_dir)


def main() -> None:
    data = _load_data()

    for infl, spec in INFLATION_SPECS.items():
        ces_sample = prepare_gibbs_sample(
            data,
            pi_col=spec["pi"],
            pi_prev_col=spec["pi_prev"],
            pi_expect_col=spec["pi_expect"],
            x_col=X_NAME,
            x_prev_col=f"{X_NAME}_prev",
        )
        pi_kwargs = {
            "pi": np.asarray(ces_sample["pi"], dtype=float),
            "pi_prev": np.asarray(ces_sample["pi_prev"], dtype=float),
            "pi_expect": np.asarray(ces_sample["pi_expect"], dtype=float),
            "x": np.asarray(ces_sample["x"], dtype=float),
            "x_prev": np.asarray(ces_sample["x_prev"], dtype=float),
        }
        infl_upper = infl.upper()
        print(f"{infl_upper} CES sample: {format_sample_window(ces_sample)}", flush=True)

        _run_pair(
            f"{infl_upper}_CES",
            run_ces,
            kwargs={**pi_kwargs, "prior_specs": CES_PRIORS},
            out_dir=REPO_ROOT / "results" / "idata" / f"gibbs_ces_{infl}",
        )

        for n_slug, n_col in N_SPECS.items():
            hsa_sample = prepare_gibbs_sample(
                data,
                pi_col=spec["pi"],
                pi_prev_col=spec["pi_prev"],
                pi_expect_col=spec["pi_expect"],
                x_col=X_NAME,
                x_prev_col=f"{X_NAME}_prev",
                n_col=n_col,
            )
            hsa_kwargs = {
                "pi": np.asarray(hsa_sample["pi"], dtype=float),
                "pi_prev": np.asarray(hsa_sample["pi_prev"], dtype=float),
                "pi_expect": np.asarray(hsa_sample["pi_expect"], dtype=float),
                "x": np.asarray(hsa_sample["x"], dtype=float),
                "x_prev": np.asarray(hsa_sample["x_prev"], dtype=float),
                "N": np.asarray(hsa_sample["N"], dtype=float),
            }
            print(f"{infl_upper} HSA {n_slug} sample: {format_sample_window(hsa_sample)}", flush=True)

            _run_pair(
                f"{infl_upper}_HSA_{n_slug}",
                run_hsa_dynamic,
                kwargs={**hsa_kwargs, "prior_specs": DYNAMIC_PRIORS},
                out_dir=REPO_ROOT / "results" / "idata" / f"gibbs_hsa_dynamic_{n_slug}_{infl}",
            )
            _run_pair(
                f"{infl_upper}_HSA_{n_slug}",
                run_hsa_steady,
                kwargs={**hsa_kwargs, "prior_specs": STEADY_PRIORS},
                out_dir=REPO_ROOT / "results" / "idata" / f"gibbs_hsa_steady_{n_slug}_{infl}",
            )

    print("=== markup_inv Gibbs models finished ===", flush=True)


if __name__ == "__main__":
    main()
