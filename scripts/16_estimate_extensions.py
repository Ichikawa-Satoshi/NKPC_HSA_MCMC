from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from _bootstrap import DATA_DIR, RESULTS_DIR, ROOT
from nkpc_hsa.config import configured_data_specs, load_model_config
from nkpc_hsa.inference.wrappers import ESTIMATION_REVISION, run_model
from nkpc_hsa.reporting.cpi_ppi_spec import report_run_keys


PRIOR_FILES = {name: ROOT / "configs" / f"priors_{name}.yaml" for name in ("baseline", "weak", "tight")}


def sec_variant(config: dict, name: str | None) -> tuple[str, dict]:
    """Resolve a SEC competition aggregate from configs/models.yaml.

    Returns the variant name and its block.  The default is the firm-weighted
    aggregate, so an existing run grid keeps its spec names and run directories.
    """
    block = (config.get("sec_competition_variants", {}) or {})
    variants = block.get("variants", {}) or {}
    if not variants:
        raise ValueError("configs/models.yaml has no sec_competition_variants.variants block.")
    resolved = name or str(block.get("default", "firm_weighted"))
    if resolved not in variants:
        raise ValueError(f"Unknown SEC variant {resolved!r}; available: {sorted(variants)}")
    return resolved, variants[resolved]


def _extension_spec(base: dict, base_name: str, extension: str, sec: dict | None = None) -> dict:
    spec = dict(base)
    if extension == "sec_inverse_hhi":
        if sec is None:
            raise ValueError("sec_inverse_hhi requires a resolved variant block.")
        spec.update(
            name=f"{base_name}__sec_inverse_hhi{sec.get('spec_suffix', '')}",
            n_col=str(sec["column"]),
            sample_start="2012Q1",
        )
        # SEC XBRL begins in 2012. Keeping the baseline 2012Q4 endpoint would
        # leave only four observations, so the extension uses each production
        # activity series through its available endpoint and records it per run.
        spec.pop("sample_end", None)
    elif extension == "qcew_joint":
        spec.update(
            name=f"{base_name}__qcew_joint",
            e_col="qcew_establishments",
            e_transform="log100_centered10",
            establishment_model="joint_state",
            sample_start="1982Q1",
            sample_end="2012Q4",
        )
    else:
        raise ValueError(extension)
    return spec


def _run_one(job: dict) -> tuple[tuple[str, str, str], float]:
    started = time.perf_counter()
    config = load_model_config(job["config"])
    model, base_name, prior = job["key"]
    base = configured_data_specs(config, [base_name])[base_name]
    sec = sec_variant(config, job.get("sec_variant"))[1] if job["extension"] == "sec_inverse_hhi" else None
    spec = _extension_spec(base, base_name, job["extension"], sec)
    data = pd.read_csv(job["data"], parse_dates=["DATE"]).set_index("DATE")
    if spec["n_col"] not in data.columns:
        raise ValueError(
            f"{job['data']} has no column {spec['n_col']}. "
            "Rebuild it with scripts/15_build_extension_data.py."
        )
    frequency = "quarterly_observed" if job["extension"] == "sec_inverse_hhi" else "annual_q4"
    run_dir = Path(job["runs_dir"]) / f"{model}_{spec['name']}_{prior}_{frequency}"
    run_model(
        model,
        data=data,
        data_spec=spec,
        prior_specs=PRIOR_FILES[prior],
        prior_name=prior,
        n_iter=job["n_iter"], burn=job["burn"], thin=job["thin"], chains=job["chains"],
        seed=job["seed"], n_transform="log100_centered10",
        covariance_structure=str((config.get("defaults", {}) or {}).get("covariance_structure", "e_zeta_only")),
        n_particles=int((config.get("defaults", {}) or {}).get("n_particles", 512)),
        competition_measurement={"frequency": frequency, "annual_timing": "q4"},
        enforce_stationary=True,
        ar2_max_tries=int((config.get("defaults", {}) or {}).get("ar2_max_tries", 2000)),
        run_id=job["run_id"], run_dir=run_dir,
    )
    return job["key"], time.perf_counter() - started


def _complete(run_dir: Path, min_iter: int) -> bool:
    metadata_path = run_dir / "metadata.json"
    posterior = run_dir / "posterior.nc"
    if not metadata_path.exists() or not posterior.exists():
        return False
    try:
        meta = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return False
    return meta.get("estimation_revision") == ESTIMATION_REVISION and int(meta.get("n_iter", 0)) >= min_iter


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate the distinct QCEW joint-state or SEC inverse-HHI run grid.")
    parser.add_argument("extension", choices=["qcew_joint", "sec_inverse_hhi"])
    parser.add_argument(
        "--sec-variant",
        help="SEC competition aggregate from configs/models.yaml (default: its sec_competition_variants.default).",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    # Cell filters, so a pipeline smoke test does not have to run the whole grid.
    parser.add_argument("--only-models", nargs="+", help="Restrict to these models.")
    parser.add_argument("--only-specs", nargs="+", help="Restrict to these data specs.")
    parser.add_argument("--only-priors", nargs="+", help="Restrict to these priors.")
    args = parser.parse_args()
    if args.data is None:
        name = "model_ready_sec_inverse_hhi.csv" if args.extension == "sec_inverse_hhi" else "model_ready_qcew_joint.csv"
        args.data = DATA_DIR / "processed" / name
    if args.runs_dir is None:
        args.runs_dir = RESULTS_DIR / "extensions" / args.extension / "runs"

    config = load_model_config(args.config)
    defaults = config.get("defaults", {}) or {}
    n_iter = 80 if args.quick else int(defaults.get("n_iter", 12000))
    burn = 40 if args.quick else int(defaults.get("burn", 4000))
    thin = 2 if args.quick else int(defaults.get("thin", 5))
    chains = 2 if args.quick else int(defaults.get("chains", 2))
    keys = report_run_keys()
    for index, wanted in enumerate((args.only_models, args.only_specs, args.only_priors)):
        if wanted:
            allowed = set(wanted)
            keys = [key for key in keys if key[index] in allowed]
    suffix = ""
    variant_name = None
    if args.extension == "qcew_joint":
        keys = [key for key in keys if key[0] in {"hsa_steady", "hsa_const_theta"}]
    else:
        variant_name, variant = sec_variant(config, args.sec_variant)
        suffix = str(variant.get("spec_suffix", ""))
    frequency = "quarterly_observed" if args.extension == "sec_inverse_hhi" else "annual_q4"
    pending = []
    for key in keys:
        model, base, prior = key
        spec_name = f"{base}__{args.extension}{suffix}"
        run_dir = args.runs_dir / f"{model}_{spec_name}_{prior}_{frequency}"
        if args.force or not _complete(run_dir, n_iter):
            pending.append(key)
    label = args.extension if variant_name is None else f"{args.extension}[{variant_name}]"
    print(f"extension={label} revision={ESTIMATION_REVISION} required={len(keys)} pending={len(pending)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = [
        {
            "key": key, "extension": args.extension, "config": args.config, "data": args.data,
            "runs_dir": args.runs_dir, "n_iter": n_iter, "burn": burn, "thin": thin, "chains": chains,
            "sec_variant": variant_name,
            "seed": int(defaults.get("seed", 12345)) + i * 1009, "run_id": f"{stamp}_{i:03d}",
        }
        for i, key in enumerate(pending)
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(_run_one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            key, elapsed = future.result()
            print(f"[{i}/{len(futures)}] {'/'.join(key)} {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
