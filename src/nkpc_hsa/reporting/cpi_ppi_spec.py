from __future__ import annotations


MODEL_ORDER = ["ces", "hsa_steady", "hsa_dynamic", "hsa_const_theta", "hsa_full"]
MODEL_LABELS = {
    "ces": "CES",
    "hsa_steady": "HSA steady",
    "hsa_dynamic": "HSA dynamic",
    "hsa_const_theta": "HSA const-theta",
    "hsa_full": "HSA full",
}
INFLATION_SPECS = {
    "Headline CPI": {
        "Unemployment gap": "unemployment_gap",
        "HP output gap": "output_gap_hp",
        "BN output gap": "output_gap_bn",
    },
    "Core CPI": {
        "Unemployment gap": "unemployment_gap_core",
        "HP output gap": "output_gap_hp_core",
        "BN output gap": "output_gap_bn_core",
    },
    "PPI": {
        "Unemployment gap": "unemployment_gap_ppi",
        "HP output gap": "output_gap_hp_ppi",
        "BN output gap": "output_gap_bn_ppi",
    },
}
PRIMARY_SPECS = {
    inflation: activity_specs["Unemployment gap"]
    for inflation, activity_specs in INFLATION_SPECS.items()
}
PRIOR_ORDER = ["baseline", "weak", "tight"]
BIAS_RUN_KEYS = [
    ("ces", "inv_markup", "baseline"),
    ("hsa_dynamic", "inv_markup", "baseline"),
]


def report_run_keys() -> list[tuple[str, str, str]]:
    """The 75 main cells plus 2 direct CES/HSA slope-bias cells."""
    baseline_specs = {
        data_spec
        for activity_specs in INFLATION_SPECS.values()
        for data_spec in activity_specs.values()
    }
    keys = {
        (model, data_spec, "baseline")
        for model in MODEL_ORDER
        for data_spec in baseline_specs
    }
    keys.update(
        (model, data_spec, prior)
        for model in MODEL_ORDER
        for data_spec in PRIMARY_SPECS.values()
        for prior in ("weak", "tight")
    )
    keys.update(BIAS_RUN_KEYS)
    return sorted(keys)


def annual_q4_run_keys() -> list[tuple[str, str, str]]:
    """HSA cells that must be re-estimated when annual N is observed only in Q4.

    CES contains no competition state, so its 16 report cells are shared with
    the quarterly-interpolated comparison instead of being duplicated.
    """
    return [key for key in report_run_keys() if key[0] != "ces"]
