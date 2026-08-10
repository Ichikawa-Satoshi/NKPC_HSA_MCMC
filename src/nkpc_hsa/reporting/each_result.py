"""Group saved runs into one directory per (activity x price) x competition cell.

The production report presents the grid collapsed across data specifications. This
module supports the complementary view: every data combination on its own, so a
cell can be read without the surrounding comparison.

Layout::

    results/each_result/<slack>_<price>/<competition>/
        <cell>.tex          one self-contained document per cell
        tables/*.tex        \\input by the document
        figures/*.pdf       \\includegraphics by the document

The activity and price index are in the directory name; the three priors are
compared *inside* the document, because a prior is a robustness dimension of one
cell rather than a different cell.

A cell is identified from each run's saved ``data_spec.json`` and
``metadata.json`` rather than from its directory name: the name encodes the same
information but has to be parsed against model names that themselves contain
underscores, and the spec is what the sampler actually used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Activity series -> (directory fragment, human label).
SLACK_LABELS: dict[str, tuple[str, str]] = {
    "unemp_gap": ("unempgap", "Unemployment gap ($u^* - u$)"),
    "output_gap_BN": ("output_BN", "Output gap, Beveridge--Nelson filter"),
    "output_gap_HP": ("output_HP", "Output gap, HP filter ($\\lambda = 1600$)"),
    "labor_share_gap_HP": ("laborshare_HP", "Labour share gap, HP filter"),
    "markup_BN_inv": ("inv_markup", "Inverse markup (real marginal cost proxy)"),
}

# Inflation series -> (directory fragment, human label).
PRICE_LABELS: dict[str, tuple[str, str]] = {
    "pi_cpi": ("cpi", "Headline CPI"),
    "pi_cpi_core": ("core", "Core CPI"),
    "pi_ppi": ("ppi", "PPI"),
}

# Competition series -> (directory fragment, human label).
COMPETITION_LABELS: dict[str, tuple[str, str]] = {
    "N_Gustavo": ("N_gustavo", "Inverse HHI of listed firms, annual"),
    "N_TNIC": ("N_tnic", "TNIC-3 inverse HHI"),
    "N_SEC_inverse_HHI": ("N_sec", "SEC inverse HHI, firm-count-weighted market mean"),
    "N_SEC_inverse_HHI_revw": ("N_sec_revw", "SEC inverse HHI, revenue-weighted market mean"),
    "N_SEC_inverse_HHI_logrevw": ("N_sec_logrevw", "SEC inverse HHI, revenue-weighted geometric mean"),
    "N_SEC_inverse_HHI_logrevw_exfin": (
        "N_sec_logrevw_exfin",
        "SEC inverse HHI, revenue-weighted geometric mean, excluding SIC 6000--6999",
    ),
}

# Establishment series -> label. Its presence appends "_E" to the competition
# directory, because a joint N/E run is a different estimation, not a different
# competition measure.
ESTABLISHMENT_LABELS: dict[str, str] = {
    "qcew_establishments": "QCEW private establishments, quarterly",
    "establishment_stock": "BED establishment stock (deprecated direct loading)",
}

MODEL_ORDER = ["ces", "hsa_steady", "hsa_dynamic", "hsa_const_theta", "hsa_full"]
PRIOR_ORDER = ["baseline", "weak", "tight"]


@dataclass(frozen=True)
class RunRef:
    """One saved run, resolved to the cell it belongs in."""

    path: Path
    model: str
    prior: str
    frequency: str
    spec_name: str
    data_spec: dict
    metadata: dict

    @property
    def sample(self) -> tuple[str, str]:
        return str(self.metadata.get("sample_start", "")), str(self.metadata.get("sample_end", ""))

    @property
    def n_obs(self) -> int | None:
        value = self.metadata.get("n_obs")
        return None if value is None else int(value)


@dataclass(frozen=True)
class Cell:
    """Every run sharing one (activity, price index, competition series)."""

    slack_dir: str
    competition_dir: str
    slack_label: str
    price_label: str
    competition_label: str
    establishment_label: str | None
    runs: tuple[RunRef, ...]

    @property
    def relative_dir(self) -> Path:
        return Path(self.slack_dir) / self.competition_dir

    @property
    def name(self) -> str:
        return f"{self.slack_dir}__{self.competition_dir}"

    @property
    def title(self) -> str:
        base = f"{self.slack_label} $\\times$ {self.price_label} $\\times$ {self.competition_label}"
        return base if self.establishment_label is None else f"{base}, joint with {self.establishment_label}"

    def sorted_runs(self) -> list[RunRef]:
        def key(run: RunRef) -> tuple[int, int, str]:
            model = MODEL_ORDER.index(run.model) if run.model in MODEL_ORDER else len(MODEL_ORDER)
            prior = PRIOR_ORDER.index(run.prior) if run.prior in PRIOR_ORDER else len(PRIOR_ORDER)
            return model, prior, run.path.name

        return sorted(self.runs, key=key)

    def models(self) -> list[str]:
        seen = {run.model for run in self.runs}
        ordered = [model for model in MODEL_ORDER if model in seen]
        return ordered + sorted(seen.difference(ordered))

    def priors(self) -> list[str]:
        seen = {run.prior for run in self.runs}
        ordered = [prior for prior in PRIOR_ORDER if prior in seen]
        return ordered + sorted(seen.difference(ordered))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _prior_from(metadata: dict, directory: Path) -> str:
    for key in ("prior_name", "prior_spec"):
        value = metadata.get(key)
        if isinstance(value, str) and value in PRIOR_ORDER:
            return value
    # Fall back to the directory name, which always carries the prior.
    for prior in PRIOR_ORDER:
        if f"_{prior}_" in directory.name:
            return prior
    return "baseline"


def load_run(directory: Path) -> RunRef | None:
    """Read one run directory, or return None if it is not a complete run."""
    if not (directory / "posterior.nc").exists():
        return None
    metadata = _read_json(directory / "metadata.json")
    if metadata is None:
        return None
    data_spec = _read_json(directory / "data_spec.json") or metadata.get("data_spec") or {}
    model = str(metadata.get("model", "")).strip()
    if not model:
        return None
    return RunRef(
        path=directory,
        model=model,
        prior=_prior_from(metadata, directory),
        frequency=str(metadata.get("competition_measurement_frequency", "unknown")),
        spec_name=str(data_spec.get("name", directory.name)),
        data_spec=dict(data_spec),
        metadata=dict(metadata),
    )


def discover_runs(roots: list[Path]) -> list[RunRef]:
    """Collect every run directory under the given roots."""
    runs: list[RunRef] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            resolved = directory.resolve()
            if resolved in seen:
                continue
            run = load_run(directory)
            if run is not None:
                seen.add(resolved)
                runs.append(run)
    return runs


def competition_key(data_spec: dict) -> tuple[str, str, str | None]:
    """Return (directory fragment, competition label, establishment label)."""
    n_col = str(data_spec.get("n_col", "")) or "unknown"
    fragment, label = COMPETITION_LABELS.get(n_col, (n_col, n_col))
    e_col = data_spec.get("e_col")
    if not e_col:
        return fragment, label, None
    establishment = ESTABLISHMENT_LABELS.get(str(e_col), str(e_col))
    return f"{fragment}_E", label, establishment


def slack_key(data_spec: dict) -> tuple[str, str, str, str]:
    """Return (directory fragment, activity label, price label, price fragment)."""
    x_col = str(data_spec.get("x_col", "")) or "unknown"
    pi_col = str(data_spec.get("pi_col", "")) or "unknown"
    slack_fragment, slack_label = SLACK_LABELS.get(x_col, (x_col, x_col))
    price_fragment, price_label = PRICE_LABELS.get(pi_col, (pi_col, pi_col))
    return f"{slack_fragment}_{price_fragment}", slack_label, price_label, price_fragment


def build_cells(runs: list[RunRef]) -> list[Cell]:
    """Group runs into cells, dropping nothing and inventing nothing."""
    grouped: dict[tuple[str, str], list[RunRef]] = {}
    labels: dict[tuple[str, str], tuple[str, str, str, str | None]] = {}
    for run in runs:
        slack_dir, slack_label, price_label, _ = slack_key(run.data_spec)
        competition_dir, competition_label, establishment_label = competition_key(run.data_spec)
        key = (slack_dir, competition_dir)
        grouped.setdefault(key, []).append(run)
        labels.setdefault(key, (slack_label, price_label, competition_label, establishment_label))

    cells: list[Cell] = []
    for key in sorted(grouped):
        slack_dir, competition_dir = key
        slack_label, price_label, competition_label, establishment_label = labels[key]
        cells.append(
            Cell(
                slack_dir=slack_dir,
                competition_dir=competition_dir,
                slack_label=slack_label,
                price_label=price_label,
                competition_label=competition_label,
                establishment_label=establishment_label,
                runs=tuple(grouped[key]),
            )
        )
    return cells
