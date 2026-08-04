"""Report which report artifacts changed, against the pre-change snapshot.

Compares the current generated tables/macros with the copies saved in
``results/_review_baseline/`` before any of the five fixes were applied, and
prints a per-file summary plus a line-level diff for the macro files. Used to
document the numerical consequences of the changes rather than asserting them.

    python scripts/report_artifact_diff.py
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

BASE = ROOT / "results" / "_review_baseline"
PAIRS = [
    (BASE / "tables_en", ROOT / "results" / "tables" / "cpi_ppi_report_en", "English report tables"),
    (BASE / "tables_ja", ROOT / "results" / "tables" / "cpi_ppi_report", "Japanese source tables"),
]
NUM = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def _numbers(text: str) -> list[str]:
    return NUM.findall(text)


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"No baseline snapshot at {BASE}")

    for base_dir, new_dir, label in PAIRS:
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        changed, unchanged, added, removed = [], [], [], []
        base_files = {p.relative_to(base_dir) for p in base_dir.rglob("*.tex")}
        new_files = {p.relative_to(new_dir) for p in new_dir.rglob("*.tex")}
        for rel in sorted(base_files | new_files):
            if rel not in new_files:
                removed.append(rel)
                continue
            if rel not in base_files:
                added.append(rel)
                continue
            old_text = (base_dir / rel).read_text(encoding="utf-8")
            new_text = (new_dir / rel).read_text(encoding="utf-8")
            if old_text == new_text:
                unchanged.append(rel)
            else:
                numeric = _numbers(old_text) != _numbers(new_text)
                changed.append((rel, numeric))

        print(f"unchanged: {len(unchanged)}   changed: {len(changed)}   "
              f"new: {len(added)}   removed: {len(removed)}")
        if added:
            print("\n  NEW:")
            for rel in added:
                print(f"    + {rel}")
        if removed:
            print("\n  REMOVED:")
            for rel in removed:
                print(f"    - {rel}")
        if changed:
            print("\n  CHANGED (numeric = at least one number differs):")
            for rel, numeric in changed:
                print(f"    * {rel}   numeric={'YES' if numeric else 'no (labels only)'}")

        # Full diff for the macro files, which carry the headline numbers.
        for rel, _ in changed:
            if "macros" not in rel.name:
                continue
            print(f"\n  --- diff {rel} ---")
            old_lines = (base_dir / rel).read_text(encoding="utf-8").splitlines()
            new_lines = (new_dir / rel).read_text(encoding="utf-8").splitlines()
            for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0):
                if line.startswith(("---", "+++", "@@")):
                    continue
                print(f"      {line}")


if __name__ == "__main__":
    main()
