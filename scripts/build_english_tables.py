"""Generate English copies of the auto-generated report tables.

The CPI/PPI report pipeline (``scripts/12_build_cpi_ppi_report.py`` and
``scripts/11_additional_report_evidence.py``) writes the ``\\input`` table
fragments with a handful of Japanese label literals, because those tables are
shared with the Japanese report (``paper/nkpc_hsa_report_ja.tex``). The English
report (``paper/nkpc_hsa_report.tex``) needs the same tables with English
labels.

This script does *not* re-estimate or recompute anything. It reads the existing
generated ``.tex`` fragments and writes English copies into parallel ``*_en``
directories, replacing only the bounded set of Japanese label strings. Numeric
content is untouched, so the two language versions stay perfectly consistent and
the Japanese tables are left byte-for-byte unchanged.

Re-run this whenever the report tables are regenerated:

    python scripts/build_english_tables.py
"""

from __future__ import annotations

from pathlib import Path

from _bootstrap import ROOT

# Source -> destination directory pairs (recursive; mirrors sub-directories).
DIR_PAIRS: list[tuple[Path, Path]] = [
    (
        ROOT / "results" / "tables" / "cpi_ppi_report",
        ROOT / "results" / "tables" / "cpi_ppi_report_en",
    ),
    (
        ROOT / "results" / "tables" / "report_additions",
        ROOT / "results" / "tables" / "report_additions_en",
    ),
]

# Ordered label replacements. Longer / compound strings must precede any of
# their substrings (e.g. the theory-correspondence label before "inverse
# markup", the "current + 4 lags" cell before "current").
REPLACEMENTS: list[tuple[str, str]] = [
    ("逆マークアップ（理論対応）", "Inverse markup (theory)"),
    ("当期+4ラグの和", "current + 4 lags"),
    ("標本内1標準偏差", "one sample s.d."),
    ("事後潜在トレンド", "posterior latent trend"),
    ("元データのBNトレンド", "source BN trend"),
    ("集計マークアップ", "aggregate markup"),
    ("逆マークアップ", "inverse markup"),
    ("上場企業数", "listed firm count"),
    ("観測企業数", "observed N"),
    ("失業ギャップ", "unemployment gap"),
    ("要注意", "watch"),
    ("当期", "current"),
    ("ポイント", " pp"),
    ("年", " yr"),
]


def translate(text: str) -> str:
    for src, dst in REPLACEMENTS:
        text = text.replace(src, dst)
    return text


def _has_cjk(text: str) -> bool:
    return any(
        "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" for ch in text
    )


def main() -> None:
    written = 0
    leftover: list[str] = []
    for src_dir, dst_dir in DIR_PAIRS:
        if not src_dir.exists():
            continue
        for src_path in sorted(src_dir.rglob("*.tex")):
            rel = src_path.relative_to(src_dir)
            dst_path = dst_dir / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            translated = translate(src_path.read_text(encoding="utf-8"))
            dst_path.write_text(translated, encoding="utf-8")
            written += 1
            if _has_cjk(translated):
                leftover.append(str(dst_path.relative_to(ROOT)))

    print(f"Wrote {written} English table fragment(s).")
    if leftover:
        print("WARNING: residual CJK characters remain in:")
        for name in leftover:
            print(f"  - {name}")
    else:
        print("No residual Japanese labels detected.")


if __name__ == "__main__":
    main()
