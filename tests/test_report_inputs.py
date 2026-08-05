"""Every file the report pulls in must be produced by a script in the build.

Two failures motivated this file, both silent:

* ``pp_kappa_ces_vs_steady.png`` was ``\\includegraphics``-ed by the report but had
  no producer anywhere in the code or in git history. It survived only because the
  build directory was never deleted; a clean rebuild could not reproduce the paper.
* Macro files were written to a path the ``.tex`` no longer read, so the
  ``\\providecommand`` fallbacks won and the PDF shipped ``??`` where numbers
  belonged -- LaTeX reports nothing, because a fallback is a legitimate definition.

These tests read the ``.tex`` and check it against the build, so neither can
recur without a test failure.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "nkpc_hsa_report.tex"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

pytestmark = pytest.mark.skipif(
    not REPORT.exists() or not TABLES.exists(),
    reason="report inputs not built; run scripts/build_report.py",
)


def _source() -> str:
    return REPORT.read_text(encoding="utf-8")


def _body() -> str:
    """The .tex after \\begin{document}, with comment lines dropped."""
    text = _source().split(r"\begin{document}", 1)[1]
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))


def test_every_input_exists() -> None:
    missing = [
        path for path in re.findall(r"\\input\{([^}]*)\}", _source())
        if not (REPORT.parent / path).exists()
    ]
    assert not missing, f"\\input targets that no script produced: {missing}"


def test_every_iffileexists_guard_points_at_a_real_path() -> None:
    """A guard that names the wrong path fails open, not loud.

    ``\\IfFileExists`` silently takes the else-branch, so a stale path here means
    the macros never load and the fallbacks are what gets typeset.
    """
    stale = []
    for guarded, inputted in re.findall(
        r"\\IfFileExists\{([^}]*)\}\{%?\s*\\input\{([^}]*)\}", _source()
    ):
        if guarded != inputted:
            stale.append((guarded, inputted))
        elif not (REPORT.parent / guarded).exists():
            stale.append((guarded, "does not exist"))
    assert not stale, f"IfFileExists guard disagrees with its \\input: {stale}"


def test_every_includegraphics_exists() -> None:
    graphics_root = REPORT.parent / re.search(
        r"\\graphicspath\{\{([^}]*)\}\}", _source()
    ).group(1)
    missing = [
        path for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", _source())
        if not any((graphics_root / (path + suffix)).exists() for suffix in ("", ".png", ".pdf"))
    ]
    assert not missing, f"figures the report needs but the build does not produce: {missing}"


def test_no_macro_name_contains_a_digit() -> None:
    """TeX control sequences are letters only.

    ``\\providecommand{\\AnnualAr2WorstCount}{56}`` defines ``\\AnnualAr`` and leaves
    ``2WorstCount`` as literal text, in both the definition and the use. Nothing
    errors; the PDF just reads ``22WorstCount``.
    """
    offenders = set()
    for path in list(TABLES.rglob("*.tex")) + [REPORT]:
        offenders |= {
            name for name in re.findall(r"\\providecommand\{\\(\w+)\}", path.read_text(encoding="utf-8"))
            if any(character.isdigit() for character in name)
        }
    assert not offenders, f"macro names must be letters only: {sorted(offenders)}"


def test_every_generated_macro_is_defined_where_the_report_uses_it() -> None:
    source = _source()
    defined = set(re.findall(r"\\providecommand\{\\(\w+)\}", source))
    for path in TABLES.rglob("*.tex"):
        defined |= set(re.findall(r"\\providecommand\{\\(\w+)\}", path.read_text(encoding="utf-8")))
    defined |= set(re.findall(r"\\newcommand\{\\(\w+)\}", source))
    # Control sequences the LaTeX kernel and the loaded packages provide.
    builtin = {
        "Delta", "Sigma", "Omega", "Gamma", "Lambda", "Pr", "IfFileExists",
        "E", "P", "N", "R", "T", "Big", "Bigg", "Large", "Leftrightarrow", "Rightarrow",
    }
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)\b", _body()))
    assert not used - defined - builtin, (
        f"macros used but never generated: {sorted(used - defined - builtin)}"
    )


def test_both_observation_designs_are_separated() -> None:
    """Neither design may sit at the top level of the build directory.

    "the files without a subdirectory" is not a readable way to name a design,
    and it is how the interpolated tables previously got mixed in with the
    design-independent ones.
    """
    for root in (TABLES, FIGURES):
        # Dot-files are the operating system's, not the build's: Finder drops a
        # .DS_Store into any directory it displays, and failing on that would make
        # the test depend on whether somebody opened the folder.
        loose = [p.name for p in root.iterdir() if p.is_file() and not p.name.startswith(".")]
        assert not loose, f"{root.name}/ should hold only subdirectories, found {loose}"
        assert {"annual_q4", "quarterly_interpolated", "shared"} <= {
            p.name for p in root.iterdir() if p.is_dir()
        }, f"{root.name}/ is missing one of annual_q4 / quarterly_interpolated / shared"
