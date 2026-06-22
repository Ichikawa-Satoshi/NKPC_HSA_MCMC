from __future__ import annotations

import pandas as pd

from nkpc_hsa.inference.period_robustness import apply_period
from nkpc_hsa.report.latex import write_default_report
from nkpc_hsa.report.tables import write_latex_fragment


def test_latex_table_generation(tmp_path) -> None:
    out = tmp_path / "table.tex"
    write_latex_fragment(pd.DataFrame({"parameter": ["alpha"], "mean": [0.5]}), out)
    text = out.read_text(encoding="utf-8")
    assert "\\begin{tabular}" in text
    assert "alpha" in text


def test_report_source_has_expected_paths(tmp_path) -> None:
    tex = write_default_report(tmp_path / "main.tex")
    text = tex.read_text(encoding="utf-8")
    assert "../tables/coefficient_means.tex" in text
    assert "../tables/kappa_model_comparison.tex" in text
    assert "../tables/time_varying_coefficients.tex" in text
    assert "../tables/sddr.tex" in text
    assert "../tables/period_robustness.tex" in text
    assert "../model_comparison/model_comparison.tex" in text


def test_period_filter_excludes_covid() -> None:
    dates = pd.date_range("2019-01-01", periods=12, freq="QE")
    df = pd.DataFrame({"x": range(12)}, index=dates)
    out = apply_period(df, {"exclude": [["2020-01-01", "2020-12-31"]]})
    assert not ((out.index >= "2020-01-01") & (out.index <= "2020-12-31")).any()
