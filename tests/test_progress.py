"""The progress display must stay display-only and must not garble the output."""

from __future__ import annotations

import io
import json

import pytest

from nkpc_hsa.progress import (
    EVENT_PREFIX,
    ProgressBoard,
    ProgressReporter,
    parse_event,
    resolve_style,
)


class _Tty(io.StringIO):
    def isatty(self) -> bool:  # noqa: D102
        return True


def test_auto_style_follows_the_stream(monkeypatch):
    monkeypatch.delenv("NKPC_HSA_PROGRESS", raising=False)
    assert resolve_style(None, stream=_Tty()) == "bar"
    assert resolve_style(None, stream=io.StringIO()) == "off"


def test_env_var_overrides_auto(monkeypatch):
    monkeypatch.setenv("NKPC_HSA_PROGRESS", "plain")
    assert resolve_style(None, stream=_Tty()) == "plain"
    # An explicit argument still wins over the environment.
    assert resolve_style("off", stream=_Tty()) == "off"


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError):
        resolve_style("loud")


def test_stream_events_round_trip():
    out = io.StringIO()
    reporter = ProgressReporter(100, label="hsa_r1 [spec]", key="hsa_r1:spec", style="stream", stream=out)
    reporter.update(50, force=True)
    reporter.finish()
    events = [parse_event(line) for line in out.getvalue().splitlines()]
    assert all(event is not None for event in events)
    assert events[0]["key"] == "hsa_r1:spec"
    assert events[0]["done"] == 50
    assert events[-1] == {**events[-1], "done": 100, "final": True}
    assert parse_event("Estimating hsa_r1 [spec]...") is None
    assert parse_event(EVENT_PREFIX + "not json") is None


def test_off_style_writes_nothing():
    out = io.StringIO()
    reporter = ProgressReporter(10, label="x", style="off", stream=out)
    reporter.update(5, force=True)
    reporter.finish()
    assert out.getvalue() == ""


def test_callback_offsets_each_chain():
    """One reporter spans every chain, so chain 2 must continue where chain 1 ended."""
    out = io.StringIO()
    reporter = ProgressReporter(200, label="x", style="stream", stream=out, min_interval=0.0)
    reporter.callback(offset=0)(100, 100)
    reporter.callback(offset=100)(100, 100)
    dones = [json.loads(line[len(EVENT_PREFIX):])["done"] for line in out.getvalue().splitlines()]
    assert dones == [100, 200]


def test_board_overall_is_the_mean_of_the_cells():
    out = io.StringIO()
    board = ProgressBoard(["a", "b"], style="plain", stream=out, redraw_interval=0.0, plain_interval=0.0)
    board.apply_event({"key": "a", "done": 500, "total": 1000})
    board.finish_cell("b")
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    # a is half done, b is finished -> 75%.
    assert " 75.0%" in lines[-1]
    assert "1/2 cells" in lines[-1]


def test_board_reports_a_failed_cell():
    out = io.StringIO()
    board = ProgressBoard(["a"], style="plain", stream=out, redraw_interval=0.0, plain_interval=0.0)
    board.finish_cell("a", ok=False, note="exit 1")
    assert any("FAILED" in line for line in board._lines())


def test_sampler_honours_the_progress_callback():
    """The hook the samplers call must fire on every draw, burn-in included."""
    import numpy as np

    from nkpc_hsa.gibbs.hsa_theory.model import func_hsa_f0

    rng = np.random.default_rng(0)
    T = 40
    seen: list[tuple[int, int]] = []
    func_hsa_f0(
        pi_data=rng.normal(size=T),
        pi_prev_data=rng.normal(size=T),
        Epi_data=rng.normal(size=T),
        x_data=rng.normal(size=T),
        x_prev_data=rng.normal(size=T),
        N_data=rng.normal(size=T),
        n_burn=5,
        n_keep=5,
        priors={},
        opts={"seed": 1, "store_every": 1, "progress_callback": lambda it, total: seen.append((it, total))},
    )
    assert seen == [(it, 10) for it in range(1, 11)]
