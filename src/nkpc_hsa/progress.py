"""Optional live progress reporting for long sampler runs.

One format, three consumers:

* a sampler loop calls the callback it finds in ``opts["progress_callback"]``;
* an interactive single-cell run renders a one-line bar on stderr;
* ``production/main_scripts/run_restriction_production.py`` runs its cells as subprocesses,
  puts the children in ``stream`` style and aggregates their machine-readable
  events into one board with an overall bar.

Progress is display only. Nothing here touches the draws, the seeds, the
priors, or the saved run, so a run with the bar on and the same run with it off
are the same run.

Style resolution, in order: the explicit argument, ``NKPC_HSA_PROGRESS``, then
``auto`` — a bar when the output stream is a TTY and ``off`` when it is not, so
piping to a log file does not fill it with redraws.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, TextIO

__all__ = [
    "EVENT_PREFIX",
    "ProgressBoard",
    "ProgressReporter",
    "STYLES",
    "format_duration",
    "parse_event",
    "resolve_style",
]

STYLES = ("auto", "bar", "plain", "stream", "off")
ENV_VAR = "NKPC_HSA_PROGRESS"
EVENT_PREFIX = "##NKPC-PROGRESS "

_BAR_FILL = "#"
_BAR_EMPTY = "."
_WRITE_LOCK = threading.Lock()


def resolve_style(requested: str | None = None, *, stream: TextIO | None = None) -> str:
    """Return one of ``bar``/``plain``/``stream``/``off``."""
    style = (requested or os.environ.get(ENV_VAR) or "auto").strip().lower()
    if style not in STYLES:
        raise ValueError(f"Unknown progress style {style!r}; expected one of {', '.join(STYLES)}.")
    if style != "auto":
        return style
    target = stream if stream is not None else sys.stderr
    try:
        interactive = bool(target.isatty())
    except (AttributeError, ValueError):
        interactive = False
    return "bar" if interactive else "off"


def format_duration(seconds: float) -> str:
    if not (seconds == seconds) or seconds < 0 or seconds == float("inf"):  # NaN/inf guard
        return "--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _bar(fraction: float, width: int) -> str:
    width = max(4, width)
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return _BAR_FILL * filled + _BAR_EMPTY * (width - filled)


def _terminal_width(default: int = 100) -> int:
    return max(40, shutil.get_terminal_size((default, 24)).columns)


def parse_event(line: str) -> dict[str, Any] | None:
    """Decode one ``stream``-style line, or return ``None`` for ordinary output."""
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(EVENT_PREFIX):])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class ProgressReporter:
    """A single unit of work whose completion is reported as ``done``/``total``.

    A sampler run reports over every chain of one cell, so the bar crosses the
    whole cell once rather than restarting per chain.
    """

    def __init__(
        self,
        total: int,
        *,
        label: str = "",
        key: str | None = None,
        style: str | None = None,
        stream: TextIO | None = None,
        min_interval: float = 0.4,
        plain_interval: float = 60.0,
    ) -> None:
        self.total = max(1, int(total))
        self.label = label
        self.key = key or label or "run"
        self.style = resolve_style(style, stream=stream)
        # ``stream`` events go to stdout because the parent pipeline reads the
        # child's stdout; the human-facing bar goes to stderr so it never lands
        # in a redirected result file.
        self._out = stream if stream is not None else (sys.stdout if self.style == "stream" else sys.stderr)
        self._min_interval = float(min_interval)
        self._plain_interval = float(plain_interval)
        # Time is only consulted on a fraction of the iterations: a 400k-draw
        # chain would otherwise call monotonic() a million times for a bar that
        # can redraw a few hundred times at most.
        self._gate = max(1, self.total // 2000)
        self._start = time.monotonic()
        self._last_emit = 0.0
        self._done = 0
        self._closed = False

    # -- reporting ---------------------------------------------------------
    def update(self, done: int, *, force: bool = False) -> None:
        if self._closed:
            return
        done = max(0, min(int(done), self.total))
        self._done = done
        if not force and done % self._gate and done != self.total:
            return
        now = time.monotonic()
        interval = self._min_interval if self.style != "plain" else self._plain_interval
        if not force and done != self.total and (now - self._last_emit) < interval:
            return
        self._last_emit = now
        self._emit(now - self._start, final=False)

    def callback(self, offset: int = 0) -> Callable[[int, int], None]:
        """Return the ``(iteration, total_iter)`` hook the sampler loops call."""
        return lambda iteration, _total_iter, _offset=int(offset): self.update(_offset + iteration)

    def write_line(self, text: str) -> None:
        """Print ordinary output without leaving it on top of the bar."""
        if self.style != "bar":
            print(text, flush=True)
            return
        with _WRITE_LOCK:
            self._out.write("\r\x1b[2K" + text.rstrip("\n") + "\n")
            self._out.flush()
        self.update(self._done, force=True)

    def finish(self, note: str = "") -> None:
        if self._closed:
            return
        elapsed = time.monotonic() - self._start
        self._done = self.total
        self._emit(elapsed, final=True, note=note)
        self._closed = True

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish("failed" if exc_type is not None else "")

    # -- rendering ---------------------------------------------------------
    @property
    def fraction(self) -> float:
        return self._done / self.total

    def _eta(self, elapsed: float) -> float:
        if self._done <= 0:
            return float("inf")
        return elapsed * (self.total - self._done) / self._done

    def _emit(self, elapsed: float, *, final: bool, note: str = "") -> None:
        if self.style == "off":
            return
        if self.style == "stream":
            payload = {
                "key": self.key,
                "label": self.label,
                "done": self._done,
                "total": self.total,
                "elapsed": round(elapsed, 2),
                "final": final,
            }
            if note:
                payload["note"] = note
            text = EVENT_PREFIX + json.dumps(payload)
            with _WRITE_LOCK:
                self._out.write(text + "\n")
                self._out.flush()
            return

        percent = f"{self.fraction * 100:5.1f}%"
        counts = f"{self._done}/{self.total}"
        timing = f"{format_duration(elapsed)}<{format_duration(self._eta(elapsed))}"
        if final:
            timing = f"{format_duration(elapsed)} elapsed"
        tail = f" {note}" if note else ""

        if self.style == "plain":
            line = f"[progress] {self.label} {percent} {counts} {timing}{tail}"
            with _WRITE_LOCK:
                self._out.write(line + "\n")
                self._out.flush()
            return

        # Reserve a fixed amount for the counters so the bar keeps one width for
        # the whole run instead of breathing as the numbers grow.
        reserve = len(self.label) + 2 * len(str(self.total)) + 34
        width = max(8, min(40, _terminal_width() - reserve))
        line = f"{self.label} [{_bar(self.fraction, width)}] {percent} {counts} {timing}{tail}"
        with _WRITE_LOCK:
            self._out.write("\r\x1b[2K" + line[: _terminal_width() - 1])
            self._out.write("\n" if final else "")
            self._out.flush()


class ProgressBoard:
    """Aggregate several cells estimated in parallel into one live display.

    The overall fraction is the mean of the per-cell fractions, so it advances
    smoothly inside a cell instead of only when one finishes — which matters
    when a single cell is 400,000 draws across four chains.
    """

    def __init__(
        self,
        keys: Iterable[str],
        *,
        style: str | None = None,
        stream: TextIO | None = None,
        redraw_interval: float = 0.4,
        plain_interval: float = 60.0,
        title: str = "estimation",
    ) -> None:
        self._out = stream if stream is not None else sys.stderr
        self.style = resolve_style(style, stream=self._out)
        self.title = title
        self._keys = list(keys)
        self._cells: dict[str, dict[str, Any]] = {
            key: {"done": 0, "total": 0, "state": "pending", "note": ""} for key in self._keys
        }
        self._redraw_interval = float(redraw_interval)
        self._plain_interval = float(plain_interval)
        self._start = time.monotonic()
        self._last_draw = 0.0
        self._drawn_lines = 0
        self._lock = threading.Lock()

    # -- state -------------------------------------------------------------
    def start_cell(self, key: str) -> None:
        with self._lock:
            cell = self._cells.setdefault(key, {"done": 0, "total": 0, "state": "pending", "note": ""})
            if cell["state"] == "pending":
                cell["state"] = "running"
        self.render()

    def apply_event(self, event: Mapping[str, Any]) -> None:
        key = str(event.get("key") or "")
        if not key:
            return
        with self._lock:
            cell = self._cells.setdefault(key, {"done": 0, "total": 0, "state": "running", "note": ""})
            cell["done"] = int(event.get("done", cell["done"]))
            cell["total"] = int(event.get("total", cell["total"]))
            if cell["state"] == "pending":
                cell["state"] = "running"
        self.render()

    def finish_cell(self, key: str, *, ok: bool = True, note: str = "") -> None:
        with self._lock:
            cell = self._cells.setdefault(key, {"done": 0, "total": 0, "state": "running", "note": ""})
            cell["state"] = "done" if ok else "failed"
            cell["note"] = note
            if ok and cell["total"]:
                cell["done"] = cell["total"]
        self.render(force=True)

    # -- rendering ---------------------------------------------------------
    def _cell_fraction(self, cell: Mapping[str, Any]) -> float:
        if cell["state"] == "done":
            return 1.0
        if not cell["total"]:
            return 0.0
        return max(0.0, min(1.0, cell["done"] / cell["total"]))

    def _lines(self) -> list[str]:
        # One snapshot for the whole frame: a worker thread finishing mid-render
        # would otherwise put a header and its own cell lines out of agreement.
        with self._lock:
            cells = [(key, dict(cell)) for key, cell in self._cells.items()]
        completed = sum(1 for _, cell in cells if cell["state"] == "done")
        fraction = sum(self._cell_fraction(cell) for _, cell in cells) / len(cells) if cells else 0.0
        elapsed = time.monotonic() - self._start
        eta = float("inf") if fraction <= 0 else elapsed * (1 - fraction) / fraction
        head = (
            f"{self.title} [{_bar(fraction, 32)}] {fraction * 100:5.1f}%  "
            f"{completed}/{len(cells)} cells  {format_duration(elapsed)}<{format_duration(eta)}"
        )
        lines = [head]
        width = max(len(key) for key, _ in cells) if cells else 0
        for key, cell in cells:
            state = cell["state"]
            if state == "pending":
                detail = "pending"
            elif state == "done":
                detail = f"done{(' ' + cell['note']) if cell['note'] else ''}"
            elif state == "failed":
                detail = f"FAILED{(' ' + cell['note']) if cell['note'] else ''}"
            elif cell["total"]:
                detail = f"{self._cell_fraction(cell) * 100:5.1f}%  {cell['done']}/{cell['total']} draws"
            else:
                detail = "starting"
            lines.append(f"  {key:<{width}}  {detail}")
        return lines

    def render(self, *, force: bool = False) -> None:
        if self.style == "off":
            return
        now = time.monotonic()
        interval = self._plain_interval if self.style == "plain" else self._redraw_interval
        if not force and (now - self._last_draw) < interval:
            return
        self._last_draw = now
        lines = self._lines()
        with _WRITE_LOCK:
            if self.style == "plain":
                self._out.write(lines[0] + "\n")
            else:
                if self._drawn_lines:
                    self._out.write(f"\x1b[{self._drawn_lines}A")
                self._out.write("".join(f"\x1b[2K{line}\n" for line in lines))
                self._drawn_lines = len(lines)
            self._out.flush()

    def write_line(self, text: str) -> None:
        """Print ordinary output above the board instead of into it."""
        if self.style != "bar":
            print(text, flush=True)
            return
        with _WRITE_LOCK:
            if self._drawn_lines:
                self._out.write(f"\x1b[{self._drawn_lines}A\x1b[J")
                self._drawn_lines = 0
            self._out.write(text.rstrip("\n") + "\n")
            self._out.flush()
        self.render(force=True)

    def close(self) -> None:
        if self.style == "off":
            return
        self.render(force=True)
        with _WRITE_LOCK:
            self._out.flush()
