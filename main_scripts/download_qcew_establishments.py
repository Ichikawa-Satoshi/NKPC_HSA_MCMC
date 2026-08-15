"""Download the official BLS quarterly QCEW archives used by the extension."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import fsspec

from _bootstrap import DATA_DIR


def archive_url(year: int) -> str:
    # BLS supplies ownership totals only for the pre-1990 NAICS reconstruction.
    name = f"{year}_qtrly_naics10_totals.zip" if year <= 1989 else f"{year}_qtrly_by_industry.zip"
    return f"https://data.bls.gov/cew/data/files/{year}/csv/{name}"


def _wayback(url: str) -> str:
    # Fixed archival vintage, used only when the live BLS host is unavailable.
    return "https://web.archive.org/web/20250701id_/" + url


def _download_small_archive(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "NKPC-HSA-QCEW/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
    except Exception:
        request = urllib.request.Request(_wayback(url), headers={"User-Agent": "NKPC-HSA-QCEW/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = response.read()
    if not payload.startswith(b"PK"):
        raise RuntimeError(f"Source did not return a ZIP; first bytes={payload[:80]!r}")
    target.write_bytes(payload)


def _extract_total_industry_remotely(url: str, target: Path) -> None:
    # The full yearly by-industry ZIP can exceed 400 MB, but it stores one CSV
    # per industry and supports HTTP range requests. Read the central directory
    # and the Total-all-industries member only.
    last_error: Exception | None = None
    for remote in (url, _wayback(url)):
        try:
            fs = fsspec.filesystem("zip", fo=remote)
            members = fs.find("")
            candidates = [name for name in members if " 10 Total, all industries.csv" in name]
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one total-industry member, found {len(candidates)}")
            with fs.open(candidates[0], "rb") as source, target.open("wb") as output:
                output.write(source.read())
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not selectively retrieve {url}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "raw" / "competition" / "qcew")
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=2012)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for year in range(args.start_year, args.end_year + 1):
        url = archive_url(year)
        target = args.out_dir / (
            url.rsplit("/", 1)[-1] if year <= 1989 else f"{year}_qcew_total_industry.csv"
        )
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"Downloading {url}", flush=True)
        if year <= 1989:
            _download_small_archive(url, target)
        else:
            _extract_total_industry_remotely(url, target)


if __name__ == "__main__":
    main()
