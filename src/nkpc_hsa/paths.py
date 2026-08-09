from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR_ENV = "NKPC_HSA_PROJECT_DIR"
DROPBOX_DIR_ENV = "NKPC_HSA_DROPBOX_DIR"


def _configured_root(env_name: str) -> Path:
    configured = os.environ.get(env_name, "").strip()
    if not configured:
        raise RuntimeError(f"Required environment variable {env_name} is not set.")
    return Path(configured).expanduser().resolve()


def project_root() -> Path:
    """Return the explicitly configured GitHub repository root."""
    return _configured_root(PROJECT_DIR_ENV)


def dropbox_root() -> Path:
    """Return the explicitly configured Dropbox project-storage root."""
    return _configured_root(DROPBOX_DIR_ENV)


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the configured repository root; *start* is retained for API compatibility."""
    _ = start
    return project_root()


def data_root(root: str | Path | None = None) -> Path:
    """Return ``<NKPC_HSA_DROPBOX_DIR>/data``."""
    _ = root
    return dropbox_root() / "data"


def results_root(root: str | Path | None = None) -> Path:
    """Return ``<NKPC_HSA_DROPBOX_DIR>/results``."""
    _ = root
    return dropbox_root() / "results"


def project_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    mapped = tuple(map(Path, parts))
    if mapped and mapped[0] == Path("data"):
        return data_root(root).joinpath(*mapped[1:])
    if mapped and mapped[0] == Path("results"):
        return results_root(root).joinpath(*mapped[1:])
    base = project_root()
    return base.joinpath(*mapped)
