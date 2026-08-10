from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR_ENV = "NKPC_HSA_PROJECT_DIR"
DROPBOX_DIR_ENV = "NKPC_HSA_DROPBOX_DIR"


def _configured_root(env_name: str) -> Path | None:
    configured = os.environ.get(env_name, "").strip()
    return Path(configured).expanduser().resolve() if configured else None


def _is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "nkpc_hsa").is_dir()


def _discover_project_root(start: str | Path | None = None) -> Path:
    starts = [Path(start).expanduser().resolve()] if start is not None else []
    starts.extend((Path.cwd().resolve(), Path(__file__).resolve()))
    for candidate in starts:
        base = candidate if candidate.is_dir() else candidate.parent
        for path in (base, *base.parents):
            if _is_project_root(path):
                return path
    raise RuntimeError(
        f"Could not discover the repository root; set {PROJECT_DIR_ENV} explicitly."
    )


def project_root() -> Path:
    """Return the configured repository root, or discover it from this checkout."""
    return _configured_root(PROJECT_DIR_ENV) or _discover_project_root()


def dropbox_root() -> Path:
    """Return the configured or locally discovered Dropbox project folder."""
    configured = _configured_root(DROPBOX_DIR_ENV)
    if configured is not None:
        return configured

    project_name = project_root().name
    home = Path.home()
    candidates = (
        home / "Library" / "CloudStorage" / "Dropbox" / project_name,
        home / "Dropbox" / project_name,
    )
    matches = list(dict.fromkeys(path.resolve() for path in candidates if path.is_dir()))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        rendered = ", ".join(map(str, matches))
        raise RuntimeError(
            f"Multiple Dropbox project folders found ({rendered}); set {DROPBOX_DIR_ENV}."
        )
    raise RuntimeError(
        f"Could not find Dropbox/{project_name}; set {DROPBOX_DIR_ENV} explicitly."
    )


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the configured repository root or discover it above *start*."""
    return _configured_root(PROJECT_DIR_ENV) or _discover_project_root(start)


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
