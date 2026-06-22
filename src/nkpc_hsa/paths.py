from __future__ import annotations

from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the repository root by walking upward from *start*."""
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "README.md").exists() and (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not locate project root from {current}")


def project_path(*parts: str | Path, root: str | Path | None = None) -> Path:
    base = find_project_root(root)
    return base.joinpath(*map(Path, parts))
