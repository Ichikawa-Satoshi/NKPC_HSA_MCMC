from __future__ import annotations

import pytest

from nkpc_hsa.paths import (
    DROPBOX_DIR_ENV,
    PROJECT_DIR_ENV,
    data_root,
    dropbox_root,
    project_path,
    project_root,
    results_root,
)


def test_explicit_project_and_dropbox_roots(monkeypatch, tmp_path) -> None:
    project = tmp_path / "github" / "NKPC_HSA_MCMC"
    dropbox = tmp_path / "dropbox" / "NKPC_HSA_MCMC"
    monkeypatch.setenv(PROJECT_DIR_ENV, str(project))
    monkeypatch.setenv(DROPBOX_DIR_ENV, str(dropbox))

    assert project_root() == project
    assert dropbox_root() == dropbox
    assert data_root() == dropbox / "data"
    assert results_root() == dropbox / "results"
    assert project_path("data", "raw", "input.csv") == dropbox / "data" / "raw" / "input.csv"
    assert project_path("results", "runs") == dropbox / "results" / "runs"
    assert project_path("configs", "models.yaml") == project / "configs" / "models.yaml"


def test_roots_are_discovered_without_environment(monkeypatch) -> None:
    monkeypatch.delenv(PROJECT_DIR_ENV, raising=False)
    monkeypatch.delenv(DROPBOX_DIR_ENV, raising=False)

    discovered_project = project_root()
    assert discovered_project.name == "NKPC_HSA_MCMC"
    assert (discovered_project / "pyproject.toml").is_file()
    assert dropbox_root().name == discovered_project.name
    assert (dropbox_root() / "data").is_dir()
