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


@pytest.mark.parametrize("missing", [PROJECT_DIR_ENV, DROPBOX_DIR_ENV])
def test_required_root_must_be_configured(monkeypatch, missing) -> None:
    monkeypatch.setenv(PROJECT_DIR_ENV, "/tmp/project")
    monkeypatch.setenv(DROPBOX_DIR_ENV, "/tmp/dropbox")
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=missing):
        if missing == PROJECT_DIR_ENV:
            project_root()
        else:
            dropbox_root()
