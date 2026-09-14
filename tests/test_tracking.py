import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from recforge.tracking import GitState, RuntimeEnvironment, record_run

STARTED = datetime(2026, 1, 1, tzinfo=UTC)
ENVIRONMENT = RuntimeEnvironment(
    python_version="3.12.0",
    platform="TestOS",
    machine="test-machine",
    processor="test-processor",
)
GIT_STATE = GitState(commit="a" * 40, dirty=False)


def test_record_run_writes_hashed_reproducible_bundle(tmp_path: Path) -> None:
    run_directory = record_run(
        output_root=tmp_path,
        repository_root=tmp_path,
        experiment="smoke",
        metrics={"recall@10": 0.5},
        config={"k": 10},
        dataset_fingerprints={"fixture": "sha256:abc"},
        seed=7,
        started_at=STARTED,
        finished_at=STARTED + timedelta(seconds=2),
        duration_seconds=1.75,
        command=["recforge-smoke"],
        run_id="fixed-run",
        git_state=GIT_STATE,
        environment=ENVIRONMENT,
    )
    manifest = json.loads((run_directory / "manifest.json").read_text())
    metrics = (run_directory / "metrics.json").read_bytes()
    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "fixed-run"
    assert manifest["git"] == {"commit": "a" * 40, "dirty": False}
    assert manifest["metrics_sha256"] == hashlib.sha256(metrics).hexdigest()


def _record_fixed_run(output_root: Path) -> Path:
    return record_run(
        output_root=output_root,
        repository_root=output_root,
        experiment="smoke",
        metrics={"value": 1},
        config={},
        dataset_fingerprints={"fixture": "v1"},
        seed=0,
        started_at=STARTED,
        finished_at=STARTED,
        duration_seconds=0.0,
        command=["test"],
        run_id="same-run",
        git_state=GIT_STATE,
        environment=ENVIRONMENT,
    )


def test_record_run_refuses_to_overwrite_existing_bundle(tmp_path: Path) -> None:
    _record_fixed_run(tmp_path)
    with pytest.raises(FileExistsError):
        _record_fixed_run(tmp_path)


def test_record_run_rejects_invalid_temporal_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not precede"):
        record_run(
            output_root=tmp_path,
            repository_root=tmp_path,
            experiment="smoke",
            metrics={"value": 1},
            config={},
            dataset_fingerprints={"fixture": "v1"},
            seed=0,
            started_at=STARTED,
            finished_at=STARTED - timedelta(seconds=1),
            duration_seconds=0.0,
            git_state=GIT_STATE,
            environment=ENVIRONMENT,
        )
