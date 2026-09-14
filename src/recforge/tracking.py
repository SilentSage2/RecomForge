"""Small, dependency-free run manifests for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class GitState:
    commit: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    python_version: str
    platform: str
    machine: str
    processor: str


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: int
    run_id: str
    experiment: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    seed: int
    command: tuple[str, ...]
    git: GitState
    environment: RuntimeEnvironment
    config: Mapping[str, JsonValue]
    dataset_fingerprints: Mapping[str, str]
    metrics_sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and artifact comparison."""
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def collect_git_state(repository_root: Path) -> GitState:
    """Read the exact revision and whether tracked or untracked changes exist."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return GitState(commit=commit, dirty=bool(porcelain))


def collect_runtime_environment() -> RuntimeEnvironment:
    """Collect a compact, non-identifying runtime and hardware summary."""
    return RuntimeEnvironment(
        python_version=platform.python_version(),
        platform=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
    )


def new_run_id(experiment: str, started_at: datetime) -> str:
    """Create a sortable run ID with enough entropy for concurrent launches."""
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware")
    safe_experiment = "".join(
        character if character.isalnum() or character in "-_" else "-" for character in experiment
    ).strip("-")
    if not safe_experiment:
        raise ValueError("experiment must contain a filename-safe character")
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{safe_experiment}-{uuid.uuid4().hex[:8]}"


def record_run(
    *,
    output_root: Path,
    repository_root: Path,
    experiment: str,
    metrics: Mapping[str, JsonValue],
    config: Mapping[str, JsonValue],
    dataset_fingerprints: Mapping[str, str],
    seed: int,
    started_at: datetime,
    finished_at: datetime,
    duration_seconds: float,
    command: Sequence[str] | None = None,
    run_id: str | None = None,
    git_state: GitState | None = None,
    environment: RuntimeEnvironment | None = None,
) -> Path:
    """Write a new metrics/manifest bundle without overwriting an existing run."""
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        raise ValueError("started_at must be timezone-aware")
    if finished_at.tzinfo is None or finished_at.utcoffset() is None:
        raise ValueError("finished_at must be timezone-aware")
    if finished_at < started_at:
        raise ValueError("finished_at must not precede started_at")
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be non-negative")
    if not dataset_fingerprints:
        raise ValueError("dataset_fingerprints must not be empty")

    resolved_run_id = run_id or new_run_id(experiment, started_at)
    run_directory = output_root / resolved_run_id
    metrics_bytes = canonical_json_bytes(metrics)
    manifest = RunManifest(
        schema_version=1,
        run_id=resolved_run_id,
        experiment=experiment,
        status="completed",
        started_at_utc=started_at.astimezone(UTC).isoformat(),
        finished_at_utc=finished_at.astimezone(UTC).isoformat(),
        duration_seconds=round(duration_seconds, 6),
        seed=seed,
        command=tuple(command if command is not None else sys.argv),
        git=git_state or collect_git_state(repository_root),
        environment=environment or collect_runtime_environment(),
        config=dict(config),
        dataset_fingerprints=dict(dataset_fingerprints),
        metrics_sha256=sha256_bytes(metrics_bytes),
    )
    manifest_bytes = canonical_json_bytes(asdict(manifest))

    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "metrics.json").write_bytes(metrics_bytes)
    (run_directory / "manifest.json").write_bytes(manifest_bytes)
    return run_directory
