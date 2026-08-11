"""Checkpoint metadata helpers for RecBole model provenance and validation."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONFIG_FIELDS_IGNORED_FOR_VALIDATION = frozenset({"data_path", "device"})


def _get_git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    git_path = shutil.which("git")
    if git_path is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603
            [git_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def build_checkpoint_metadata(
    model_name: str,
    model_family: str,
    dataset_name: str,
    config: dict[str, Any],
    best_hyperparameters: dict[str, Any],
    seed: int,
    source_hyperparameter_file: str | Path | None = None,
    source_data_files: list[str] | None = None,
) -> dict[str, Any]:
    """Build a metadata dictionary for a trained model checkpoint.

    Args:
        model_name: Canonical model name (e.g., ``"EASE"``).
        model_family: Model category (e.g., ``"cf"``, ``"sequential"``).
        dataset_name: RecBole dataset name (e.g., ``"redial"``).
        config: Full RecBole config dictionary used for training.
        best_hyperparameters: Best hyperparameters from HPO.
        seed: Random seed used during training.
        source_hyperparameter_file: Path to the HPO result file that
            produced the best hyperparameters.
        source_data_files: List of data file paths used for training.

    Returns:
        Dictionary with model metadata, training provenance, and timestamp.

    """
    return {
        "model_name": model_name,
        "model_family": model_family,
        "dataset_name": dataset_name,
        "config": config,
        "best_hyperparameters": best_hyperparameters,
        "seed": seed,
        "source_hyperparameter_file": (
            str(source_hyperparameter_file) if source_hyperparameter_file else None
        ),
        "source_data_files": list(source_data_files) if source_data_files else [],
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
    }


def save_checkpoint_metadata(
    metadata: dict[str, Any],
    checkpoint_path: str | Path,
) -> Path:
    """Save metadata next to the checkpoint file.

    Writes ``ModelName.metadata.json`` alongside ``ModelName.pth``.

    Args:
        metadata: Metadata dictionary from :func:`build_checkpoint_metadata`.
        checkpoint_path: Path to the ``.pth`` checkpoint file.

    Returns:
        Path to the written metadata sidecar file.

    """
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.debug("Saved checkpoint metadata to %s", metadata_path)
    return metadata_path


def load_checkpoint_metadata(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load metadata from the sidecar JSON next to the checkpoint.

    Args:
        checkpoint_path: Path to the ``.pth`` checkpoint file.

    Returns:
        Metadata dictionary.

    Raises:
        FileNotFoundError: If the sidecar metadata file does not exist.

    """
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        msg = f"Metadata file not found: {metadata_path}"
        raise FileNotFoundError(msg)
    with metadata_path.open() as f:
        return json.load(f)


def validate_checkpoint_metadata(
    current: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Compare current metadata-relevant fields with saved metadata.

    Validates that the saved sidecar matches the expected ``current``
    fields. Returns a status of ``match``, ``missing``, or ``mismatch``
    together with explanatory reasons.

    Args:
        current: Expected metadata dictionary.
        checkpoint_path: Path to the ``.pth`` checkpoint file.

    Returns:
        Dictionary with keys:
        - ``status``: ``"match"``, ``"missing"``, or ``"mismatch"``
        - ``reasons``: list of human-readable discrepancy strings

    """
    checkpoint_path = Path(checkpoint_path)
    metadata_path = checkpoint_path.with_suffix(".metadata.json")

    if not metadata_path.exists():
        return {
            "status": "missing",
            "reasons": [f"No metadata sidecar found at {metadata_path}"],
        }

    saved = load_checkpoint_metadata(checkpoint_path)
    reasons: list[str] = []

    scalar_fields = [
        "model_name",
        "model_family",
        "dataset_name",
        "seed",
    ]
    for field in scalar_fields:
        saved_val = saved.get(field)
        current_val = current.get(field)
        if saved_val != current_val:
            reasons.append(
                f"Field '{field}' mismatch: saved={saved_val!r}, current={current_val!r}"
            )

    saved_config = _canonicalize_config_for_validation(saved.get("config", {}))
    current_config = _canonicalize_config_for_validation(current.get("config", {}))
    if saved_config != current_config:
        reasons.append(
            f"Field 'config' mismatch: saved={saved_config!r}, current={current_config!r}"
        )

    dict_fields = ["best_hyperparameters"]
    for field in dict_fields:
        saved_val = saved.get(field)
        current_val = current.get(field)
        if saved_val != current_val:
            reasons.append(
                f"Field '{field}' mismatch: saved={saved_val!r}, current={current_val!r}"
            )

    if reasons:
        return {"status": "mismatch", "reasons": reasons}

    return {"status": "match", "reasons": []}


def _canonicalize_config_for_validation(value: Any) -> Any:  # noqa: ANN401
    """Normalize config fields for environment-independent checkpoint validation."""
    if isinstance(value, dict):
        return {
            key: _canonicalize_config_for_validation(val)
            for key, val in sorted(value.items())
            if key not in CONFIG_FIELDS_IGNORED_FOR_VALIDATION
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_config_for_validation(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
