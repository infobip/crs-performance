"""Tests for candidate pool generation provenance helpers."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from stability.checkpoint import save_checkpoint_metadata

# Import the helper functions from the generation script by executing
# the module spec so that the sys.path manipulation in the script is
# not required during testing.

_spec = importlib.util.spec_from_file_location(
    "generate_candidate_pools",
    Path(__file__).parent.parent / "scripts" / "13_generate_candidate_pools.py",
)
if _spec is None or _spec.loader is None:
    msg = "Could not import scripts/13_generate_candidate_pools.py"
    raise ImportError(msg)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

build_pool_metadata = _gen.build_pool_metadata
save_pool_metadata = _gen.save_pool_metadata
save_candidate_sidecar = _gen.save_candidate_sidecar
_check_checkpoint_metadata_status = _gen._check_checkpoint_metadata_status  # noqa: SLF001


def _write_sidecar(checkpoint_path: Path, metadata: dict) -> Path:
    """Write a metadata sidecar next to a checkpoint file."""
    return save_checkpoint_metadata(metadata, checkpoint_path)


class TestCheckCheckpointMetadataStatus:
    def test_present(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "EASE.pth"
        ckpt.write_text("dummy")
        meta = {"model_name": "EASE", "model_family": "cf"}
        _write_sidecar(ckpt, meta)
        result = _check_checkpoint_metadata_status(ckpt)
        assert result["status"] == "present"
        assert result["metadata"] == meta

    def test_missing(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "EASE.pth"
        ckpt.write_text("dummy")
        result = _check_checkpoint_metadata_status(ckpt)
        assert result["status"] == "missing"
        assert result["metadata"] is None


class TestBuildPoolMetadata:
    def test_fields_and_types(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "EASE.pth"
        ckpt.write_text("dummy")
        meta = build_pool_metadata(
            model_name="EASE",
            model_type="cf",
            top_k=250,
            checkpoint_path=ckpt,
            data_paths={"templates": "t.jsonl"},
        )
        assert meta["retriever_type"] == "cf"
        assert meta["retriever_model"] == "EASE"
        assert meta["n_candidates"] == 250
        assert meta["source_checkpoint_path"] == str(ckpt)
        assert meta["checkpoint_metadata_status"] == "missing"
        assert meta["data_paths"] == {"templates": "t.jsonl"}
        assert meta["git_commit"] is None or isinstance(meta["git_commit"], str)

        # Timestamp should be parseable UTC ISO
        ts = datetime.fromisoformat(meta["timestamp"])
        assert ts.tzinfo == timezone.utc

    def test_with_checkpoint_metadata(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "SASRec.pth"
        ckpt.write_text("dummy")
        sidecar = {"model_name": "SASRec", "seed": 42}
        _write_sidecar(ckpt, sidecar)
        meta = build_pool_metadata(
            model_name="SASRec",
            model_type="sequential",
            top_k=250,
            checkpoint_path=ckpt,
            data_paths={},
        )
        assert meta["checkpoint_metadata_status"] == "present"
        assert meta["checkpoint_metadata"] == sidecar


class TestSavePoolMetadata:
    def test_round_trip(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "pool.jsonl"
        meta = {"retriever_model": "EASE", "n_candidates": 250}
        path = save_pool_metadata(meta, jsonl_path)
        assert path == jsonl_path.with_suffix(".metadata.json")
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == meta


class TestSaveCandidateSidecar:
    def test_round_trip(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "pool.jsonl"
        prov = [
            {
                "candidate_item_ids": [1, 2],
                "candidate_titles": ["A", "B"],
                "candidate_scores": [0.9, 0.8],
            },
            {
                "candidate_item_ids": [3],
                "candidate_titles": ["C"],
                "candidate_scores": [0.7],
            },
        ]
        path = save_candidate_sidecar(prov, jsonl_path)
        assert path == jsonl_path.with_suffix(".candidates.jsonl")
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == prov[0]
        assert json.loads(lines[1]) == prov[1]

    def test_empty(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "pool.jsonl"
        path = save_candidate_sidecar([], jsonl_path)
        assert path.exists()
        assert path.read_text() == ""
