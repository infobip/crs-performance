"""Tests for checkpoint metadata helpers."""

import pytest

from stability.checkpoint import (
    build_checkpoint_metadata,
    load_checkpoint_metadata,
    save_checkpoint_metadata,
    validate_checkpoint_metadata,
)


class TestBuildCheckpointMetadata:
    """Tests for build_checkpoint_metadata."""

    def test_basic_structure(self):
        """Metadata dict should contain all expected keys."""
        meta = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={"embedding_size": 64},
            best_hyperparameters={"reg_weight": 250.0},
            seed=42,
            source_hyperparameter_file="hpo/EASE.yaml",
            source_data_files=["train.inter", "test.inter"],
        )
        assert meta["model_name"] == "EASE"
        assert meta["model_family"] == "cf"
        assert meta["dataset_name"] == "redial"
        assert meta["config"] == {"embedding_size": 64}
        assert meta["best_hyperparameters"] == {"reg_weight": 250.0}
        assert meta["seed"] == 42
        assert meta["source_hyperparameter_file"] == "hpo/EASE.yaml"
        assert meta["source_data_files"] == ["train.inter", "test.inter"]
        assert "timestamp" in meta
        assert "git_commit" in meta

    def test_optional_fields_none(self):
        """Optional fields should default cleanly when omitted."""
        meta = build_checkpoint_metadata(
            model_name="SASRec",
            model_family="sequential",
            dataset_name="redial_seq",
            config={},
            best_hyperparameters={},
            seed=0,
        )
        assert meta["source_hyperparameter_file"] is None
        assert meta["source_data_files"] == []

    def test_seed_int_conversion(self):
        """Seed should be stored as an integer."""
        meta = build_checkpoint_metadata(
            model_name="BPR",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=2024,
        )
        assert isinstance(meta["seed"], int)
        assert meta["seed"] == 2024


class TestSaveAndLoadCheckpointMetadata:
    """Tests for save_checkpoint_metadata and load_checkpoint_metadata."""

    def test_round_trip(self, tmp_path):
        """Save and load should preserve metadata exactly."""
        checkpoint = tmp_path / "EASE.pth"
        meta = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={"embedding_size": 64},
            best_hyperparameters={"reg_weight": 250.0},
            seed=42,
        )
        save_checkpoint_metadata(meta, checkpoint)
        loaded = load_checkpoint_metadata(checkpoint)
        assert loaded["model_name"] == "EASE"
        assert loaded["model_family"] == "cf"
        assert loaded["dataset_name"] == "redial"
        assert loaded["config"] == {"embedding_size": 64}
        assert loaded["best_hyperparameters"] == {"reg_weight": 250.0}
        assert loaded["seed"] == 42

    def test_metadata_file_location(self, tmp_path):
        """Sidecar should be placed next to checkpoint with correct name."""
        checkpoint = tmp_path / "models" / "LightGCN.pth"
        meta = build_checkpoint_metadata(
            model_name="LightGCN",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=1,
        )
        sidecar = save_checkpoint_metadata(meta, checkpoint)
        assert sidecar == tmp_path / "models" / "LightGCN.metadata.json"
        assert sidecar.exists()

    def test_load_missing_metadata(self, tmp_path):
        """Loading missing metadata should raise FileNotFoundError."""
        checkpoint = tmp_path / "NonExistent.pth"
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            load_checkpoint_metadata(checkpoint)


class TestValidateCheckpointMetadata:
    """Tests for validate_checkpoint_metadata."""

    def test_matching_metadata(self, tmp_path):
        """Validation should return match when fields are identical."""
        checkpoint = tmp_path / "EASE.pth"
        meta = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={"embedding_size": 64},
            best_hyperparameters={"reg_weight": 250.0},
            seed=42,
        )
        save_checkpoint_metadata(meta, checkpoint)
        result = validate_checkpoint_metadata(meta, checkpoint)
        assert result["status"] == "match"
        assert result["reasons"] == []

    def test_missing_metadata(self, tmp_path):
        """Validation should return missing when sidecar does not exist."""
        checkpoint = tmp_path / "EASE.pth"
        current = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=42,
        )
        result = validate_checkpoint_metadata(current, checkpoint)
        assert result["status"] == "missing"
        assert any("No metadata sidecar found" in r for r in result["reasons"])

    def test_mismatched_model_name(self, tmp_path):
        """Different model names should produce a mismatch."""
        checkpoint = tmp_path / "EASE.pth"
        saved = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=42,
        )
        save_checkpoint_metadata(saved, checkpoint)
        current = build_checkpoint_metadata(
            model_name="LightGCN",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=42,
        )
        result = validate_checkpoint_metadata(current, checkpoint)
        assert result["status"] == "mismatch"
        assert any("model_name" in r for r in result["reasons"])

    def test_mismatched_hyperparameters(self, tmp_path):
        """Different best_hyperparameters should produce a mismatch."""
        checkpoint = tmp_path / "EASE.pth"
        saved = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={"embedding_size": 64},
            best_hyperparameters={"reg_weight": 250.0},
            seed=42,
        )
        save_checkpoint_metadata(saved, checkpoint)
        current = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={"embedding_size": 64},
            best_hyperparameters={"reg_weight": 500.0},
            seed=42,
        )
        result = validate_checkpoint_metadata(current, checkpoint)
        assert result["status"] == "mismatch"
        assert any("best_hyperparameters" in r for r in result["reasons"])

    def test_mismatched_config(self, tmp_path):
        """Different config should produce a mismatch."""
        checkpoint = tmp_path / "EASE.pth"
        saved = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={"embedding_size": 64},
            best_hyperparameters={},
            seed=42,
        )
        save_checkpoint_metadata(saved, checkpoint)
        current = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={"embedding_size": 128},
            best_hyperparameters={},
            seed=42,
        )
        result = validate_checkpoint_metadata(current, checkpoint)
        assert result["status"] == "mismatch"
        assert any("config" in r for r in result["reasons"])

    def test_environment_config_fields_are_ignored(self, tmp_path):
        """CWD/device-only config differences should not force retraining."""
        checkpoint = tmp_path / "EASE.pth"
        saved = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={
                "data_path": "data/recbole",
                "device": "cpu",
                "benchmark_filename": ["train", "valid", "test"],
            },
            best_hyperparameters={"reg_weight": 250.0},
            seed=42,
        )
        save_checkpoint_metadata(saved, checkpoint)
        current = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={
                "data_path": "../data/recbole",
                "device": "cuda",
                "benchmark_filename": ("train", "valid", "test"),
            },
            best_hyperparameters={"reg_weight": 250.0},
            seed=42,
        )
        result = validate_checkpoint_metadata(current, checkpoint)
        assert result["status"] == "match"
        assert result["reasons"] == []

    def test_mismatched_seed(self, tmp_path):
        """Different seed should produce a mismatch."""
        checkpoint = tmp_path / "EASE.pth"
        saved = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=42,
        )
        save_checkpoint_metadata(saved, checkpoint)
        current = build_checkpoint_metadata(
            model_name="EASE",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=43,
        )
        result = validate_checkpoint_metadata(current, checkpoint)
        assert result["status"] == "mismatch"
        assert any("seed" in r for r in result["reasons"])

    def test_directory_created(self, tmp_path):
        """Saving to a nested directory should create parents."""
        checkpoint = tmp_path / "nested" / "dir" / "NeuMF.pth"
        meta = build_checkpoint_metadata(
            model_name="NeuMF",
            model_family="cf",
            dataset_name="redial",
            config={},
            best_hyperparameters={},
            seed=0,
        )
        sidecar = save_checkpoint_metadata(meta, checkpoint)
        assert sidecar.exists()
        loaded = load_checkpoint_metadata(checkpoint)
        assert loaded["model_name"] == "NeuMF"
