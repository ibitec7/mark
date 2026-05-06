import json
import os
import math
import tempfile

import pytest
import torch

from src.perplexity import (
    ModelEntry,
    apply_checkpoint_dir_overrides,
    build_arg_parser,
    discover_datasets,
    main,
    parse_checkpoint_dir_overrides,
    resolve_runtime_path,
    BENCHMARKS_DIR,
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


BENCHMARK_METRICS = ("raw_nll", "raw_ppl", "raw_bpb", "weighted_nll", "weighted_ppl", "weighted_bpb")


class TestPerplexitySmokeRTX3050:
    """Smoke test: chebyshev stage 1 × ptb, limit_val_batches=10.

    Designed to run on RTX 3050 (4 GB VRAM) with batch_size=1.
    """

    @pytest.fixture
    def results_dir(self, tmp_path):
        return str(tmp_path / "benchmark_results")

    @pytest.fixture
    def single_model(self):
        return [
            ModelEntry("chebyshev", 1, "configs/benchmark_config_chebyshev_stage1.yaml", ""),
        ]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_single_model_single_dataset(self, single_model, results_dir):
        """Run chebyshev stage1 on ptb with 10 batches and verify output."""

        # Only use ptb dataset
        ptb_dir = os.path.join(BENCHMARKS_DIR, "ptb")
        if not os.path.exists(ptb_dir):
            pytest.skip(f"PTB benchmark data not found at {ptb_dir}")

        results = main(
            models=single_model,
            benchmarks_dir=BENCHMARKS_DIR,
            results_dir=results_dir,
            limit_val_batches=10,
        )

        assert "chebyshev_stage1" in results
        model_results = results["chebyshev_stage1"]

        # Should have at least ptb (may have others depending on discover_datasets)
        assert len(model_results) > 0, "No dataset results produced"

        # Check that at least one dataset has valid metrics
        has_valid = False
        for ds_name, metrics in model_results.items():
            if "error" in metrics:
                continue
            has_valid = True
            for key in BENCHMARK_METRICS:
                assert key in metrics, f"Missing metric '{key}' for {ds_name}"
                val = metrics[key]
                assert isinstance(val, (int, float)), f"Metric '{key}' is not a number: {type(val)}"
                assert math.isfinite(val), f"Metric '{key}' is not finite: {val}"

        assert has_valid, "No dataset produced valid metrics"

    def test_discover_datasets(self):
        """Verify dataset discovery finds expected benchmarks."""
        if not os.path.exists(BENCHMARKS_DIR):
            pytest.skip(f"Benchmarks directory not found: {BENCHMARKS_DIR}")

        datasets = discover_datasets(BENCHMARKS_DIR)
        assert len(datasets) >= 1
        # ptb should always be present
        assert "ptb" in datasets
        assert ".cache" not in datasets

    def test_model_registry_paths_exist(self, single_model):
        """Verify that config files referenced in the registry exist."""
        for entry in single_model:
            assert os.path.exists(entry.config_path), f"Config not found: {entry.config_path}"

    def test_checkpoint_exists(self, single_model):
        """Verify that checkpoint files exist for the smoke-test model."""
        for entry in single_model:
            assert os.path.exists(entry.checkpoint_path), f"Checkpoint not found: {entry.checkpoint_path}"


class TestPerplexityUnit:
    """Unit tests for perplexity pipeline utilities (no GPU required)."""

    def test_model_entry_properties(self):
        entry = ModelEntry("chebyshev", 2, "configs/benchmark_config_chebyshev_stage2.yaml", "-v1")
        assert entry.name == "chebyshev_stage2"
        assert entry.checkpoint_dirs == [
            "./models/hydra_mark_chebyshev",
            "./models/hydra_mark_chebyshev_apr",
        ]

    def test_model_entry_uses_existing_apr_checkpoint(self, monkeypatch):
        entry = ModelEntry("chebyshev", 2, "configs/benchmark_config_chebyshev_stage2.yaml", "-v1")

        def fake_exists(path):
            return path == "./models/hydra_mark_chebyshev_apr/best_hydra_mark-v1.ckpt"

        monkeypatch.setattr(os.path, "exists", fake_exists)

        assert entry.checkpoint_path == "./models/hydra_mark_chebyshev_apr/best_hydra_mark-v1.ckpt"

    def test_model_entry_falls_back_to_best_checkpoint(self, monkeypatch):
        entry = ModelEntry("hypernet", 3, "configs/benchmark_config_hypernet_stage3.yaml", "-v2")

        monkeypatch.setattr(os.path, "exists", lambda path: False)
        monkeypatch.setattr(
            "src.perplexity.get_best_checkpoint",
            lambda checkpoint_dir: "./models/hydra_mark_hypernet_apr/last.ckpt"
            if checkpoint_dir == "./models/hydra_mark_hypernet_apr"
            else None,
        )

        assert entry.checkpoint_path == "./models/hydra_mark_hypernet_apr/last.ckpt"

    def test_parse_checkpoint_dir_overrides(self):
        overrides = parse_checkpoint_dir_overrides(
            [
                "chebyshev=/models/run_a",
                "chebyshev=/models/run_b",
                "dct=/models/run_c",
            ]
        )

        assert overrides == {
            "chebyshev": ["/models/run_a", "/models/run_b"],
            "dct": ["/models/run_c"],
        }

    def test_parse_checkpoint_dir_overrides_rejects_invalid_values(self):
        with pytest.raises(ValueError, match="KERNEL=DIR"):
            parse_checkpoint_dir_overrides(["/models/run_a"])

        with pytest.raises(ValueError, match="Unsupported kernel"):
            parse_checkpoint_dir_overrides(["unknown=/models/run_a"])

    def test_apply_checkpoint_dir_overrides(self):
        models = [
            ModelEntry("chebyshev", 1, "cfg_a", ""),
            ModelEntry("dct", 1, "cfg_b", ""),
        ]

        updated_models = apply_checkpoint_dir_overrides(
            models,
            {"chebyshev": ["/models/custom_chebyshev"]},
        )

        assert updated_models[0].checkpoint_dirs == ["/models/custom_chebyshev"]
        assert updated_models[1].checkpoint_dirs == [
            "./models/hydra_mark_dct",
            "./models/hydra_mark_dct_apr",
        ]

    def test_resolve_runtime_path_translates_host_repo_path(self, monkeypatch):
        host_path = "/host/home/shadeform/experiment/models/hydra_mark_chebyshev_apr"
        runtime_path = "/workspace/models/hydra_mark_chebyshev_apr"

        monkeypatch.setattr("src.perplexity.PROJECT_ROOT", Path("/workspace"))
        monkeypatch.setattr(
            Path,
            "exists",
            lambda self: str(self) in {runtime_path},
        )

        assert resolve_runtime_path(host_path) == runtime_path

    def test_build_arg_parser_accepts_repeated_checkpoint_dirs(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--checkpoint-dir",
                "chebyshev=/models/run_a",
                "--checkpoint-dir",
                "dct=/models/run_b",
            ]
        )

        assert args.checkpoint_dir == ["chebyshev=/models/run_a", "dct=/models/run_b"]

    def test_discover_datasets_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            discover_datasets("/nonexistent/path")

    def test_discover_datasets_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No dataset subdirectories"):
            discover_datasets(str(tmp_path))
