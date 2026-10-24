import pytest
import torch
from training.checkpoint import save_checkpoint
from training.verify_checkpoint import inspect_checkpoint


def test_checkpoint_verifier_reports_hash_and_tensor_count(tmp_path, tiny_model):
    path = tmp_path / "model.pt"
    opt = torch.optim.AdamW(tiny_model.parameters())
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  vocab_size: 260\n  d_model: 32\n  n_layers: 2\n  n_heads: 4\n  n_kv_heads: 2\n  d_ff: 64\n  max_seq_len: 32\n  num_experts: 4\n  top_k: 2\n", encoding="utf-8"
    )
    save_checkpoint(path, tiny_model, opt, 3, str(config))
    report = inspect_checkpoint(str(path))
    assert report["status"] == "checkpoint_architecture_compatible"
    assert len(report["sha256"]) == 64
    assert report["state_tensor_count"] > 0
    assert report["unique_model_parameter_elements"] == sum(p.numel() for p in tiny_model.parameters())
    assert report["step"] == 3


def test_checkpoint_verifier_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_checkpoint(str(tmp_path / "missing.pt"))
