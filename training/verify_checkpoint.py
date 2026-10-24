"""Safely inspect a local Ups-dups checkpoint without constructing a model.

This validates container structure and tensor metadata only. It does not establish
that weights are useful, trained sufficiently, or trustworthy for deployment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from model.config import load_config
from model.transformer import UpsDupsLM


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(checkpoint_path: str, config_override: str | None = None) -> dict:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # weights_only avoids unrestricted pickle object loading.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    required = {"model", "step", "config_path"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"checkpoint missing required keys: {missing}")
    if not isinstance(payload["model"], dict) or not payload["model"]:
        raise ValueError("checkpoint model state must be a non-empty mapping")
    step = payload["step"]
    if not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    config_path = Path(config_override or payload["config_path"])
    if not config_path.is_file():
        raise FileNotFoundError(
            f"model config not found: {config_path}; pass --config to specify the matching config"
        )
    raw, cfg = load_config(str(config_path))
    state = payload["model"]
    tensor_count = 0
    state_elements_including_aliases = 0
    non_tensor = []
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            non_tensor.append(str(key))
            continue
        tensor_count += 1
        state_elements_including_aliases += value.numel()
    if non_tensor:
        raise ValueError(f"model state contains non-tensor entries: {non_tensor[:5]}")
    model = UpsDupsLM(cfg)
    model.load_state_dict(state, strict=True)
    unique_parameter_elements = sum(parameter.numel() for parameter in model.parameters())
    return {
        "status": "checkpoint_architecture_compatible",
        "checkpoint": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "step": step,
        "config": str(config_path.resolve()),
        "model_config": raw["model"],
        "configured_vocab_size": cfg.vocab_size,
        "state_tensor_count": tensor_count,
        "unique_model_parameter_elements": unique_parameter_elements,
        "state_tensor_elements_including_tied_aliases": state_elements_including_aliases,
        "optimizer_state_present": payload.get("optimizer") is not None,
        "warning": "This confirms strict state-dict compatibility with the supplied config; it does not prove model quality or useful training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--config", default=None, help="Override the config path stored in the checkpoint")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    report = inspect_checkpoint(args.checkpoint, args.config)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
