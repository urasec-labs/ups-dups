import json
from pathlib import Path

import pytest

from data.prepare_corpus import normalize_document, prepare, redact_sensitive


def test_normalization_and_redaction():
    assert normalize_document("  Türkçe\tmetin  \n\n\n ikinci satır ") == "Türkçe metin\n\nikinci satır"
    redacted, count = redact_sensitive("İletişim: test@example.com, api_key=supersecretvalue123 and +90 555 123 4567")
    assert "test@example.com" not in redacted
    assert "supersecretvalue123" not in redacted
    assert count >= 3


def test_prepare_deduplicates_and_emits_manifest(tmp_path):
    source = tmp_path / "corpus.txt"
    repeated = "Bu, eğitim hattı için hazırlanmış yeterince uzun örnek bir Türkçe belgedir."
    source.write_text("\n\n".join([repeated, repeated, repeated + " İkinci.", repeated + " Üçüncü.", repeated + " Dördüncü."]), encoding="utf-8")
    out = tmp_path / "prepared"
    manifest = prepare(source, out, "user-provided test corpus", "user confirms permission",
                       train_fraction=0.6, val_fraction=0.2, min_chars=20)
    assert manifest["stats"]["duplicates"] == 1
    assert manifest["stats"]["kept_documents"] == 4
    assert (out / "manifest.json").exists()
    assert sum(manifest["split_documents"].values()) == 4
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_prepare_requires_provenance_and_valid_splits(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("A sufficiently long test document for data processing.\n\nAnother sufficiently long test document.", encoding="utf-8")
    with pytest.raises(ValueError, match="source and license"):
        prepare(source, tmp_path / "out", "", "unknown")
    with pytest.raises(ValueError, match="sum to less than 1"):
        prepare(source, tmp_path / "out", "source", "license", 0.8, 0.2)


def test_dataset_can_use_pre_split_files_without_resplitting(tmp_path):
    from training.dataset import TokenBlockDataset
    train = tmp_path / "train.txt"
    val = tmp_path / "validation.txt"
    train.write_text(("Türkçe eğitim belgesi." * 10) + "\n\n" + ("İkinci eğitim belgesi." * 10), encoding="utf-8")
    val.write_text("Validation için ayrı tutulmuş belge." * 10, encoding="utf-8")
    train_ds = TokenBlockDataset(train, seq_len=8, split="all")
    val_ds = TokenBlockDataset(val, seq_len=8, split="all")
    assert len(train_ds) > 0 and len(val_ds) > 0
    x, y = val_ds[0]
    assert x.shape == y.shape == (8,)
