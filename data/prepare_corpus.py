"""Prepare a locally supplied UTF-8 text corpus for small-scale LM experiments.

This tool does not download data and does not establish rights to use the input.
The operator must supply and record a valid source/license declaration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
# Conservative redaction of likely phone numbers; may also redact some non-phone numeric IDs.
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
SECRET_PATTERNS = (
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"(?i)\b(api[_ -]?key|access[_ -]?token|client[_ -]?secret)\s*[:=]\s*['\"]?[^\s'\"]{8,}"),
)


def normalize_document(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", " ")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"[\t\r ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def redact_sensitive(text: str) -> tuple[str, int]:
    count = 0
    for pattern in (EMAIL_RE, PHONE_RE, *SECRET_PATTERNS):
        text, n = pattern.subn("[REDACTED]", text)
        count += n
    return text, count


def stable_split(digest: str, train_fraction: float, val_fraction: float) -> str:
    # Hash-based assignment makes splits reproducible and independent of input order.
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_fraction:
        return "train"
    if bucket < train_fraction + val_fraction:
        return "validation"
    return "test"


def prepare(input_path: Path, output_dir: Path, source: str, license_name: str,
            train_fraction: float = 0.90, val_fraction: float = 0.05,
            min_chars: int = 40, max_chars: int = 100_000,
            redact: bool = True) -> dict:
    if not source.strip() or not license_name.strip():
        raise ValueError("source and license declaration are required")
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1 and train_fraction + val_fraction < 1):
        raise ValueError("train_fraction and val_fraction must be positive and sum to less than 1")
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("invalid document length limits")

    raw = input_path.read_text(encoding="utf-8", errors="replace")
    # Blank-line-delimited documents; a line-only corpus is still supported.
    chunks = [x for x in re.split(r"\n\s*\n", raw) if x.strip()]
    if len(chunks) < 2:
        chunks = [line for line in raw.splitlines() if line.strip()]

    seen: set[str] = set()
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    stats = {"input_documents": len(chunks), "empty_or_short": 0, "too_long": 0,
             "duplicates": 0, "redactions": 0, "kept_documents": 0}
    for chunk in chunks:
        doc = normalize_document(chunk)
        if len(doc) < min_chars:
            stats["empty_or_short"] += 1
            continue
        if len(doc) > max_chars:
            stats["too_long"] += 1
            continue
        if redact:
            doc, n = redact_sensitive(doc)
            stats["redactions"] += n
        digest = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        if digest in seen:
            stats["duplicates"] += 1
            continue
        seen.add(digest)
        splits[stable_split(digest, train_fraction, val_fraction)].append(doc)
        stats["kept_documents"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, docs in splits.items():
        # Preserve the dataset loader's blank-line-delimited format.
        (output_dir / f"{name}.txt").write_text("\n\n".join(docs) + ("\n" if docs else ""), encoding="utf-8")
    manifest = {
        "input_file_name": input_path.name,
        "source_declaration": source,
        "license_declaration": license_name,
        "split_method": "sha256(document content) bucket; deterministic",
        "split_fractions": {"train": train_fraction, "validation": val_fraction,
                            "test": 1 - train_fraction - val_fraction},
        "redaction_enabled": redact,
        "stats": stats,
        "split_documents": {name: len(docs) for name, docs in splits.items()},
        "split_warnings": [f"{name} split is empty; provide a larger corpus or adjust split fractions" for name, docs in splits.items() if not docs],
        "warning": "Operator must verify source rights and inspect redaction quality. This manifest is not legal advice.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="UTF-8 source text file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Human-readable data source/provenance declaration")
    parser.add_argument("--license", required=True, dest="license_name", help="License/permission declaration")
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--no-redact", action="store_true", help="Disable automatic PII/secret redaction (not recommended)")
    args = parser.parse_args()
    result = prepare(args.input, args.output_dir, args.source, args.license_name,
                     args.train_fraction, args.val_fraction, args.min_chars, args.max_chars,
                     redact=not args.no_redact)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
