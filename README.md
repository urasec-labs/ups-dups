# Ups-dups Enterprise

Independent, from-scratch language-model engineering scaffold: compact decoder-only Transformer, optional top-k Mixture-of-Experts, byte tokenizer, CPU training/evaluation, checkpoints, autoregressive sampling, local FastAPI API and browser UI.

> **Honest status:** this is a working engineering prototype, not a pretrained foundation model. No large-scale corpus was downloaded and no production model was trained. The 2.8T total-parameter, ~104B active-parameter, and 1,000,000-token context values are targets, not implemented/validated capabilities. See `reports/`.

## Quick start (CPU)

Python 3.11+ recommended (Python 3.13 works in the build environment):

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python -m training.train --config configs/dev.yaml --data data/smoke_train.txt --steps 20
python -m inference.generate --checkpoint checkpoints/latest.pt --prompt "Merhaba, dünya" --max-new-tokens 40
UPS_DUPS_CHECKPOINT="$PWD/checkpoints/latest.pt" uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The API refuses generation if no valid checkpoint is loaded; it does not fabricate replies.

## Train on authorized text

```bash
python -m training.train --config configs/training.yaml --data path/to/authorized_corpus.txt --steps 1000
```

Use UTF-8 text you are authorized to use. The included tiny smoke corpus is only for pipeline testing, not pretraining. For larger files, prepare independent splits with `python -m data.prepare_corpus --input authorized.txt --output-dir data/prepared --source "exact source/version" --license "verified license or permission"`, then shard them with `python -m data.shard_corpus`. The streaming loader reduces whole-corpus RAM use but does not yet provide global shuffle, near-duplicate detection, or distributed sharding.

## Verify a checkpoint

Inspect a local checkpoint using restricted PyTorch loading, validate its required metadata, and check strict compatibility with its model config:

```bash
python -m training.verify_checkpoint --checkpoint checkpoints/latest.pt --config configs/dev.yaml --output reports/checkpoint_verification.json
```

The report includes SHA-256, training step, tensor metadata, and unique model parameter count (tied weights counted once). Structural compatibility is not a measure of language quality. Never load untrusted checkpoint files.

## Parameter and context analysis

```bash
python -m model.parameter_report --config configs/dev.yaml
python -m model.parameter_report --config configs/dev.yaml --context 1000000
```

Parameter counts derive from config; they do not imply a trained model. `max_seq_len` is a trained/validated limit, not a value to change casually.

## Status

- **Implemented:** compact decoder Transformer, RoPE, GQA-compatible attention, optional top-k MoE, load-balancing loss, byte tokenizer, trainable byte-level BPE, short-context KV-cache decoding, CPU training, basic streaming shard loader, corpus normalization/dedup/redaction/split manifest, context-memory estimator, Turkish smoke-suite runner, local API/UI.
- **Experimental:** CUDA autocast path and DDP helper scaffold; neither GPU training nor distributed training was available for validation. Streaming does not include global shuffle or near-deduplication.
- **Not implemented/validated:** 2.8T model, 104B active deployment, validated 1M context, RoPE scaling, sliding-window/global attention hybrid, sequence parallelism, distributed expert-parallel all-to-all, and production auth/rate limiting.

## Security

Bind the API to loopback unless you add authentication and deployment hardening. Never load untrusted checkpoints. Logs omit prompt text. Use lawfully licensed data.

## License

MIT for original repository code; third-party dependencies retain their licenses.

## Additional engineering modules

- Train a byte-level BPE tokenizer: `python -m model.train_tokenizer --input data/authorized_corpus.txt --output artifacts/tokenizer.json --vocab-size 4096`. Update `model.vocab_size` in your YAML config to exactly match the resulting `vocab_size` before training with `--tokenizer artifacts/tokenizer.json`.
- Stream text shards without loading the entire corpus into memory: `python -m training.train --config configs/dev.yaml --data data/prepared/train.txt --validation-data data/prepared/validation.txt --streaming --steps 20`. For a real run, ensure both split files have enough tokens and have been independently prepared.
- Split large files into bounded-size shards: `python -m data.shard_corpus --input data/prepared/train.txt --output-dir data/shards/train --max-bytes 67108864`.
- Estimate context KV cache: `python -c "from model.config import load_config; from model.context_memory import estimate; print(estimate(load_config('configs/training.yaml')[1]))"`.
- Run the tiny Turkish smoke suite: `python -m evaluation.run_suite --checkpoint checkpoints/latest.pt --suite evaluation/turkish_suite.jsonl`. It is a smoke suite, not a validated benchmark.
- `docs/long_context.md` records KV-cache status and the 1M-context validation plan. `docs/distributed_training_plan.md` records the GPU/distributed roadmap and limitations.

### Tokenizer compatibility warning

The default `ByteTokenizer` has vocabulary size 260. A trained BPE tokenizer has a larger learned vocabulary. The model config must use the exact BPE vocabulary size, and training, inference, evaluation, and API must use the same tokenizer artifact. Do not swap tokenizers after training. For API use, set `UPS_DUPS_TOKENIZER` to the tokenizer JSON path when the checkpoint was trained with BPE.

### Distribution status

Minimal DDP helpers are included, but the main training loop does not yet integrate DDP samplers/wrapping or expert-parallel all-to-all dispatch. Multi-GPU/multi-node training remains planned and unvalidated.
