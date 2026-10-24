# Test and smoke-run report

Execution date: 2026-10-24, in the active build container. Commands use `python -m ...` to ensure the active interpreter is used.

## Automated tests

- `python -m compileall -q model training inference api data evaluation tests`: passed.
- `python -m pytest -q`: **30 passed** in 1.18 seconds.
- Coverage includes byte tokenizer round-trip, BPE training/save/load/Unicode round-trip, Transformer forward/loss/backprop, RoPE shapes, MoE routing, optimizer step, checkpoint tests, parameter estimate, KV-cache/full-prefix logits equivalence, generation context bounds, streaming dataset blocks, data normalization/deduplication/redaction/splitting, memory formula, DDP helper initialization, checkpoint metadata/strict architecture verification, and API validation/no-checkpoint behavior.

## Executed workflows

- BPE tokenizer trained on bundled smoke data: target vocabulary 280, actual vocabulary 280; Unicode round-trip passed. This data is only smoke-test text, not a useful tokenizer training corpus.
- Corpus sharding command ran and emitted a shard manifest.
- CPU training on `configs/dev.yaml`, `data/smoke_train.txt`, 2 steps: validation loss before 5.565217; after 2 steps 5.557106. This tiny delta does not demonstrate useful language learning.
- Streaming CPU training ran for 1 step with separate path inputs: initial validation mean loss 5.600579, after step 5.586834. The temporary validation file was a workflow smoke test only, not an independent quality benchmark.
- Inference command completed for `Merhaba`; output was `Merhabaaa`. This is not evidence of language quality.
- Local API via FastAPI TestClient with smoke checkpoint: `/health` 200 and model loaded; `/v1/chat` 200; `/v1/chat/stream` 200 and `[DONE]` terminator emitted.
- Tiny Turkish evaluation suite: **0/4 normalized exact matches (0.0)**. The suite is intentionally tiny and the checkpoint is barely trained; no language-quality claim is made.
- Checkpoint verification command validated the bundled checkpoint against `configs/dev.yaml`; unique model parameters: 247,168. The checkpoint contains only 2 training steps.
- ZIP integrity check is performed during packaging.

## Not tested

No CUDA/GPU execution, mixed-precision execution, DDP training, multi-node execution, expert-parallel all-to-all, large-corpus throughput, long-context training, 1M-token context execution, or meaningful Turkish benchmark was run. These require appropriate hardware/data and further implementation.
