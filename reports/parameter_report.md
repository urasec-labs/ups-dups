# Parameter and memory report

Actual parameter count is configuration-dependent and computed by `python -m model.parameter_report --config configs/dev.yaml`. This is not a count of a trained checkpoint. `configs/moe.yaml` is illustrative and not a 2.8T model; its dimensions are far beyond the CPU smoke configuration.

Targets are checked explicitly: 2,800,000,000,000 total parameters and 104,000,000,000 approximate active parameters/token. A configuration passes only if computed values meet those thresholds.

For GQA, KV-cache approximation is `2 * layers * context_tokens * kv_heads * head_dim * batch * bytes_per_value`. It excludes weights, activations, temporary buffers, allocator overhead, and optimizer state. Dense attention score tensor estimate is `batch * query_heads * context_tokens^2 * bytes_per_value` if materialized; optimized SDPA may avoid materializing it, but context compute can still be expensive.

One-million-token context has not been trained or tested. Long-context attention design, sequence parallelism, RoPE scaling validation, retrieval/needle-in-haystack and long-document reasoning evaluations are required. Merely changing `max_seq_len` is not support.

## Explicit 1M-token estimate for the supplied dev architecture

For `configs/dev.yaml` (2 layers, 4 query heads, 4 KV heads, head dimension 16, batch 1, bf16/fp16 cache at 2 bytes/value), the estimator gives:

- KV cache: 512,000,000 bytes = about 0.477 GiB.
- Dense attention score tensor, **if materialized**: 8,000,000,000,000 bytes = about 7.28 TiB.

These are only arithmetic estimates for the tiny dev architecture, not evidence that 1M-token attention will run. The model config is limited to 32 tokens and has not been trained for long context. SDPA can avoid materializing the full score tensor, but a naive dense attention computation at one million tokens remains prohibitively expensive; an appropriate long-context architecture/kernel and validated training/evaluation are required.

The two headline targets are not inherently contradictory: 104B active is about 3.71% of 2.8T total, a ratio that can be plausible for a sufficiently expert-heavy MoE with sparse top-k routing and a relatively smaller shared trunk. But the actual layer/expert dimensions must be solved and counted precisely. This repository's development configuration is nowhere near either target, as the executable report shows.


## Training configuration context estimate

For `configs/training.yaml` (6 layers, 4 KV heads, head dimension 32, batch 1, 2 bytes per cache value), the 1M-token KV cache estimate is 3,072,000,000 bytes (about 2.86 GiB), excluding all other memory. A conceptual FP32 dense attention-score tensor per layer would be 32,000,000,000,000 bytes (about 29.10 TiB) if materialized. These figures are arithmetic estimates, not an executable 1M-context result.
