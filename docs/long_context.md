# Long-context engineering status

## Implemented and validated

- RoPE position offsets are applied to cached incremental decoding.
- GQA-style KV head counts are supported when `n_heads` is divisible by `n_kv_heads`.
- Generation uses a per-layer KV cache after prefill and enforces `prompt_tokens + new_tokens <= model.max_seq_len`.
- `model/context_memory.py` estimates KV-cache storage from layers, batch size, sequence length, KV heads, head dimension and bytes per element.

## Not implemented / not validated

- One-million-token execution has **not** been run. A configured `max_seq_len=1000000` is not evidence that it fits memory or has been trained at that length.
- Sliding-window/global-attention hybrids, sequence parallelism, context extension training and long-context retrieval benchmarks are not implemented.
- The attention implementation still performs full attention over the configured context. The SDPA backend may avoid materializing a full score matrix, but compute remains quadratic for dense attention.
- RoPE scaling is not implemented; do not claim extrapolation quality beyond the trained context.

## Memory formula

KV cache bytes (K and V) approximately:

`2 * layers * batch * sequence_tokens * n_kv_heads * head_dim * bytes_per_element`

This excludes model weights, activations, allocator overhead, temporary buffers, and training state. For full dense FP32 attention scores, a conceptual per-layer upper-bound tensor is `batch * n_heads * sequence_tokens^2 * 4` bytes; implementations need not materialize it.

## Benchmark plan before claiming 1M context

1. Validate exact model/tokenizer/checkpoint hashes and context training history.
2. Run 8k, 32k, 128k, 256k, 512k, then 1M token prefill and incremental decode tests on target hardware.
3. Record peak allocated/reserved VRAM, host RAM, prefill latency, decode tokens/s, OOMs, and numerical stability.
4. Run long-context retrieval at multiple positions (early/middle/late), needle-in-haystack, multi-document question answering, and long-document summarization with held-out examples.
5. Report hardware, batch, precision, attention backend, tokenizer, exact context lengths, failures and repeat count. No extrapolation from short tests.
