# Architecture status — 2026-10-24

| Area | Status | Evidence / limitation |
|---|---|---|
| Decoder-only Transformer | Implemented | `model/transformer.py`; CPU forward/backprop tests pass |
| RoPE | Implemented | Position offsets support cached decode; no validated RoPE scaling |
| GQA-compatible attention | Implemented | `n_heads` must be divisible by `n_kv_heads`; KV heads repeated for attention compute |
| Top-k MoE | Implemented in prototype | Routing, capacity handling and auxiliary load-balancing loss; not distributed expert parallelism |
| KV cache | Implemented | Incremental generation test compares cached and full-prefix logits |
| Byte tokenizer | Implemented | UTF-8 byte tokenizer, fixed vocabulary 260 |
| Trainable byte-level BPE | Implemented | `model.train_tokenizer`; artifact save/load and Unicode round-trip tested; not production-speed optimized |
| Small corpus training | Implemented | CPU smoke run completed; no evidence of general language competence |
| Streaming shard loader | Implemented | Reads shard files lazily; no global shuffle, approximate dedup or distributed partitioning |
| Data sharding and manifest | Implemented | Byte size and SHA-256 checksums; does not establish data rights |
| Independent Turkish smoke suite | Implemented | 4 tiny cases; 0/4 normalized exact matches in the recorded run; not a meaningful benchmark |
| Checkpoint verifier | Implemented | Restricted `weights_only` loading, SHA-256, strict state-dict/config compatibility, and unique parameter count; does not establish language quality |
| Local FastAPI/chat/stream | Implemented | Local TestClient run returned 200 for health, chat and streaming with a tiny checkpoint |
| Mixed precision | Experimental | CUDA-only autocast path added; not testable in CPU-only environment |
| DDP helper | Experimental scaffold | Initialization and wrapper helpers only; main training loop does not yet integrate DDP samplers/wrapping |
| Expert parallel all-to-all | Not implemented | Required for multi-node MoE scaling |
| 1M-token context | Not validated at 1M context | KV cache is present, but full attention compute, training, RoPE extension and evaluation are not solved |
| 2.8T total / 104B active | Not implemented | Current dev config computes 247,168 total and ~148,864 active/token |
