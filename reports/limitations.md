# Limitations and non-claims

- This repository is an independent model-engineering prototype, not a trained foundation model or a ChatGPT-equivalent system.
- The bundled checkpoint is a tiny smoke-training artifact. Do not present it as pretrained, production-ready, or capable of reliable reasoning.
- The target of 2.8T total parameters and approximately 104B active parameters is not implemented. The dev configuration has 247,168 total and about 148,864 active parameters/token according to the configuration-derived report.
- KV-cache incremental decoding is implemented and tested at short context. One-million-token context is not validated. KV-cache estimates omit model weights, activations, runtime overhead, temporary buffers and training state.
- The trainable BPE implementation is for correctness/experimentation; it is not optimized for production throughput. Always use the exact same tokenizer artifact for training, inference, evaluation and API.
- Candidate data sources are not bundled and are not pre-approved. Public availability is not permission. Verify exact snapshot licenses, attribution, privacy and training rights before ingestion.
- The streaming loader bounds text-corpus memory but does not yet provide global shuffling, near-duplicate detection, data mixing weights, or distributed rank-aware sharding.
- The tiny Turkish suite measures normalized exact match only and currently scores 0/4. It is not a validated benchmark and cannot characterize general language quality.
- DDP helpers and mixed precision are scaffolds/experimental. The main training loop does not yet integrate DDP or expert parallelism; multi-node and GPU execution have not been validated.
- No external LLM API is used. A missing or incompatible checkpoint/tokenizer must fail explicitly rather than return fabricated output.
