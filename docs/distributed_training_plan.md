# GPU and distributed training plan

## Current state

The active build container had CPU-only PyTorch and no CUDA device. GPU and multi-node performance have not been measured. `training/distributed.py` contains a minimal `torchrun` environment initializer and DDP wrapper; it is a scaffold, not a validated distributed trainer. Expert-parallel MoE all-to-all dispatch is not implemented.

## Recommended staged implementation

1. **Single GPU correctness:** AMP (`bf16` where supported, otherwise `fp16` with GradScaler), gradient accumulation, activation checkpointing, and peak-memory logging. Compare loss against FP32 on a fixed tiny batch.
2. **Single-node DDP:** launch with `torchrun --standalone --nproc_per_node=N`; use `DistributedSampler`, call `sampler.set_epoch(epoch)`, rank-zero-only checkpoint writes, and all-reduce validation metrics.
3. **Checkpoint/restart:** save model, optimizer, scheduler, scaler, step, RNG state, tokenizer hash, data-manifest hash, git revision and configuration. Test restart equivalence.
4. **MoE scaling:** first implement expert parallel groups and token dispatch/combine with tested capacity overflow behavior. Do not equate ordinary DDP with expert parallelism.
5. **Multi-node:** benchmark network bandwidth/latency, topology, NCCL health, checkpoint throughput and failure recovery before scaling the run.
6. **Large-scale orchestration:** only after measured single-node stability, estimate compute, dataset token budget, checkpoint storage, network and run duration. A 2.8T parameter model requires a dedicated cluster and substantial budget.

## Required run log

Record GPU model/count, VRAM, driver/CUDA/PyTorch versions, world size, precision, tokens/s, peak VRAM, data tokens/s, validation loss, checkpoint duration, failed steps and restart tests. Never mark a stage complete without artifacts and measurements.

## Example launch (not validated here)

```bash
torchrun --standalone --nproc_per_node=2 -m training.train --config configs/training.yaml --data data/prepared/train.txt --validation-data data/prepared/validation.txt
```

This launch command is a planning example only; the current training entry point does not yet integrate DDP wrapping or distributed samplers. Use it only after those integrations are implemented and tested.
