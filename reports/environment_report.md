# Environment report

Observed in the build container on 2026-10-24:
- OS/platform: Linux x86_64, kernel reported as `6.18.55-x86_64`, glibc 2.41.
- Python: 3.13.5.
- CPU: 3 logical CPUs exposed to the process.
- RAM: 6,237,200,384 bytes total (~5.81 GiB); at inspection about 4.84 GB available.
- Disk: overlay filesystem 32G total, about 30G available at inspection.
- PyTorch: 2.10.0+cpu.
- CUDA: unavailable; `torch.cuda.is_available()` false, zero GPUs; `nvcc` and `nvidia-smi` not present.
- FastAPI, Pydantic, PyYAML, uvicorn, pytest, httpx and SentencePiece were present at inspection; `tokenizers` and `ruff` were not. Delivered tokenizer is the dependency-free byte tokenizer, not SentencePiece.
- `/openai/project` had no source files and `/mnt/data` was empty before this project was created.

These are measurements of the build container, not the user's own computer. Test results are recorded separately after execution.
