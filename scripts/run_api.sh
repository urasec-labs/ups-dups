#!/usr/bin/env bash
set -euo pipefail
: "${UPS_DUPS_CHECKPOINT:?Set UPS_DUPS_CHECKPOINT to a trained checkpoint path}"
uvicorn api.app:app --host 127.0.0.1 --port 8000
