#!/usr/bin/env bash
set -euo pipefail
python -m training.train --config configs/dev.yaml --data data/smoke_train.txt --steps 20
