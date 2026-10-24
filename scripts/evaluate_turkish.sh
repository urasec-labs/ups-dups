#!/usr/bin/env sh
set -eu
python -m evaluation.run_suite --checkpoint checkpoints/latest.pt --suite evaluation/turkish_suite.jsonl --output reports/turkish_eval.json
