#!/usr/bin/env bash
set -euo pipefail
python -m model.parameter_report --config configs/dev.yaml
pytest -q
