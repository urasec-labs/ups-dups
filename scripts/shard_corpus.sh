#!/usr/bin/env sh
set -eu
python -m data.shard_corpus --input data/prepared/train.txt --output-dir data/shards/train --max-bytes 67108864
