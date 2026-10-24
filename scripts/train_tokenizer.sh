#!/usr/bin/env sh
set -eu
python -m model.train_tokenizer --input data/authorized_corpus.txt --output artifacts/tokenizer.json --vocab-size 8192 --min-frequency 2
