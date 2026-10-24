# Data provenance and policy

No external dataset is bundled or downloaded. `smoke_train.txt` is a small example corpus for exercising the pipeline only; it is not pretraining data and must not support claims of language competence.

For real training, record source/provider, collection date, license/terms, language, document count, filtering steps, and hashes/manifests. Verify the license permits training and any redistribution of resulting weights. Exclude secrets and private communications; apply a lawful basis and safeguards to personal data. Deduplicate before splitting and keep validation/test documents separate to reduce leakage. Avoid scraping contrary to site terms. Do not commit raw corpora to version control.

The default tokenizer is deterministic byte encoding. A trainable byte-level BPE implementation is available in `model.train_tokenizer`; it is a correctness-focused prototype and must be trained only on authorized text. Use the exact same saved tokenizer artifact in training, inference, evaluation, and API.

## Source registry and corpus-scale preparation

`data/sources.yaml` lists candidate source categories and the provenance fields required for review. It deliberately does not claim that an arbitrary dump is approved for training. Review the exact snapshot, terms, license, attribution requirements, privacy risks, and intended model-training use before ingesting it. Public accessibility alone is not permission.

`data/shard_corpus.py` splits a UTF-8 file into bounded-size shards and writes byte counts and SHA-256 checksums. It preserves the source's rights obligations and should be used after corpus preparation/deduplication. `training.dataset.StreamingTokenBlockDataset` reads shards lazily; it avoids loading the entire text corpus into RAM, but currently does not provide corpus-wide shuffling or approximate deduplication.
