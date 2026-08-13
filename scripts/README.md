# Helper scripts

Python utilities for working with the `.tiktoken` files this trainer produces.

```bash
pip install -r scripts/requirements.txt
```

## `test_tokenizer.py`

Loads a trained vocabulary with OpenAI's `tiktoken` and checks that encode/decode
round-trips. Run this first after any training run.

```bash
python scripts/test_tokenizer.py tokenizer.tiktoken
python scripts/test_tokenizer.py tokenizer.tiktoken examples/special_tokens.txt
```

## `vocab_viewer.py`

Terminal UI for browsing a vocabulary — search by token text or ID, inspect the raw
bytes behind each entry. Useful for sanity-checking what a training run actually
learned.

```bash
python scripts/vocab_viewer.py tokenizer.tiktoken --special-tokens examples/special_tokens.txt
```

## `tiktoken_to_hf.py`

Converts a `.tiktoken` file to a HuggingFace `PreTrainedTokenizerFast`, then verifies
that both tokenizers produce identical IDs on a set of test strings before writing the
output directory.

```bash
python scripts/tiktoken_to_hf.py tokenizer.tiktoken -s examples/special_tokens.txt -o hf_tokenizer
```

## `convert_hf_to_tiktoken.py`

The other direction: turns a HuggingFace BPE `tokenizer.json` into `.tiktoken` format.
Its main use is producing a warm-start file from an existing model's tokenizer — see
`examples/warm_start.yaml`.

```bash
python scripts/convert_hf_to_tiktoken.py ./hf_tokenizer/tokenizer.json ./base.tiktoken
```

## `convert_parquet_tree.py`

Corpus prep. The trainer reads parquet files by looking for a column named `text` and
ignores any file without one. This script walks a directory tree, extracts a chosen
column from every `.parquet` and `.jsonl.zst` file, renames it to `text`, and mirrors
the tree to an output directory.

```bash
python scripts/convert_parquet_tree.py -i /data/raw -o /data/corpus \
    --input_column content --output_column text
```
