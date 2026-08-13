# toktoktok

A high-performance BPE tokenizer trainer that produces vocabularies compatible with OpenAI's [tiktoken](https://github.com/openai/tiktoken) and Hugging Face [tokenizers](https://github.com/huggingface/tokenizers). It trains on trillions of tokens on a single machine, stays inside a memory budget you declare, and writes a `.tiktoken` file that loads directly into either library.

## The experiment behind this repository

This repository is the result of an experiment we ran in late 2025 to answer one question: *can coding agents autonomously solve a production-grade problem from scratch on their own?*

We needed a BPE tokenizer trainer for [our research on vocabulary size in edge LLMs](https://www.liquid.ai/blog/tokenizer-expansion) — the existing options were either too slow, ran out of memory on our corpora, or couldn't train at all. Instead of staffing it, we handed the problem to coding agents: a short specification stating outcomes and constraints ([AGENTS.md](AGENTS.md)), sandboxed access to production data and a large machine, and an external verification harness the agents couldn't influence — the trained vocabulary had to load in `tiktoken` and Hugging Face `tokenizers` and produce identical token IDs in both.

**Every line of code in this repository was written by an agent, and none of it has been read by us.** The full write-up of what happened — why neither agent zero-shot it, why the iteration loop against real data was what got it there, and what became standard practice on our team afterwards — is in the accompanying blog post: [github.com/Liquid4All/toktoktok](https://github.com/Liquid4All/toktoktok) *(blog URL coming soon)*.

## Features

- **tiktoken-compatible output** — the `.tiktoken` file loads directly in `tiktoken` and converts losslessly to a Hugging Face tokenizer
- **Memory-bounded** — reservoir sampling keeps training inside a configurable memory budget, so corpora far larger than RAM stay fairly represented
- **Multi-threaded** — parallel pair counting and merge application across all cores
- **Multi-phase training** — allocate vocabulary budget explicitly across languages and domains instead of letting the largest corpus dominate
- **Warm start** — extend an existing tokenizer (including one converted from Hugging Face) without invalidating its token IDs
- **`.txt` and `.parquet` input** — directories are scanned recursively; parquet files need a `text` column

## Installation

Build from source (requires a [Rust toolchain](https://rustup.rs)):

```bash
cargo build --release
```

The binary lands at `./target/release/toktoktok`.

## Quick start

Train a small tokenizer on the bundled sample corpus:

```bash
./target/release/toktoktok -c examples/from_scratch.yaml
```

Then verify the result loads in `tiktoken` and round-trips:

```bash
pip install -r scripts/requirements.txt
python scripts/test_tokenizer.py tokenizer.tiktoken examples/special_tokens.txt
```

## Configuration

Training is driven entirely by a YAML file passed with `-c` / `--config`. The three configurations below are also in [`examples/`](examples/), runnable as-is.

### Train from scratch

The minimal setup: one phase, one data source.

```yaml
output: ./tokenizer.tiktoken

# Cap on how much training data is held in memory. Anything beyond this is
# reservoir-sampled, so the sample stays representative of the whole corpus.
working_set_mb: 1024

# -1 uses every available core.
threads: -1

verbose: true

# Appended at the end of the vocabulary, after all trained merges. They are not
# written into the .tiktoken file — pass the same list when loading.
special_tokens:
  - "<|pad|>"
  - "<|startoftext|>"
  - "<|endoftext|>"
  - "<|im_start|>"
  - "<|im_end|>"

phases:
  - name: "Sample corpus"
    merges: 2000
    sources:
      # A directory is scanned recursively for .txt and .parquet files.
      - path: ./test_data
```

The final vocabulary is always:

```
256 base bytes + 1,161 hardcoded merges + Σ(phase merges) + special tokens
```

The 1,161 hardcoded merges reserve ranks for all two- and three-digit numbers, runs of spaces and newlines, `\r\n`, common programming operators, and ellipses — so numeric and whitespace encoding is consistent regardless of what the corpus happens to contain. See [AGENTS.md](AGENTS.md) for the full breakdown.

### Multi-phase training

Phases run sequentially and share one merge history: every phase starts from the vocabulary the previous phase ended with. Splitting the run this way is how you control the vocabulary budget per domain — otherwise whichever corpus is largest dominates the merge table.

```yaml
output: ./tokenizer.tiktoken

working_set_mb: 8192
threads: -1

special_tokens:
  - "<|endoftext|>"
  - "<|im_start|>"
  - "<|im_end|>"

phases:
  # Phase 1: general English text gets the largest share.
  - name: "English"
    merges: 30000
    sources:
      - path: /data/corpus/english/wikipedia
      - path: /data/corpus/english/books
      # A single file works too. Parquet files must have a "text" column.
      - file: /data/corpus/english/common_crawl.parquet

  # Phase 2: code, kept separate so it gets a guaranteed budget.
  - name: "Code"
    merges: 15000
    sources:
      - path: /data/corpus/code

  # Phase 3: other languages.
  - name: "Multilingual"
    merges: 5000
    sources:
      - path: /data/corpus/german
      - path: /data/corpus/french
      - path: /data/corpus/spanish
```

### Warm start (extend an existing tokenizer)

Load an existing `.tiktoken` file and append new merges after it. The existing ranks keep their token IDs, so text already tokenized with the base vocabulary encodes identically — this is how you grow a vocabulary for a new domain or language without invalidating an already-trained model's embedding table.

```yaml
output: ./extended_tokenizer.tiktoken

working_set_mb: 4096
threads: -1

warm_start:
  # An existing .tiktoken file — one this trainer produced, or one converted
  # from a Hugging Face tokenizer with scripts/convert_hf_to_tiktoken.py.
  file: ./tokenizer.tiktoken

phases:
  - name: "Extension"
    merges: 4000
    sources:
      # Point this at your new-domain corpus.
      - path: /data/corpus/new_domain

# Special tokens are re-applied to the extended vocabulary, so list every token
# the base tokenizer had plus any new ones — not just the additions.
special_tokens:
  - "<|endoftext|>"
  - "<|im_start|>"
  - "<|im_end|>"
```

To warm-start from an existing Hugging Face model's tokenizer, convert it first:

```bash
python scripts/convert_hf_to_tiktoken.py ./model_dir/tokenizer.json ./base.tiktoken
```

### Option reference

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `output` | yes | — | Path for the trained `.tiktoken` file |
| `working_set_mb` | no | `1024` | Memory budget in MB for the in-memory training sample; larger corpora are reservoir-sampled into it |
| `threads` | no | `-1` | Thread count; `-1` uses all cores |
| `verbose` | no | `false` | Detailed logging |
| `special_tokens` | no | — | Strings appended as atomic tokens at the end of the vocabulary; kept out of merge statistics |
| `warm_start.file` | no | — | Existing `.tiktoken` file whose ranks are loaded as-is before training |
| `phases` | yes | — | List of training phases, run sequentially |
| `phases[].name` | yes | — | Phase name, for logging |
| `phases[].merges` | yes | — | Merge budget for this phase |
| `phases[].sources` | yes | — | List of `path:` (directory, scanned recursively for `.txt`/`.parquet`) or `file:` (single file) entries |

## Using the trained tokenizer

### With tiktoken (Python)

```python
import base64, tiktoken

mergeable_ranks = {}
with open("tokenizer.tiktoken") as f:
    for line in f:
        if line.strip():
            token_b64, rank = line.split()
            mergeable_ranks[base64.b64decode(token_b64)] = int(rank)

special_tokens = ["<|pad|>", "<|startoftext|>", "<|endoftext|>"]
enc = tiktoken.Encoding(
    name="custom_bpe",
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    mergeable_ranks=mergeable_ranks,
    special_tokens={t: len(mergeable_ranks) + i for i, t in enumerate(special_tokens)},
)

print(enc.encode("Hello, world!"))
```

The regex is the GPT-4 / `cl100k_base` pre-tokenization pattern; the trainer uses the same one, so boundaries match exactly.

### With Hugging Face

Convert to a `PreTrainedTokenizerFast` — the script verifies that both tokenizers produce identical IDs on a test suite before writing anything:

```bash
python scripts/tiktoken_to_hf.py tokenizer.tiktoken -s examples/special_tokens.txt -o hf_tokenizer
```

```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast.from_pretrained("./hf_tokenizer")
```

## Helper scripts

All in [`scripts/`](scripts/) — see [`scripts/README.md`](scripts/README.md) for details.

| Script | Purpose |
|--------|---------|
| `test_tokenizer.py` | Load a trained vocabulary with `tiktoken` and check encode/decode round-trips — run this first after any training run |
| `vocab_viewer.py` | Terminal UI for browsing a vocabulary: search tokens, inspect raw bytes |
| `tiktoken_to_hf.py` | Convert `.tiktoken` → Hugging Face tokenizer, with ID-level equivalence verification |
| `convert_hf_to_tiktoken.py` | Convert a Hugging Face `tokenizer.json` → `.tiktoken`, mainly to produce warm-start files |
| `convert_parquet_tree.py` | Corpus prep: mirror a directory tree of `.parquet`/`.jsonl.zst` files, renaming a chosen column to `text` |

## Sizing a run

The `working_set_mb` budget holds roughly `working_set_mb / 4` million characters of sampled training data (~4 bytes per character). Rough guidance:

| Working set | Recommended corpus size |
|-------------|------------------------|
| 1 GB | up to ~5 GB |
| 4 GB | up to ~20 GB |
| 16 GB | up to ~100 GB |
| 64 GB+ | 100 GB and beyond |

The corpus itself is streamed, never loaded whole — only the reservoir sample lives in memory, so corpus size is limited by training time, not RAM.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
