# TokTokTok BPE Tokenizer Trainer Specification

TokTokTok is a high-performance Byte-Pair Encoding (BPE) tokenizer trainer that produces vocabularies compatible with OpenAI's tiktoken library.

## Overview

The trainer uses an iterative, multi-phase approach where each phase can train on different data sources with a specified merge budget. The output is a `.tiktoken` file that can be loaded directly by the tiktoken library.

### Key Design Principles

- **Memory-bounded operation**: Training operates within a configurable memory budget using reservoir sampling
- **Multi-threaded**: Parallel pair counting and merging across available CPU cores
- **Iterative processing**: Files are streamed and processed incrementally, not loaded entirely into memory
- **Phase-based training**: Multiple training phases allow controlled vocabulary allocation across different data domains

## Vocabulary Structure

The final vocabulary is composed of:

```
Total Vocab = 256 (base bytes) + 1,161 (hardcoded merges) + Σ(phase merges) + special tokens
```

| Component | Token ID Range | Description |
|-----------|---------------|-------------|
| Base bytes | 0-255 | Raw byte values (UTF-8 compatible) |
| Hardcoded merges | 256-1416 | Pre-defined merges for numbers, whitespace, operators |
| Trained merges | 1417+ | Corpus-learned merges from training phases |
| Special tokens | End of vocab | User-defined special tokens |

## Configuration Format (YAML)

Training is configured via a YAML file. Run with: `toktoktok -c config.yaml`

### YAML Schema

```yaml
# Output file path for the trained vocabulary
output: <path>                    # Required: .tiktoken output file

# Memory management
working_set_mb: <integer>         # Optional: Max memory in MB (default: 1024)

# Parallelization
threads: <integer>                # Optional: Thread count, -1 for auto (default: -1)

# Logging
verbose: <boolean>                # Optional: Detailed logging (default: false)

# Special tokens (added at end of vocabulary)
special_tokens:                   # Optional: List of special token strings
  - "<token1>"
  - "<token2>"

# Training phases (executed sequentially)
phases:                           # Required: At least one phase
  - name: <string>                # Required: Phase name for logging
    merges: <integer>             # Required: Number of merges for this phase
    sources:                      # Required: At least one source
      - path: <directory>         # Directory (recursive scan for .txt/.parquet)
      - file: <filepath>          # OR single file path
```

### Example Configuration

```yaml
output: ./my_tokenizer.tiktoken
working_set_mb: 4096
threads: -1
verbose: false

special_tokens:
  - "<|endoftext|>"
  - "<|pad|>"
  - "<|startoftext|>"
  - "<|im_start|>"
  - "<|im_end|>"

phases:
  - name: "English General"
    merges: 30000
    sources:
      - path: /data/corpus/english/wikipedia
      - path: /data/corpus/english/books
      - file: /data/corpus/english/common_crawl.parquet

  - name: "Programming"
    merges: 15000
    sources:
      - path: /data/corpus/code/python
      - path: /data/corpus/code/javascript

  - name: "Multilingual"
    merges: 5000
    sources:
      - path: /data/corpus/german
      - path: /data/corpus/french
```

## Supported Input Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| Plain text | `.txt` | UTF-8 encoded text files |
| Parquet | `.parquet` | Apache Parquet files with a `text` column |

When specifying a directory path, the trainer recursively scans for all `.txt` and `.parquet` files.

## Tokenization Regex

The trainer uses the GPT-4 / cl100k_base compatible regex pattern for pre-tokenization:

```regex
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

### Pattern Breakdown

| Pattern | Description | Examples |
|---------|-------------|----------|
| `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)` | English contractions (case-insensitive) | `'s`, `'T`, `'re`, `'VE` |
| `[^\r\n\p{L}\p{N}]?\p{L}+` | Optional non-letter/digit followed by letters | `Hello`, ` world`, `.test` |
| `\p{N}{1,3}` | 1-3 digit numbers | `1`, `42`, `999` |
| ` ?[^\s\p{L}\p{N}]+[\r\n]*` | Punctuation with optional leading space | ` !!!`, `...`, `->` |
| `\s*[\r\n]+` | Whitespace before newlines | `\n`, `  \n\n` |
| `\s+(?!\S)` | Trailing whitespace | Spaces at end of text |
| `\s+` | Other whitespace | Spaces between words |

This regex ensures tokenization boundaries match tiktoken's behavior for seamless compatibility.

## Hardcoded Merges

Before corpus-based training begins, 1,161 merges are applied for common patterns. These ensure efficient encoding of numbers, whitespace, and programming constructs regardless of training data.

### Hardcoded Token Categories

#### Two-Digit Numbers (100 merges, IDs 256-355)
All combinations `"00"` through `"99"`, each formed by merging two single-digit byte tokens.

| Token | ID | Composition |
|-------|-----|-------------|
| `"00"` | 256 | `'0'` + `'0'` |
| `"01"` | 257 | `'0'` + `'1'` |
| ... | ... | ... |
| `"99"` | 355 | `'9'` + `'9'` |

#### Three-Digit Numbers (1000 merges, IDs 356-1355)
All combinations `"000"` through `"999"`, each formed by merging a two-digit token with a single-digit byte.

| Token | ID | Composition |
|-------|-----|-------------|
| `"000"` | 356 | `"00"` (256) + `'0'` |
| `"001"` | 357 | `"00"` (256) + `'1'` |
| ... | ... | ... |
| `"999"` | 1355 | `"99"` (355) + `'9'` |

#### Multi-Space Tokens (31 merges, IDs 1356-1386)
Consecutive spaces from 2 to 32, built hierarchically:

| Token | ID | Composition |
|-------|-----|-------------|
| 2 spaces | 1356 | `' '` + `' '` |
| 3 spaces | 1357 | `' '` + 2 spaces |
| 4 spaces | 1358 | 2 spaces + 2 spaces |
| ... | ... | Binary-style building |
| 32 spaces | 1386 | 16 spaces + 16 spaces |

#### Multi-Newline Tokens (7 merges, IDs 1387-1393)
Consecutive newlines from 2 to 8, built hierarchically:

| Token | ID | Composition |
|-------|-----|-------------|
| 2 newlines | 1387 | `'\n'` + `'\n'` |
| 3 newlines | 1388 | `'\n'` + 2 newlines |
| ... | ... | Binary-style building |
| 8 newlines | 1393 | 4 newlines + 4 newlines |

#### Windows Line Ending (1 merge, ID 1394)
| Token | ID | Composition |
|-------|-----|-------------|
| `"\r\n"` | 1394 | `'\r'` + `'\n'` |

#### Programming Operators (20 merges, IDs 1395-1414)

| Token | ID | Token | ID |
|-------|-----|-------|-----|
| `==` | 1395 | `&&` | 1409 |
| `!=` | 1396 | `\|\|` | 1410 |
| `<=` | 1397 | `++` | 1411 |
| `>=` | 1398 | `--` | 1412 |
| `+=` | 1399 | `<<` | 1413 |
| `-=` | 1400 | `>>` | 1414 |
| `*=` | 1401 | | |
| `/=` | 1402 | | |
| `->` | 1403 | | |
| `=>` | 1404 | | |
| `::` | 1405 | | |
| `//` | 1406 | | |
| `/*` | 1407 | | |
| `*/` | 1408 | | |

#### Ellipsis (2 merges, IDs 1415-1416)
| Token | ID | Composition |
|-------|-----|-------------|
| `..` | 1415 | `'.'` + `'.'` |
| `...` | 1416 | `..` (1415) + `'.'` |

## Special Tokens

Special tokens are user-defined strings that:

1. Are identified in raw text before regex splitting
2. Are excluded from BPE merge statistics (their internal bytes are never merged)
3. Are assigned sequential IDs at the end of the vocabulary
4. Must be encoded/decoded as atomic units

Common special tokens include:
- `<|endoftext|>` - End of document marker
- `<|pad|>` - Padding token
- `<|im_start|>` / `<|im_end|>` - Chat message delimiters

## Memory Management

### Working Set Constraint

The `working_set_mb` parameter defines the maximum memory the trainer should use for the training corpus. The implementation uses **reservoir sampling** to maintain a representative sample when the corpus exceeds available memory.

**Important**: The working set is not allocated as a single contiguous block. Instead, data is stored in multiple smaller chunks to:
- Avoid large object heap fragmentation
- Enable efficient parallel processing
- Allow incremental garbage collection

### Reservoir Sampling Strategy

1. Files are streamed line-by-line (not loaded entirely)
2. Text is pre-processed through the regex and converted to initial byte tokens
3. Token sequences are stored in fixed-size chunks
4. When memory budget is reached, new data probabilistically replaces existing samples
5. This ensures the training data represents the distribution of the entire corpus

### Memory Estimation

Approximate memory cost per character: ~4 bytes (accounting for token IDs and linked-list overhead for merge tracking).

For a 4GB working set:
- ~1 billion characters of training data can be held in memory
- Larger corpora are sampled proportionally

## Multi-Threading Architecture

BPE training has inherent sequential dependencies (each merge affects global statistics), but is parallelized using a Map-Reduce pattern:

### Phase 1: Parallel Pair Counting (Map)

1. Training data is partitioned into T chunks (one per thread)
2. Each thread independently counts adjacent token pairs in its chunk
3. Thread-local hash maps accumulate pair frequencies
4. No synchronization during counting

### Phase 2: Aggregation and Selection (Reduce)

1. Main thread aggregates counts from all thread-local maps
2. Global maximum pair is identified
3. New token ID is assigned to the winning pair

### Phase 3: Parallel Merge Application

1. Winning pair is broadcast to all threads
2. Each thread scans its chunk and:
   - Replaces matching pairs with the new token
   - Updates linked-list pointers (O(1) per merge)
   - Decrements counts for broken pairs
   - Increments counts for newly created pairs
3. Process repeats until merge budget exhausted

### Data Structure for Efficient Merging

Tokens are stored in an index-based linked list to enable O(1) merge operations:

```
Node = { TokenId, NextIndex, PrevIndex }
```

Merging two adjacent tokens requires only pointer updates, avoiding expensive array shifts.

## Output Format

The trainer produces a `.tiktoken` file with one token per line:

```
<base64_encoded_bytes> <rank>
```

Example:
```
IQ== 0
Ig== 1
Iw== 2
...
SGVsbG8= 1500
```

Where:
- `IQ==` is the base64-encoded byte sequence for the token
- `0` is the rank (token ID)

## Loading with tiktoken (Python)

```python
import base64
import tiktoken

def load_custom_encoding(tiktoken_file: str, special_tokens: list[str] = None):
    """
    Load a custom .tiktoken vocabulary file.

    Args:
        tiktoken_file: Path to the .tiktoken file
        special_tokens: Optional list of special token strings

    Returns:
        tiktoken.Encoding object
    """
    # Load mergeable ranks from file
    mergeable_ranks = {}
    with open(tiktoken_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            token_bytes = base64.b64decode(parts[0])
            rank = int(parts[1])
            mergeable_ranks[token_bytes] = rank

    # Build special tokens dict with IDs after normal vocab
    special_tokens_dict = {}
    if special_tokens:
        base_id = len(mergeable_ranks)
        for i, token in enumerate(special_tokens):
            special_tokens_dict[token] = base_id + i

    # Create encoding with GPT-4 compatible regex
    return tiktoken.Encoding(
        name="custom_bpe",
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_dict
    )


def test_tokenizer(tiktoken_file: str, special_tokens: list[str] = None):
    """Test the tokenizer with sample text."""
    enc = load_custom_encoding(tiktoken_file, special_tokens)

    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def hello():\n    print('Hello, World!')\n",
        "Numbers: 42, 123, 9999",
        "    Four spaces of indentation",
    ]

    print(f"Vocabulary size: {enc.n_vocab}")
    print()

    for text in test_texts:
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        roundtrip_ok = decoded == text

        print(f"Text: {repr(text)}")
        print(f"  Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"  Roundtrip: {'PASS' if roundtrip_ok else 'FAIL'}")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_tokenizer.py <tiktoken_file> [special_tokens...]")
        sys.exit(1)

    tiktoken_file = sys.argv[1]
    special_tokens = sys.argv[2:] if len(sys.argv) > 2 else None

    test_tokenizer(tiktoken_file, special_tokens)
```

### Example Usage

```bash
# Basic usage
python load_tokenizer.py my_tokenizer.tiktoken

# With special tokens
python load_tokenizer.py my_tokenizer.tiktoken "<|endoftext|>" "<|pad|>"
```

### Encoding and Decoding

```python
# Load the tokenizer
enc = load_custom_encoding("my_tokenizer.tiktoken", ["<|endoftext|>"])

# Encode text to token IDs
tokens = enc.encode("Hello, world!")
print(tokens)  # [15496, 11, 995, 0]

# Decode tokens back to text
text = enc.decode(tokens)
print(text)  # "Hello, world!"

# Encode with special tokens
tokens = enc.encode("Hello<|endoftext|>", allowed_special={"<|endoftext|>"})
```

## Performance Considerations

### Memory vs. Corpus Size Trade-offs

| Working Set | Recommended Corpus Size | Notes |
|-------------|------------------------|-------|
| 1 GB | Up to ~5 GB | Small vocabularies, quick iteration |
| 4 GB | Up to ~20 GB | Medium-scale training |
| 16 GB | Up to ~100 GB | Large vocabularies |
| 64 GB+ | 100+ GB | Production-scale training |

### Thread Scaling

- Pair counting scales linearly with thread count
- Merge application has diminishing returns above 16-32 threads due to synchronization overhead
- Recommended: Set `threads: -1` to use all available cores

### Training Time Estimates

Training time depends on:
1. Corpus size (bytes processed)
2. Target merge count
3. Working set size (affects sampling overhead)
4. Thread count

A rough estimate: ~1-5 merges per second on modern hardware, so 50,000 merges takes approximately 3-14 hours depending on configuration.
