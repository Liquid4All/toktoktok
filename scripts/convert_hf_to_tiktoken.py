#!/usr/bin/env python3
"""
Convert a HuggingFace BPE tokenizer (tokenizer.json) to tiktoken format.

Usage:
    python convert_hf_to_tiktoken.py <input_tokenizer.json> <output.tiktoken>

Example:
    python scripts/convert_hf_to_tiktoken.py ./hf_tokenizer/tokenizer.json ./converted.tiktoken
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path


def bytes_to_unicode():
    """
    Returns GPT-2's byte-to-unicode mapping.

    GPT-2 maps bytes to unicode characters to ensure all tokens are printable.
    - Printable ASCII and some extended Latin chars stay as themselves
    - Other bytes are shifted to unicode range starting at U+0100
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def unicode_to_bytes():
    """Returns the inverse of GPT-2's byte-to-unicode mapping."""
    return {v: k for k, v in bytes_to_unicode().items()}


def decode_token(token_str: str, decoder: dict[str, int]) -> bytes:
    """
    Decode a GPT-2 style token string to raw bytes.

    Args:
        token_str: The unicode token string from HuggingFace vocab
        decoder: The unicode-to-byte mapping

    Returns:
        The raw bytes that the token represents
    """
    return bytes([decoder[c] for c in token_str])


def is_special_token(token: str) -> bool:
    """Check if a token is a special token (e.g., <|endoftext|>)."""
    return bool(re.match(r"^<\|.*\|>$", token))


def convert_hf_to_tiktoken(input_path: str, output_path: str) -> None:
    """
    Convert a HuggingFace tokenizer.json to tiktoken format.

    Args:
        input_path: Path to the HuggingFace tokenizer.json file
        output_path: Path to write the tiktoken output file
    """
    print(f"Loading HuggingFace tokenizer from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Extract vocabulary from the model section
    if "model" not in tokenizer_data:
        raise ValueError("tokenizer.json missing 'model' section")

    model = tokenizer_data["model"]
    if "vocab" not in model:
        raise ValueError("tokenizer.json model missing 'vocab' section")

    vocab = model["vocab"]
    print(f"Found {len(vocab)} tokens in vocabulary")

    # Get the unicode-to-bytes decoder
    decoder = unicode_to_bytes()

    # Separate special tokens from regular tokens
    regular_tokens = []
    special_tokens = []

    for token_str, hf_id in vocab.items():
        if is_special_token(token_str):
            special_tokens.append((token_str, hf_id))
        else:
            regular_tokens.append((token_str, hf_id))

    print(f"Found {len(special_tokens)} special tokens (will be skipped)")
    print(f"Found {len(regular_tokens)} regular tokens")

    # Sort regular tokens by their original HuggingFace ID to preserve merge order
    regular_tokens.sort(key=lambda x: x[1])

    # Convert and assign new contiguous ranks
    tiktoken_entries = []
    failed_tokens = []

    for new_rank, (token_str, hf_id) in enumerate(regular_tokens):
        try:
            token_bytes = decode_token(token_str, decoder)
            tiktoken_entries.append((token_bytes, new_rank))
        except KeyError as e:
            # Token contains characters not in GPT-2 byte encoding
            failed_tokens.append((token_str, hf_id, str(e)))

    if failed_tokens:
        print(f"Warning: {len(failed_tokens)} tokens could not be decoded:")
        for token_str, hf_id, err in failed_tokens[:5]:
            print(f"  ID {hf_id}: {repr(token_str)} - missing char {err}")
        if len(failed_tokens) > 5:
            print(f"  ... and {len(failed_tokens) - 5} more")

    # Write tiktoken format: base64(token_bytes) rank
    print(f"Writing {len(tiktoken_entries)} tokens to: {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        for token_bytes, rank in tiktoken_entries:
            b64 = base64.b64encode(token_bytes).decode("ascii")
            f.write(f"{b64} {rank}\n")

    print("Conversion complete!")
    print(f"  Input tokens: {len(vocab)}")
    print(f"  Special tokens skipped: {len(special_tokens)}")
    print(f"  Failed tokens: {len(failed_tokens)}")
    print(f"  Output tokens: {len(tiktoken_entries)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace BPE tokenizer to tiktoken format"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to HuggingFace tokenizer.json file"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output tiktoken file"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    convert_hf_to_tiktoken(args.input, args.output)


if __name__ == "__main__":
    main()
