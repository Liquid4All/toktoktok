#!/usr/bin/env python3
"""
Verification script for toktoktok BPE trainer output.
Tests that the generated .tiktoken file is compatible with OpenAI's tiktoken library.
"""

import base64
import sys
import os

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    sys.exit(1)


def load_custom_encoding(tiktoken_file: str, special_tokens_file: str | None = None):
    """Load a custom encoding from a tiktoken file."""

    # 1. Load mergeable ranks
    mergeable_ranks = {}
    with open(tiktoken_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line format: {line}")

            token_bytes = base64.b64decode(parts[0])
            rank = int(parts[1])
            mergeable_ranks[token_bytes] = rank

    # 2. Load special tokens
    special_tokens = {}
    if special_tokens_file and os.path.exists(special_tokens_file):
        with open(special_tokens_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:
                    # Assign IDs after the normal vocab
                    special_tokens[token] = len(mergeable_ranks) + i

    # 3. Construct encoding
    # Use the exact cl100k_base pattern from tiktoken
    # Note: tiktoken uses the 'regex' module which supports \p{L} etc.
    pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

    return tiktoken.Encoding(
        name="custom_bpe",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def test_roundtrip(encoding, text: str) -> bool:
    """Test that encoding and decoding produces the original text."""
    try:
        tokens = encoding.encode(text, allowed_special="all")
        decoded = encoding.decode(tokens)
        return decoded == text
    except Exception as e:
        print(f"  Error during roundtrip: {e}")
        return False


def test_encoding(tiktoken_path: str, special_tokens_path: str | None = None):
    """Run tests on a tiktoken file."""
    print(f"Testing: {tiktoken_path}")

    # Load encoding
    try:
        enc = load_custom_encoding(tiktoken_path, special_tokens_path)
        print(f"  Loaded encoding with {len(enc._mergeable_ranks)} tokens")
    except Exception as e:
        print(f"  FAIL: Could not load encoding: {e}")
        return False

    # Test cases
    test_cases = [
        "Hello world!",
        "This is a test of the custom tokenizer.",
        "The quick brown fox jumps over the lazy dog.",
        "Numbers: 123 456 789",
        "Contractions: I'm don't won't can't",
        "Unicode: café résumé naïve",
        "Whitespace:   multiple   spaces   ",
        "Newlines:\nLine 1\nLine 2\n",
        "Special chars: @#$%^&*()",
        "Mixed: Hello, World! 123 test@email.com",
        "Long text: " + "The " * 100 + "end.",
    ]

    passed = 0
    failed = 0

    for text in test_cases:
        if test_roundtrip(enc, text):
            passed += 1
        else:
            print(f"  FAIL: Roundtrip failed for: {text[:50]}...")
            failed += 1

    # Show stats for a sample
    sample = "Hello world! This is a test of the custom tokenizer."
    tokens = enc.encode(sample)
    print(f"\nSample encoding:")
    print(f"  Text: {sample}")
    print(f"  Token count: {len(tokens)}")
    print(f"  Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
    print(f"  Compression: {len(sample) / len(tokens):.2f} chars/token")

    # Summary
    print(f"\nResults: {passed}/{passed + failed} tests passed")

    if failed == 0:
        print("PASS: All roundtrip tests passed!")
        return True
    else:
        print(f"FAIL: {failed} tests failed")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_tokenizer.py <tiktoken_file> [special_tokens_file]")
        print("\nExample:")
        print("  python test_tokenizer.py model.tiktoken")
        print("  python test_tokenizer.py model.tiktoken special_tokens.txt")
        sys.exit(1)

    tiktoken_path = sys.argv[1]
    special_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(tiktoken_path):
        print(f"Error: File not found: {tiktoken_path}")
        sys.exit(1)

    success = test_encoding(tiktoken_path, special_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
