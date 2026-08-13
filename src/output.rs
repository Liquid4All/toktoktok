use base64::{engine::general_purpose::STANDARD, Engine};
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::hardcoded::HardcodedMerge;
use crate::trainer::{reconstruct_token_bytes, TrainedMerge};

/// Write the vocabulary to a tiktoken-compatible file
/// Format: base64_encoded_bytes <space> rank
pub fn write_tiktoken_file(
    output_path: &str,
    hardcoded_merges: &[HardcodedMerge],
    trained_merges: &[TrainedMerge],
    _special_tokens: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    let mut rank: u32 = 0;

    // Write base bytes (0-255)
    for byte in 0u8..=255 {
        let b64 = STANDARD.encode([byte]);
        writeln!(writer, "{} {}", b64, rank)?;
        rank += 1;
    }

    // Write hardcoded merges
    for merge in hardcoded_merges {
        let b64 = STANDARD.encode(&merge.bytes);
        writeln!(writer, "{} {}", b64, rank)?;
        rank += 1;
    }

    // Write trained merges
    for merge in trained_merges.iter() {
        // Reconstruct the byte sequence for this merge
        let bytes = reconstruct_token_bytes(merge.new_token, hardcoded_merges, trained_merges);
        let b64 = STANDARD.encode(&bytes);
        writeln!(writer, "{} {}", b64, rank)?;
        rank += 1;
    }

    // Note: Special tokens are NOT written to the .tiktoken file
    // They are handled separately when loading with tiktoken
    // The user must pass the same special tokens to the load function

    writer.flush()?;

    Ok(())
}

/// Create a Python test script for validating the tokenizer
#[allow(dead_code)]
pub fn write_test_script(output_path: &str, tiktoken_path: &str, special_tokens: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, r#"#!/usr/bin/env python3
"""
Test script for validating the trained tokenizer.
Usage: python test_tokenizer.py
"""

import base64
import tiktoken

TIKTOKEN_FILE = "{tiktoken_path}"
SPECIAL_TOKENS = {special_tokens:?}

def load_custom_encoding(tiktoken_file: str, special_tokens: list[str] = None):
    """Load a custom .tiktoken vocabulary file."""
    mergeable_ranks = {{}}
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

    special_tokens_dict = {{}}
    if special_tokens:
        base_id = len(mergeable_ranks)
        for i, token in enumerate(special_tokens):
            special_tokens_dict[token] = base_id + i

    return tiktoken.Encoding(
        name="custom_bpe",
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{{L}}\p{{N}}]?\p{{L}}+|\p{{N}}{{1,3}}| ?[^\s\p{{L}}\p{{N}}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_dict
    )

def test_tokenizer():
    print("Loading tokenizer...")
    enc = load_custom_encoding(TIKTOKEN_FILE, SPECIAL_TOKENS if SPECIAL_TOKENS else None)
    print(f"Vocabulary size: {{enc.n_vocab}}")
    print()

    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def hello():\n    print('Hello, World!')\n",
        "Numbers: 42, 123, 9999",
        "    Four spaces of indentation",
        "Testing contractions: I'm, don't, won't, they're",
        "Special chars: @#$%^&*()",
        "Multiple    spaces   and\n\nnewlines",
    ]

    print("Running tests...")
    print("-" * 60)

    all_passed = True
    for text in test_texts:
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        passed = decoded == text

        status = "PASS" if passed else "FAIL"
        print(f"[{{status}}] {{repr(text)[:50]}}")
        print(f"       Tokens ({{len(tokens)}}): {{tokens[:15]}}{{'' if len(tokens) <= 15 else '...'}}")

        if not passed:
            print(f"       Expected: {{repr(text)}}")
            print(f"       Got:      {{repr(decoded)}}")
            all_passed = False
        print()

    print("-" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return all_passed

if __name__ == "__main__":
    import sys
    success = test_tokenizer()
    sys.exit(0 if success else 1)
"#)?;

    writer.flush()?;
    Ok(())
}
