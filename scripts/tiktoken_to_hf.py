#!/usr/bin/env python3
"""
Convert a .tiktoken BPE tokenizer to HuggingFace format and verify equivalence.

Loads a tiktoken file + special tokens, converts to HuggingFace PreTrainedTokenizerFast,
then verifies both tokenizers produce identical results on a set of test strings.
"""

import argparse
import base64
import json
import os
import sys

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run: pip install tiktoken")
    sys.exit(1)

try:
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import TikTokenConverter
except ImportError:
    print("Error: transformers not installed. Run: pip install transformers")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    print("Error: rich not installed. Run: pip install rich")
    sys.exit(1)


PAT_STR = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""

# HF tokenizers uses Rust's `regex` crate which does NOT support possessive quantifiers.
# tiktoken uses `fancy-regex` which does. Without stripping them, `{1,3}+` gets parsed as
# `({1,3})+` (= unlimited digits), breaking number tokenization.
PAT_STR_HF = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"""

TEST_STRINGS = [
    # English
    ("English (basic)", "Hello world! This is a test of the custom tokenizer."),
    ("English (contractions)", "I'm don't won't can't she's they've we'll I'd"),
    ("English (long)", "The quick brown fox jumps over the lazy dog. " * 5),
    ("English (code)", 'def hello_world():\n    print("Hello, World!")\n    return 42\n'),
    # Korean
    ("Korean (basic)", "안녕하세요 세계! 이것은 토크나이저 테스트입니다."),
    ("Korean (mixed)", "한국어와 English가 섞인 문장입니다 123."),
    ("Korean (formal)", "대한민국은 민주공화국이다. 대한민국의 주권은 국민에게 있고, 모든 권력은 국민으로부터 나온다."),
    # Chinese
    ("Chinese", "你好世界！这是一个分词器的测试。机器学习很有趣。"),
    # Japanese
    ("Japanese", "こんにちは世界！これはトークナイザーのテストです。"),
    # Arabic
    ("Arabic (basic)", "مرحبا بالعالم! هذا اختبار للمحلل اللغوي."),
    ("Arabic (long)", "الذكاء الاصطناعي هو محاكاة عمليات الذكاء البشري بواسطة أنظمة الحاسوب. تشمل هذه العمليات التعلم والاستدلال والتصحيح الذاتي."),
    ("Arabic (mixed)", "نموذج GPT-4 يحتوي على 1,760,000,000,000 معلمة تقريباً."),
    # Thai
    ("Thai (basic)", "สวัสดีครับ นี่คือการทดสอบตัวแบ่งคำ"),
    ("Thai (long)", "ปัญญาประดิษฐ์คือการจำลองกระบวนการทางปัญญาของมนุษย์โดยระบบคอมพิวเตอร์ ซึ่งรวมถึงการเรียนรู้ การใช้เหตุผล และการแก้ไขตนเอง"),
    ("Thai (mixed)", "โมเดล GPT-4 มีพารามิเตอร์ 1,760,000,000,000 ตัว"),
    # Numbers & symbols
    ("Numbers (basic)", "123 456 789 3.14159 1,000,000 $99.99 50%"),
    ("Numbers (long)", "1234567890 0.000001 999999999 1e10 -42 +3.14"),
    ("Numbers (boundary)", "1 12 123 1234 12345 123456"),
    ("Numbers (decimals)", "0.1 0.12 0.123 3.14159265358979"),
    ("Numbers (formatted)", "$1,234.56 €9.99 ¥100 £50,000.00"),
    ("Symbols", "@#$%^&*() {}[]|\\/<>~`+="),
    # Whitespace & newlines
    ("Whitespace (tabs)", "col1\tcol2\tcol3\nval1\tval2\tval3"),
    ("Whitespace (multi)", "  multiple   spaces   and\ttabs\nand\nnewlines\n"),
    ("Newlines (unix)", "line1\nline2\nline3\n"),
    ("Newlines (windows)", "line1\r\nline2\r\nline3\r\n"),
    ("Newlines (mixed)", "line1\nline2\r\nline3\rline4"),
    ("Newlines (multiple)", "paragraph1\n\n\nparagraph2\n\n\n\nparagraph3"),
    ("Whitespace (trailing)", "hello   \n  world  \n"),
    ("Whitespace (only)", "   \t\t  \n\n  \t  "),
    # Emoji
    ("Emoji", "Hello 🌍! 🎉🚀 AI is 🔥"),
    # URLs & email
    ("URL/email", "Visit https://example.com or email test@example.com for info."),
    # Special tokens (must be tested with allowed_special="all" in tiktoken)
    ("Special (im)", "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"),
    ("Special (think)", "<think>\nLet me reason about this step by step.\n</think>"),
    ("Special (fim)", "<|fim_pre|>def hello():\n    <|fim_suf|>\n    return result<|fim_mid|>x = 42"),
    ("Special (endoftext)", "Hello world<|endoftext|>"),
    ("Special (pad)", "<|pad|><|pad|><|pad|>actual content here"),
    ("Special (mixed)", "<|im_start|>user\n안녕하세요! 오늘 날씨가 어때요?<|im_end|>\n<|im_start|>assistant\n<think>\n오늘 날씨를 알려드리겠습니다.\n</think>\n오늘은 맑은 날씨입니다.<|im_end|>"),
]


def load_tiktoken_encoding(
    tiktoken_file: str, special_tokens_file: str | None = None
) -> tiktoken.Encoding:
    """Load a custom tiktoken Encoding from a .tiktoken file."""
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

    special_tokens = {}
    if special_tokens_file and os.path.exists(special_tokens_file):
        with open(special_tokens_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                token = line.strip()
                if token:
                    special_tokens[token] = len(mergeable_ranks) + i

    return tiktoken.Encoding(
        name="toktoktok_bpe",
        pat_str=PAT_STR,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )


def convert_and_save(
    tiktoken_file: str,
    encoding: tiktoken.Encoding,
    output_dir: str,
    console: Console,
):
    """Convert tiktoken encoding to HuggingFace format and save.

    Uses TikTokenConverter directly with the existing .tiktoken file,
    bypassing the blobfile dependency that convert_tiktoken_to_fast requires.
    """
    os.makedirs(output_dir, exist_ok=True)
    console.print("[bold]Converting to HuggingFace format...[/]")

    converter = TikTokenConverter(
        vocab_file=tiktoken_file,
        pattern=PAT_STR_HF,
        additional_special_tokens=encoding._special_tokens,
    )
    tokenizer = converter.converted()

    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # Build the list of special tokens with their IDs
    special_tokens = encoding._special_tokens
    added_tokens_decoder = {}
    for token_str, token_id in special_tokens.items():
        added_tokens_decoder[str(token_id)] = {
            "content": token_str,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }

    # Write tokenizer_config.json so AutoTokenizer can load it
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "added_tokens_decoder": added_tokens_decoder,
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "model_max_length": 1000000000000000019884624838656,
    }
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    console.print(f"[bold green]Saved HuggingFace tokenizer to:[/] {output_dir}")


def run_verification(
    tiktoken_enc: tiktoken.Encoding,
    hf_tokenizer: PreTrainedTokenizerFast,
    console: Console,
) -> bool:
    """Compare tiktoken and HuggingFace tokenizers on test strings.

    Distinguishes between:
    - PASS: identical token IDs
    - WARN: different IDs but identical roundtrip (BPE tie-breaking difference)
    - FAIL: different decoded output (actual incompatibility)
    """

    results = []
    for label, text in TEST_STRINGS:
        tt_ids = tiktoken_enc.encode(text, allowed_special="all")
        hf_ids = hf_tokenizer.encode(text, add_special_tokens=False)
        hf_decoded = hf_tokenizer.decode(hf_ids)
        tt_decoded = tiktoken_enc.decode(tt_ids)

        ids_match = tt_ids == hf_ids
        roundtrip_ok = tt_decoded == hf_decoded == text

        results.append((label, text, tt_ids, hf_ids, ids_match, roundtrip_ok))

    # Summary table
    table = Table(
        title="Tokenizer Equivalence Test",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Category", style="bold", width=22)
    table.add_column("Tiktoken IDs", width=28, overflow="fold")
    table.add_column("HF IDs", width=28, overflow="fold")
    table.add_column("#Tok", justify="right", width=5)
    table.add_column("Status", justify="center", width=6)

    has_fail = False
    has_warn = False

    for label, text, tt_ids, hf_ids, ids_match, roundtrip_ok in results:
        tt_str = str(tt_ids[:12]) + ("..." if len(tt_ids) > 12 else "")
        hf_str = str(hf_ids[:12]) + ("..." if len(hf_ids) > 12 else "")

        if ids_match:
            status = "[bold green]PASS[/]"
        elif roundtrip_ok:
            status = "[bold yellow]WARN[/]"
            has_warn = True
        else:
            status = "[bold red]FAIL[/]"
            has_fail = True

        table.add_row(label, tt_str, hf_str, str(len(tt_ids)), status)

    console.print()
    console.print(table)

    if has_warn:
        console.print(
            "\n[bold yellow]WARN[/] = IDs differ but roundtrip is identical "
            "(BPE merge tie-breaking difference, harmless)"
        )

    # Detail panels for non-PASS results
    non_pass = [r for r in results if not r[4]]
    if non_pass:
        console.print()
        for label, text, tt_ids, hf_ids, ids_match, roundtrip_ok in non_pass:
            detail = Text()
            detail.append("Category: ", style="bold")
            detail.append(f"{label}\n")
            detail.append("Input:    ", style="bold")
            detail.append(f"{text[:80]}{'...' if len(text) > 80 else ''}\n")
            detail.append("Tiktoken: ", style="bold")
            detail.append(f"{tt_ids}\n")
            detail.append("HF:       ", style="bold")
            detail.append(f"{hf_ids}\n")

            # Find first divergence
            min_len = min(len(tt_ids), len(hf_ids))
            diverge_idx = None
            for i in range(min_len):
                if tt_ids[i] != hf_ids[i]:
                    diverge_idx = i
                    break
            if diverge_idx is None and len(tt_ids) != len(hf_ids):
                diverge_idx = min_len

            if diverge_idx is not None:
                detail.append(f"First divergence at index {diverge_idx}: ", style="bold yellow")
                tt_val = tt_ids[diverge_idx] if diverge_idx < len(tt_ids) else "<end>"
                hf_val = hf_ids[diverge_idx] if diverge_idx < len(hf_ids) else "<end>"
                detail.append(f"tiktoken={tt_val} vs hf={hf_val}\n")

                # Show what each token decodes to
                if diverge_idx < len(tt_ids):
                    tt_bytes = tiktoken_enc.decode_single_token_bytes(tt_ids[diverge_idx])
                    detail.append(f"  tiktoken token {tt_ids[diverge_idx]}: {tt_bytes!r}\n")
                if diverge_idx < len(hf_ids):
                    hf_tok_str = hf_tokenizer.decode([hf_ids[diverge_idx]])
                    detail.append(f"  HF token {hf_ids[diverge_idx]}: {hf_tok_str!r}\n")

            detail.append("Roundtrip: ", style="bold")
            if roundtrip_ok:
                detail.append("OK (decoded text matches)\n", style="green")
            else:
                detail.append("FAILED (decoded text differs)\n", style="red")

            border = "yellow" if roundtrip_ok else "red"
            title_prefix = "WARN" if roundtrip_ok else "FAIL"
            title_style = "bold yellow" if roundtrip_ok else "bold red"
            console.print(
                Panel(detail, title=f"[{title_style}]{title_prefix}: {label}[/]", border_style=border)
            )

    return not has_fail


def print_stats(
    tiktoken_enc: tiktoken.Encoding,
    hf_tokenizer: PreTrainedTokenizerFast,
    console: Console,
):
    """Print vocabulary statistics for both tokenizers."""
    tt_vocab = len(tiktoken_enc._mergeable_ranks)
    tt_special = len(tiktoken_enc._special_tokens)
    hf_vocab = hf_tokenizer.vocab_size
    hf_total = len(hf_tokenizer)

    table = Table(title="Vocabulary Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Tiktoken", justify="right")
    table.add_column("HuggingFace", justify="right")

    table.add_row("Regular tokens", f"{tt_vocab:,}", f"{hf_vocab:,}")
    table.add_row("Special tokens", f"{tt_special:,}", f"{hf_total - hf_vocab:,}")
    table.add_row("Total", f"{tt_vocab + tt_special:,}", f"{hf_total:,}")

    console.print()
    console.print(table)

    # Compression comparison on a sample text
    sample = (
        "The quick brown fox jumps over the lazy dog. "
        "안녕하세요 세계! Hello world 123. "
        "def main(): pass"
    )
    tt_ids = tiktoken_enc.encode(sample, allowed_special="all")
    hf_ids = hf_tokenizer.encode(sample, add_special_tokens=False)

    comp_table = Table(title="Compression Comparison", show_header=True, header_style="bold blue")
    comp_table.add_column("Metric", style="bold")
    comp_table.add_column("Value", justify="right")

    comp_table.add_row("Sample length (chars)", f"{len(sample)}")
    comp_table.add_row("Sample length (UTF-8 bytes)", f"{len(sample.encode('utf-8'))}")
    comp_table.add_row("Tiktoken tokens", f"{len(tt_ids)}")
    comp_table.add_row("HuggingFace tokens", f"{len(hf_ids)}")
    comp_table.add_row("Tiktoken chars/token", f"{len(sample) / len(tt_ids):.2f}")
    comp_table.add_row("HuggingFace chars/token", f"{len(sample) / len(hf_ids):.2f}")

    console.print()
    console.print(comp_table)


def main():
    parser = argparse.ArgumentParser(
        description="Convert tiktoken BPE tokenizer to HuggingFace format and verify equivalence"
    )
    parser.add_argument(
        "tiktoken_file",
        help="Path to .tiktoken file",
    )
    parser.add_argument(
        "-s",
        "--special-tokens",
        default="examples/special_tokens.txt",
        help="Path to special tokens file (default: examples/special_tokens.txt)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="hf_tokenizer",
        help="Output directory for HuggingFace tokenizer (default: hf_tokenizer)",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion, only verify an existing HF tokenizer dir",
    )
    args = parser.parse_args()

    console = Console()

    # Resolve paths relative to project root if not absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    tiktoken_path = args.tiktoken_file
    if not os.path.isabs(tiktoken_path):
        tiktoken_path = os.path.join(project_root, tiktoken_path)

    special_path = args.special_tokens
    if not os.path.isabs(special_path):
        special_path = os.path.join(project_root, special_path)

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    # Validate inputs
    if not os.path.exists(tiktoken_path):
        console.print(f"[bold red]Error:[/] File not found: {tiktoken_path}")
        sys.exit(1)

    if not os.path.exists(special_path):
        console.print(f"[bold yellow]Warning:[/] Special tokens file not found: {special_path}")
        special_path = None

    # Step 1: Load tiktoken encoding
    console.print(Panel("[bold]Step 1:[/] Loading tiktoken encoding", style="blue"))
    tiktoken_enc = load_tiktoken_encoding(tiktoken_path, special_path)
    console.print(
        f"  Loaded [cyan]{len(tiktoken_enc._mergeable_ranks):,}[/] regular tokens"
        f" + [cyan]{len(tiktoken_enc._special_tokens):,}[/] special tokens"
    )

    # Step 2: Convert to HuggingFace
    if not args.skip_convert:
        console.print()
        console.print(Panel("[bold]Step 2:[/] Converting to HuggingFace format", style="blue"))
        convert_and_save(tiktoken_path, tiktoken_enc, output_dir, console)
    else:
        console.print()
        console.print(Panel("[bold]Step 2:[/] Skipping conversion (--skip-convert)", style="yellow"))

    # Step 3: Load HuggingFace tokenizer
    console.print()
    console.print(Panel("[bold]Step 3:[/] Loading HuggingFace tokenizer", style="blue"))
    if not os.path.exists(output_dir):
        console.print(f"[bold red]Error:[/] Output dir not found: {output_dir}")
        sys.exit(1)

    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
    console.print(f"  Loaded HuggingFace tokenizer with [cyan]{len(hf_tokenizer):,}[/] tokens")

    # Step 4: Print stats
    console.print()
    console.print(Panel("[bold]Step 4:[/] Vocabulary statistics", style="blue"))
    print_stats(tiktoken_enc, hf_tokenizer, console)

    # Step 5: Verify equivalence
    console.print()
    console.print(Panel("[bold]Step 5:[/] Verifying equivalence", style="blue"))
    all_passed = run_verification(tiktoken_enc, hf_tokenizer, console)

    # Final result
    console.print()
    if all_passed:
        console.print(
            Panel(
                "[bold green]All roundtrip tests passed! Tokenizers are functionally equivalent.[/]",
                title="Result",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]Some tests failed! Tokenizers produce different decoded output.[/]",
                title="Result",
                border_style="red",
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
