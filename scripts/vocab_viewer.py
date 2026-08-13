#!/usr/bin/env python3
"""
Tiktoken Vocabulary Viewer TUI

Interactive terminal interface for browsing tiktoken vocabulary files.

Usage:
    python vocab_viewer.py <tiktoken_file> [--special-tokens <file>]
"""

import argparse
import base64
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Label, Static


class VocabLoader:
    """Load and manage tiktoken vocabulary files."""

    def __init__(self, tiktoken_path: str, special_tokens_path: str | None = None):
        self.tiktoken_path = Path(tiktoken_path)
        self.special_tokens_path = Path(special_tokens_path) if special_tokens_path else None
        self.vocab: dict[int, bytes] = {}
        self.special_tokens: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        """Load the tiktoken file and optional special tokens."""
        # Load main vocabulary
        with open(self.tiktoken_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    token_bytes = base64.b64decode(parts[0])
                    rank = int(parts[1])
                    self.vocab[rank] = token_bytes

        # Load special tokens if provided
        if self.special_tokens_path and self.special_tokens_path.exists():
            base_rank = max(self.vocab.keys()) + 1 if self.vocab else 0
            with open(self.special_tokens_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    token = line.strip()
                    if token:
                        self.special_tokens[token] = base_rank + i
                        self.vocab[base_rank + i] = token.encode("utf-8")

    @property
    def size(self) -> int:
        """Return total vocabulary size."""
        return len(self.vocab)

    def get_sorted_tokens(self) -> list[tuple[int, bytes]]:
        """Return tokens sorted by rank."""
        return sorted(self.vocab.items(), key=lambda x: x[0])

    @staticmethod
    def format_hex(data: bytes) -> str:
        """Format bytes as hex string."""
        return " ".join(f"{b:02x}" for b in data)

    @staticmethod
    def format_decoded(data: bytes) -> str:
        """Format bytes as decoded string with escapes for non-printable."""
        result = []
        try:
            text = data.decode("utf-8")
            for char in text:
                code = ord(char)
                if char == "\n":
                    result.append("\\n")
                elif char == "\r":
                    result.append("\\r")
                elif char == "\t":
                    result.append("\\t")
                elif char == " ":
                    result.append("\u2423")  # visible space symbol
                elif code < 32 or code == 127:
                    result.append(f"<0x{code:02x}>")
                elif code >= 128 and code <= 159:
                    result.append(f"<U+{code:04X}>")
                else:
                    result.append(char)
            return "".join(result)
        except UnicodeDecodeError:
            # Fall back to hex representation for non-UTF8
            return "".join(f"<0x{b:02x}>" for b in data)


class JumpToIndexScreen(ModalScreen[int | None]):
    """Modal screen for jumping to a specific index."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="jump-dialog"):
            yield Label("Jump to index:", id="jump-label")
            yield Input(placeholder="Enter rank number...", id="jump-input")

    def on_mount(self) -> None:
        self.query_one("#jump-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            index = int(event.value)
            self.dismiss(index)
        except ValueError:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SearchScreen(ModalScreen[str | None]):
    """Modal screen for searching tokens."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="search-dialog"):
            yield Label("Search token text:", id="search-label")
            yield Input(placeholder="Enter search text...", id="search-input")

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value if event.value else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class VocabViewerApp(App):
    """Main TUI application for viewing tiktoken vocabularies."""

    CSS = """
    #jump-dialog, #search-dialog {
        align: center middle;
        width: 50;
        height: 7;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #jump-label, #search-label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    #jump-input, #search-input {
        width: 100%;
    }

    #info-bar {
        height: 1;
        background: $primary-background;
        color: $text;
        padding: 0 1;
    }

    DataTable {
        height: 1fr;
    }

    #main-container {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("g", "jump_to_index", "Go to index"),
        Binding("/", "search", "Search"),
        Binding("home", "go_home", "First", show=False),
        Binding("end", "go_end", "Last", show=False),
        Binding("pageup", "page_up", "Page Up", show=False),
        Binding("pagedown", "page_down", "Page Down", show=False),
    ]

    def __init__(self, vocab_loader: VocabLoader):
        super().__init__()
        self.vocab_loader = vocab_loader
        self.tokens = vocab_loader.get_sorted_tokens()
        self.rank_to_row: dict[int, int] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main-container"):
            yield Static(
                f"File: {self.vocab_loader.tiktoken_path} | Vocab size: {self.vocab_loader.size:,}",
                id="info-bar",
            )
            yield DataTable(id="vocab-table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#vocab-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True

        # Add columns
        table.add_column("Rank", width=8)
        table.add_column("Hex", width=32)
        table.add_column("Decoded", width=40)
        table.add_column("Len", width=5)

        # Populate table
        for row_idx, (rank, token_bytes) in enumerate(self.tokens):
            self.rank_to_row[rank] = row_idx
            table.add_row(
                str(rank),
                VocabLoader.format_hex(token_bytes),
                VocabLoader.format_decoded(token_bytes),
                str(len(token_bytes)),
            )

        # Update title
        self.title = f"Tiktoken Vocab Viewer - {self.vocab_loader.tiktoken_path.name}"

    def action_jump_to_index(self) -> None:
        """Show jump to index dialog."""

        def handle_result(index: int | None) -> None:
            if index is not None:
                table = self.query_one("#vocab-table", DataTable)
                if index in self.rank_to_row:
                    row = self.rank_to_row[index]
                    table.move_cursor(row=row)
                else:
                    # Find closest rank
                    ranks = sorted(self.rank_to_row.keys())
                    if index < ranks[0]:
                        table.move_cursor(row=0)
                    elif index > ranks[-1]:
                        table.move_cursor(row=len(ranks) - 1)
                    else:
                        # Find closest
                        closest = min(ranks, key=lambda x: abs(x - index))
                        table.move_cursor(row=self.rank_to_row[closest])

        self.push_screen(JumpToIndexScreen(), handle_result)

    def action_search(self) -> None:
        """Show search dialog."""

        def handle_result(search_text: str | None) -> None:
            if search_text:
                table = self.query_one("#vocab-table", DataTable)
                current_row = table.cursor_row
                search_lower = search_text.lower()

                # Search from current position forward, then wrap
                for offset in range(1, len(self.tokens) + 1):
                    idx = (current_row + offset) % len(self.tokens)
                    rank, token_bytes = self.tokens[idx]
                    decoded = VocabLoader.format_decoded(token_bytes).lower()
                    try:
                        utf8_text = token_bytes.decode("utf-8").lower()
                    except UnicodeDecodeError:
                        utf8_text = ""

                    if search_lower in decoded or search_lower in utf8_text:
                        table.move_cursor(row=idx)
                        break

        self.push_screen(SearchScreen(), handle_result)

    def action_go_home(self) -> None:
        """Jump to first token."""
        table = self.query_one("#vocab-table", DataTable)
        table.move_cursor(row=0)

    def action_go_end(self) -> None:
        """Jump to last token."""
        table = self.query_one("#vocab-table", DataTable)
        table.move_cursor(row=len(self.tokens) - 1)

    def action_page_up(self) -> None:
        """Move up one page."""
        table = self.query_one("#vocab-table", DataTable)
        current = table.cursor_row
        page_size = max(1, table.size.height - 2)
        table.move_cursor(row=max(0, current - page_size))

    def action_page_down(self) -> None:
        """Move down one page."""
        table = self.query_one("#vocab-table", DataTable)
        current = table.cursor_row
        page_size = max(1, table.size.height - 2)
        table.move_cursor(row=min(len(self.tokens) - 1, current + page_size))


def main():
    parser = argparse.ArgumentParser(
        description="Interactive TUI for browsing tiktoken vocabulary files"
    )
    parser.add_argument("tiktoken_file", help="Path to .tiktoken vocabulary file")
    parser.add_argument(
        "--special-tokens", "-s", help="Path to special tokens file (one per line)"
    )

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.tiktoken_file).exists():
        print(f"Error: File not found: {args.tiktoken_file}", file=sys.stderr)
        sys.exit(1)

    # Load vocabulary
    try:
        loader = VocabLoader(args.tiktoken_file, args.special_tokens)
    except Exception as e:
        print(f"Error loading vocabulary: {e}", file=sys.stderr)
        sys.exit(1)

    if loader.size == 0:
        print("Error: Vocabulary is empty", file=sys.stderr)
        sys.exit(1)

    # Run TUI
    app = VocabViewerApp(loader)
    app.run()


if __name__ == "__main__":
    main()
