#!/usr/bin/env python3
"""
Convert parquet and jsonl.zst files in a directory tree, transforming one column to another.
All outputs are in parquet format.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import pyarrow.parquet as pq
import pyarrow as pa
import zstandard as zstd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

console = Console()


def find_input_files(root_path: Path) -> list[Path]:
    """Recursively find all parquet and jsonl.zst files in the directory tree."""
    input_files = []

    # Find parquet files
    for path in root_path.rglob("*.parquet"):
        if path.is_file():
            input_files.append(path)

    # Find jsonl.zst files
    for path in root_path.rglob("*.jsonl.zst"):
        if path.is_file():
            input_files.append(path)

    return sorted(input_files)


def convert_jsonl_zst_file(
    input_file: Path,
    output_file: Path,
    input_column: str,
    output_column: str,
    transform_fn: Optional[Callable] = None
) -> tuple[bool, Optional[str], int]:
    """
    Convert a single jsonl.zst file to parquet.

    Returns:
        (success, error_message, num_rows)
    """
    try:
        values = []

        # Read and decompress the jsonl.zst file
        with open(input_file, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = reader.read().decode('utf-8')
                for line in text_stream.strip().split('\n'):
                    if line:
                        try:
                            record = json.loads(line)
                            if input_column in record:
                                values.append(record[input_column])
                            else:
                                return False, f"Column '{input_column}' not found in JSON record", 0
                        except json.JSONDecodeError as e:
                            return False, f"JSON decode error: {e}", 0

        if not values:
            return False, "No records found in file", 0

        # Create PyArrow array from values
        column_data = pa.array(values)

        # Apply transformation if provided
        if transform_fn is not None:
            column_data = transform_fn(column_data)

        # Create new table with the column
        new_table = pa.table({output_column: column_data})

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the output parquet file
        pq.write_table(new_table, output_file)

        return True, None, len(new_table)

    except Exception as e:
        return False, str(e), 0


def convert_parquet_file(
    input_file: Path,
    output_file: Path,
    input_column: str,
    output_column: str,
    transform_fn: Optional[Callable] = None
) -> tuple[bool, Optional[str], int]:
    """
    Convert a single parquet file.

    Returns:
        (success, error_message, num_rows)
    """
    try:
        # Read the parquet file
        table = pq.read_table(input_file, columns=[input_column])

        # Get the column data
        column_data = table.column(input_column)

        # Apply transformation if provided
        if transform_fn is not None:
            column_data = transform_fn(column_data)

        # Create new table with renamed column
        new_table = pa.table({output_column: column_data})

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the output parquet file
        pq.write_table(new_table, output_file)

        return True, None, len(new_table)

    except KeyError:
        return False, f"Column '{input_column}' not found in file", 0
    except Exception as e:
        return False, str(e), 0


def convert_file(
    input_file: Path,
    output_file: Path,
    input_column: str,
    output_column: str,
    transform_fn: Optional[Callable] = None
) -> tuple[bool, Optional[str], int]:
    """
    Convert a file (parquet or jsonl.zst) to parquet format.

    Returns:
        (success, error_message, num_rows)
    """
    if input_file.suffix == '.zst' and input_file.stem.endswith('.jsonl'):
        return convert_jsonl_zst_file(input_file, output_file, input_column, output_column, transform_fn)
    elif input_file.suffix == '.parquet':
        return convert_parquet_file(input_file, output_file, input_column, output_column, transform_fn)
    else:
        return False, f"Unsupported file type: {input_file.suffix}", 0


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet and jsonl.zst files in a directory tree, transforming one column to another. All outputs are in parquet format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i ./input_data -o ./output_data --input_column text --output_column content
  %(prog)s -i /data/raw -o /data/processed --input_column raw_text --output_column processed_text
        """
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input directory containing parquet and jsonl.zst files"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for converted parquet files"
    )

    parser.add_argument(
        "--input_column",
        type=str,
        required=True,
        help="Name of the input column to extract"
    )

    parser.add_argument(
        "--output_column",
        type=str,
        required=True,
        help="Name of the output column in the new parquet files"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input.exists():
        console.print(f"[red]Error: Input directory does not exist: {args.input}[/red]")
        sys.exit(1)

    if not args.input.is_dir():
        console.print(f"[red]Error: Input path is not a directory: {args.input}[/red]")
        sys.exit(1)

    # Show configuration
    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="yellow")
    config_table.add_row("Input Directory", str(args.input.absolute()))
    config_table.add_row("Output Directory", str(args.output.absolute()))
    config_table.add_row("Input Column", args.input_column)
    config_table.add_row("Output Column", args.output_column)
    console.print(config_table)
    console.print()

    # Find all input files
    console.print("[cyan]Scanning for parquet and jsonl.zst files...[/cyan]")
    input_files = find_input_files(args.input)

    if not input_files:
        console.print(f"[yellow]No parquet or jsonl.zst files found in {args.input}[/yellow]")
        sys.exit(0)

    # Count file types
    parquet_count = sum(1 for f in input_files if f.suffix == '.parquet')
    jsonl_count = sum(1 for f in input_files if f.suffix == '.zst')

    console.print(f"[green]Found {len(input_files)} file(s): {parquet_count} parquet, {jsonl_count} jsonl.zst[/green]")
    console.print()

    # Process files with progress bar
    total_rows = 0
    successful = 0
    failed = 0
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Converting files...", total=len(input_files))

        for input_file in input_files:
            # Calculate relative path and output path
            rel_path = input_file.relative_to(args.input)

            # Convert .jsonl.zst to .parquet in output path
            if input_file.suffix == '.zst' and input_file.stem.endswith('.jsonl'):
                # Remove .jsonl.zst and add .parquet
                output_rel_path = rel_path.parent / (rel_path.stem.replace('.jsonl', '') + '.parquet')
            else:
                output_rel_path = rel_path

            output_file = args.output / output_rel_path

            # Update progress description
            progress.update(task, description=f"[cyan]Converting: {rel_path}")

            # Convert the file
            success, error_msg, num_rows = convert_file(
                input_file,
                output_file,
                args.input_column,
                args.output_column
            )

            if success:
                successful += 1
                total_rows += num_rows
            else:
                failed += 1
                errors.append((rel_path, error_msg))

            progress.advance(task)

    # Show summary
    console.print()
    summary = Table(title="Conversion Summary", show_header=False)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="yellow")
    summary.add_row("Total Files", str(len(input_files)))
    summary.add_row("Successful", f"[green]{successful}[/green]")
    summary.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    summary.add_row("Total Rows Processed", f"{total_rows:,}")
    console.print(summary)

    # Show errors if any
    if errors:
        console.print()
        console.print("[red]Errors:[/red]")
        for file_path, error_msg in errors:
            console.print(f"  [yellow]{file_path}[/yellow]: {error_msg}")
        sys.exit(1)
    else:
        console.print()
        console.print(Panel("[green]✓ All files converted successfully![/green]", border_style="green"))
        sys.exit(0)


if __name__ == "__main__":
    main()
