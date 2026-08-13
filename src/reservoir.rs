use ahash::AHashMap;
use crossbeam::channel::{bounded, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use walkdir::WalkDir;

use crate::config::Source;
use crate::hardcoded::apply_hardcoded_merges;
use crate::tokenizer::{pre_tokenize_with_special_tokens, PreToken};

// Sentinel token to mark boundaries between documents
// Using a value that won't conflict with real tokens
const SENTINEL: u32 = u32::MAX;

// Minimum tokens per chunk to reduce overhead
// With 10K tokens/chunk: overhead is 72 bytes / (10K * 12) = 0.06%
const MIN_CHUNK_SIZE: usize = 10_000;

/// A chunk of tokenized data
/// Stores token IDs in a linked-list structure for efficient merging
#[derive(Debug)]
pub struct TokenChunk {
    /// Token IDs (dense storage) - contains SENTINEL markers between documents
    pub tokens: Vec<u32>,
    /// Next pointers (index-based linked list, -1 = end of segment)
    pub next: Vec<i32>,
    /// Previous pointers (index-based linked list, -1 = start of segment)
    pub prev: Vec<i32>,
    /// Number of active tokens (excludes sentinels and deleted tokens)
    pub active_count: usize,
    /// Starting indices of each chain segment (multiple heads due to sentinels)
    pub segment_heads: Vec<i32>,
}

impl TokenChunk {
    /// Create a new chunk from tokens with sentinel markers
    /// Sentinels are excluded from the linked list to prevent merging across boundaries
    pub fn new(tokens: Vec<u32>) -> Self {
        let len = tokens.len();
        if len == 0 {
            return Self {
                tokens: Vec::new(),
                next: Vec::new(),
                prev: Vec::new(),
                active_count: 0,
                segment_heads: Vec::new(),
            };
        }

        let mut next = vec![-1i32; len];
        let mut prev = vec![-1i32; len];
        let mut segment_heads = Vec::new();
        let mut active_count = 0;
        let mut last_non_sentinel: i32 = -1;
        let mut segment_started = false;

        for i in 0..len {
            if tokens[i] == SENTINEL {
                // Sentinel: break the chain, next segment will start fresh
                last_non_sentinel = -1;
                segment_started = false;
            } else {
                active_count += 1;
                if !segment_started {
                    // Start of a new segment
                    segment_heads.push(i as i32);
                    segment_started = true;
                }
                if last_non_sentinel >= 0 {
                    next[last_non_sentinel as usize] = i as i32;
                    prev[i] = last_non_sentinel;
                }
                last_non_sentinel = i as i32;
            }
        }

        Self {
            tokens,
            next,
            prev,
            active_count,
            segment_heads,
        }
    }

    /// Estimate memory usage in bytes
    pub fn memory_size(&self) -> usize {
        // tokens: 4 bytes per u32
        // next: 4 bytes per i32
        // prev: 4 bytes per i32
        // Total: 12 bytes per token position
        // Plus Vec overhead and struct size
        self.tokens.len() * 12 +
        self.segment_heads.len() * 4 +
        96 + std::mem::size_of::<Self>()
    }

    /// Iterate over adjacent pairs across all segments
    pub fn iter_pairs(&self) -> PairIterator<'_> {
        PairIterator {
            chunk: self,
            segment_idx: 0,
            current: self.segment_heads.first().copied().unwrap_or(-1),
        }
    }
}

pub struct PairIterator<'a> {
    chunk: &'a TokenChunk,
    segment_idx: usize,
    current: i32,
}

impl<'a> Iterator for PairIterator<'a> {
    type Item = (u32, u32, i32); // (left_token, right_token, left_index)

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current < 0 {
                // Try next segment
                self.segment_idx += 1;
                if self.segment_idx >= self.chunk.segment_heads.len() {
                    return None;
                }
                self.current = self.chunk.segment_heads[self.segment_idx];
                continue;
            }

            let current_idx = self.current as usize;
            let next_idx = self.chunk.next[current_idx];

            if next_idx < 0 {
                // End of this segment, try next
                self.segment_idx += 1;
                if self.segment_idx >= self.chunk.segment_heads.len() {
                    return None;
                }
                self.current = self.chunk.segment_heads[self.segment_idx];
                continue;
            }

            let left = self.chunk.tokens[current_idx];
            let right = self.chunk.tokens[next_idx as usize];
            let result = (left, right, self.current);

            self.current = next_idx;
            return Some(result);
        }
    }
}

/// Reservoir of training data with bounded memory usage
pub struct Reservoir {
    /// Chunks of tokenized data
    pub chunks: Vec<TokenChunk>,
    /// Maximum memory in bytes
    max_bytes: usize,
    /// Current estimated memory usage
    current_bytes: usize,
    /// Total items seen (for reservoir sampling)
    total_seen: u64,
    /// Random number generator
    rng: Xoshiro256PlusPlus,
}

impl Reservoir {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            chunks: Vec::new(),
            max_bytes,
            current_bytes: 0,
            total_seen: 0,
            rng: Xoshiro256PlusPlus::seed_from_u64(42),
        }
    }

    /// Add a large consolidated chunk to the reservoir
    pub fn add(&mut self, tokens: Vec<u32>) -> bool {
        if tokens.is_empty() {
            return false;
        }

        let chunk = TokenChunk::new(tokens);
        let chunk_size = chunk.memory_size();

        self.total_seen += 1;

        if self.current_bytes + chunk_size <= self.max_bytes {
            self.current_bytes += chunk_size;
            self.chunks.push(chunk);
            true
        } else {
            // Reservoir sampling
            let reservoir_size = self.chunks.len() as f64;
            let prob = reservoir_size / (self.total_seen as f64);

            if self.rng.gen::<f64>() < prob {
                let idx = self.rng.gen_range(0..self.chunks.len());
                let old_size = self.chunks[idx].memory_size();
                self.current_bytes = self.current_bytes - old_size + chunk_size;
                self.chunks[idx] = chunk;
                true
            } else {
                false
            }
        }
    }

    /// Total number of active tokens across all chunks
    pub fn total_tokens(&self) -> usize {
        self.chunks.iter().map(|c| c.active_count).sum()
    }

    /// Memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_bytes
    }

    pub fn sampling_rate(&self) -> f64 {
        if self.total_seen == 0 {
            1.0
        } else {
            (self.chunks.len() as f64) / (self.total_seen as f64)
        }
    }
}

/// Collect all source files from the given sources
pub fn collect_files(sources: &[Source]) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();

    for source in sources {
        if let Some(path) = &source.path {
            for entry in WalkDir::new(path).follow_links(true).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                    if ext == "txt" || ext == "parquet" {
                        files.push(path.to_path_buf());
                    }
                }
            }
        }

        if let Some(file) = &source.file {
            if file.exists() {
                files.push(file.clone());
            }
        }
    }

    files
}

/// Batch accumulator for consolidating small token sequences
struct TokenBatcher {
    buffer: Vec<u32>,
    sender: Sender<Vec<u32>>,
}

impl TokenBatcher {
    fn new(sender: Sender<Vec<u32>>) -> Self {
        Self {
            buffer: Vec::with_capacity(MIN_CHUNK_SIZE * 2),
            sender,
        }
    }

    fn add(&mut self, tokens: Vec<u32>) {
        if tokens.is_empty() {
            return;
        }

        // Add sentinel if buffer not empty (to separate documents)
        if !self.buffer.is_empty() {
            self.buffer.push(SENTINEL);
        }
        self.buffer.extend(tokens);

        // Flush if we have enough
        if self.buffer.len() >= MIN_CHUNK_SIZE {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if !self.buffer.is_empty() {
            let batch = std::mem::replace(&mut self.buffer, Vec::with_capacity(MIN_CHUNK_SIZE * 2));
            let _ = self.sender.send(batch);
        }
    }
}

impl Drop for TokenBatcher {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Process a batch of lines in parallel, return consolidated tokens
fn process_batch_to_tokens(
    lines: &[String],
    pair_to_id: &AHashMap<(u32, u32), u32>,
    byte_to_token: &[u32; 256],
    special_tokens: &[String],
) -> Vec<u32> {
    // Tokenize in parallel
    let token_seqs: Vec<Vec<u32>> = lines
        .par_iter()
        .flat_map(|line| {
            let pre_tokens = pre_tokenize_with_special_tokens(line, special_tokens);
            pre_tokens
                .into_iter()
                .filter_map(|pt| match pt {
                    PreToken::Regular(bytes) => {
                        let tokens = apply_hardcoded_merges(&bytes, pair_to_id, byte_to_token);
                        if tokens.is_empty() { None } else { Some(tokens) }
                    }
                    PreToken::Special(()) => None,
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Consolidate with sentinels
    let total_len: usize = token_seqs.iter().map(|s| s.len()).sum::<usize>() + token_seqs.len();
    let mut result = Vec::with_capacity(total_len);

    for (i, seq) in token_seqs.into_iter().enumerate() {
        if i > 0 {
            result.push(SENTINEL);
        }
        result.extend(seq);
    }

    result
}

/// Process a text file, batching tokens
fn process_txt_file_batched(
    path: &Path,
    batcher: &mut TokenBatcher,
    pair_to_id: &AHashMap<(u32, u32), u32>,
    byte_to_token: &[u32; 256],
    special_tokens: &[String],
    lines_counter: &AtomicU64,
    verbose: bool,
) -> std::io::Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(256 * 1024, file);

    const BATCH_SIZE: usize = 1000;
    let mut batch = Vec::with_capacity(BATCH_SIZE);
    let mut first_line_logged = false;

    for line in reader.lines() {
        if let Ok(line) = line {
            // Log first line of file for debugging
            if verbose && !first_line_logged && !line.is_empty() {
                let preview: String = line.chars().take(100).collect();
                eprintln!("        [DEBUG] First line of {:?}: {:?}", path.file_name().unwrap_or_default(), preview);
                first_line_logged = true;
            }

            batch.push(line);
            if batch.len() >= BATCH_SIZE {
                let tokens = process_batch_to_tokens(&batch, pair_to_id, byte_to_token, special_tokens);
                lines_counter.fetch_add(batch.len() as u64, Ordering::Relaxed);
                batcher.add(tokens);
                batch.clear();
            }
        }
    }

    if !batch.is_empty() {
        let tokens = process_batch_to_tokens(&batch, pair_to_id, byte_to_token, special_tokens);
        lines_counter.fetch_add(batch.len() as u64, Ordering::Relaxed);
        batcher.add(tokens);
    }

    Ok(())
}

/// Process a parquet file, batching tokens
fn process_parquet_file_batched(
    path: &Path,
    batcher: &mut TokenBatcher,
    pair_to_id: &AHashMap<(u32, u32), u32>,
    byte_to_token: &[u32; 256],
    special_tokens: &[String],
    lines_counter: &AtomicU64,
    verbose: bool,
) -> std::io::Result<()> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    let metadata = reader.metadata();
    let schema = metadata.file_metadata().schema_descr();
    let mut text_col_idx = None;

    for (idx, col) in schema.columns().iter().enumerate() {
        if col.name() == "text" {
            text_col_idx = Some(idx);
            break;
        }
    }

    let text_col_idx = match text_col_idx {
        Some(idx) => idx,
        None => return Ok(()),
    };

    let mut row_iter = reader.get_row_iter(None)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    const BATCH_SIZE: usize = 1000;
    let mut batch = Vec::with_capacity(BATCH_SIZE);
    let mut first_row_logged = false;

    while let Some(row_result) = row_iter.next() {
        if let Ok(row) = row_result {
            if let Ok(text) = row.get_string(text_col_idx) {
                // Log first row of file for debugging
                if verbose && !first_row_logged && !text.is_empty() {
                    let preview: String = text.chars().take(100).collect();
                    eprintln!("        [DEBUG] First row of {:?}: {:?}", path.file_name().unwrap_or_default(), preview);
                    first_row_logged = true;
                }

                batch.push(text.to_string());
                if batch.len() >= BATCH_SIZE {
                    let tokens = process_batch_to_tokens(&batch, pair_to_id, byte_to_token, special_tokens);
                    lines_counter.fetch_add(batch.len() as u64, Ordering::Relaxed);
                    batcher.add(tokens);
                    batch.clear();
                }
            }
        }
    }

    if !batch.is_empty() {
        let tokens = process_batch_to_tokens(&batch, pair_to_id, byte_to_token, special_tokens);
        lines_counter.fetch_add(batch.len() as u64, Ordering::Relaxed);
        batcher.add(tokens);
    }

    Ok(())
}

/// Fill reservoir from sources - memory efficient with batched chunks
pub fn fill_reservoir(
    sources: &[Source],
    max_bytes: usize,
    pair_to_id: &AHashMap<(u32, u32), u32>,
    byte_to_token: &[u32; 256],
    special_tokens: &[String],
    verbose: bool,
) -> Reservoir {
    let files = collect_files(sources);
    let num_files = files.len();

    println!("      Found {} files to process", num_files);
    println!("      Memory budget: {:.2} GB", max_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
    println!("      Chunk size: {} tokens (reduces overhead)", MIN_CHUNK_SIZE);

    let pb = ProgressBar::new(num_files as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("      [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );

    // Bounded channel - each message is now a large batch (~10K tokens)
    let (sender, receiver) = bounded::<Vec<u32>>(1000);

    let lines_counter = AtomicU64::new(0);

    let pair_to_id = pair_to_id.clone();
    let byte_to_token = *byte_to_token; // Copy the array
    let special_tokens = special_tokens.to_vec();
    let pb_clone = pb.clone();

    // Producer: process files sequentially but tokenize in parallel
    // Sequential file processing to control memory, parallel tokenization for speed
    let producer = thread::spawn(move || {
        let mut batcher = TokenBatcher::new(sender);

        for file_path in &files {
            let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");

            match ext {
                "txt" => {
                    let _ = process_txt_file_batched(
                        file_path, &mut batcher, &pair_to_id, &byte_to_token, &special_tokens, &lines_counter, verbose
                    );
                }
                "parquet" => {
                    let _ = process_parquet_file_batched(
                        file_path, &mut batcher, &pair_to_id, &byte_to_token, &special_tokens, &lines_counter, verbose
                    );
                }
                _ => {}
            }

            pb_clone.inc(1);
        }

        // Batcher drops here, flushing remaining tokens
        drop(batcher);
        lines_counter.load(Ordering::Relaxed)
    });

    // Consumer: add batched chunks to reservoir
    let mut reservoir = Reservoir::new(max_bytes);

    for tokens in receiver {
        reservoir.add(tokens);
    }

    let total_lines = producer.join().unwrap_or(0);

    pb.finish_with_message("done");

    println!(
        "      Processed {} lines/rows from {} files",
        total_lines,
        num_files
    );
    println!(
        "      Reservoir: {} chunks, {} tokens, {:.2} GB ({:.2}% of budget)",
        reservoir.chunks.len(),
        reservoir.total_tokens(),
        reservoir.memory_usage() as f64 / 1024.0 / 1024.0 / 1024.0,
        (reservoir.memory_usage() as f64 / max_bytes as f64) * 100.0
    );

    if reservoir.total_seen > reservoir.chunks.len() as u64 {
        println!(
            "      Sampling rate: {:.4}% ({} seen, {} kept)",
            reservoir.sampling_rate() * 100.0,
            reservoir.total_seen,
            reservoir.chunks.len()
        );
    }

    reservoir
}
