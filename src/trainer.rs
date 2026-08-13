use ahash::AHashMap;
use base64::{engine::general_purpose::STANDARD, Engine};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::config::{Config, Source};
use crate::hardcoded::{generate_hardcoded_merges, HardcodedMerge, TRAINED_START};
use crate::reservoir::{fill_reservoir, Reservoir, TokenChunk};

/// A trained merge operation
#[derive(Debug, Clone)]
pub struct TrainedMerge {
    pub left_token: u32,
    pub right_token: u32,
    pub new_token: u32,
}

/// A loaded token from warm start file
#[derive(Debug, Clone)]
pub struct LoadedToken {
    pub bytes: Vec<u8>,
    pub rank: u32,
}

/// BPE Trainer with optional warm start support
#[allow(dead_code)]
pub struct BpeTrainer {
    config: Config,
    /// Mode: either cold start with hardcoded merges, or warm start from file
    mode: TrainerMode,
    /// Map from byte sequence to token ID (for applying during loading)
    bytes_to_id: AHashMap<Vec<u8>, u32>,
    /// Map from (left, right) pair to new token ID
    pair_to_id: AHashMap<(u32, u32), u32>,
    /// Map from raw byte value (0-255) to token ID
    /// For cold start: identity mapping (byte X = token X)
    /// For warm start: looked up from loaded vocabulary
    byte_to_token: [u32; 256],
    /// Trained merges (will be filled during training)
    trained_merges: Vec<TrainedMerge>,
    /// Current vocabulary size
    vocab_size: u32,
}

#[derive(Debug)]
enum TrainerMode {
    /// Cold start: use hardcoded merges
    ColdStart {
        hardcoded_merges: Vec<HardcodedMerge>,
    },
    /// Warm start: loaded from existing .tiktoken file
    WarmStart {
        loaded_tokens: Vec<LoadedToken>,
    },
}

impl BpeTrainer {
    /// Create a new trainer (cold start with hardcoded merges)
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        // Set thread pool size
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.get_thread_count())
            .build_global()
            .ok();

        // Check if warm start is configured
        if let Some(warm_start) = &config.warm_start {
            return Self::from_warm_start(config.clone(), &warm_start.file);
        }

        // Cold start with hardcoded merges
        let (hardcoded_merges, bytes_to_id, pair_to_id) = generate_hardcoded_merges();

        // Cold start: identity mapping (byte X = token X)
        let mut byte_to_token = [0u32; 256];
        for i in 0..256 {
            byte_to_token[i] = i as u32;
        }

        Ok(Self {
            config,
            mode: TrainerMode::ColdStart { hardcoded_merges },
            bytes_to_id,
            pair_to_id,
            byte_to_token,
            trained_merges: Vec::new(),
            vocab_size: TRAINED_START,
        })
    }

    /// Create a trainer from an existing .tiktoken file (warm start)
    fn from_warm_start(config: Config, tiktoken_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[1/{}] Loading warm start vocabulary from {:?}...",
                 config.phases.len() + 2, tiktoken_path);

        let file = File::open(tiktoken_path)?;
        let reader = BufReader::new(file);

        let mut loaded_tokens: Vec<LoadedToken> = Vec::new();
        let mut bytes_to_id: AHashMap<Vec<u8>, u32> = AHashMap::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }

            let bytes = STANDARD.decode(parts[0])?;
            let rank: u32 = parts[1].parse()?;

            bytes_to_id.insert(bytes.clone(), rank);
            loaded_tokens.push(LoadedToken { bytes, rank });
        }

        let vocab_size = loaded_tokens.len() as u32;
        println!("      Loaded {} tokens from warm start file", vocab_size);

        // CRITICAL: Sort loaded_tokens by rank so that loaded_tokens[i].rank == i
        // This is required because reconstruct_token_bytes_warm uses token_id as array index
        loaded_tokens.sort_by_key(|t| t.rank);

        // Validate that ranks are contiguous from 0 to vocab_size-1
        for (i, token) in loaded_tokens.iter().enumerate() {
            if token.rank != i as u32 {
                return Err(format!(
                    "Warm start file has non-contiguous ranks: expected rank {} but found {} at index {}",
                    i, token.rank, i
                ).into());
            }
        }
        println!("      Validated ranks are contiguous 0..{}", vocab_size);

        // Build byte_to_token lookup: maps raw byte value to token ID
        // This is critical because apply_hardcoded_merges needs to convert bytes to token IDs
        let mut byte_to_token_opt: [Option<u32>; 256] = [None; 256];
        let mut missing_bytes = Vec::new();
        for b in 0u8..=255 {
            if let Some(&id) = bytes_to_id.get(&vec![b]) {
                byte_to_token_opt[b as usize] = Some(id);
            } else {
                missing_bytes.push(b);
            }
        }
        if !missing_bytes.is_empty() {
            println!("      WARNING: {} single-byte tokens missing from vocabulary", missing_bytes.len());
            if missing_bytes.len() <= 20 {
                println!("        Missing bytes: {:?}", missing_bytes);
            }
        }

        // Convert to final array - missing bytes use their byte value as fallback
        // (this may cause issues but won't crash)
        let mut byte_to_token = [0u32; 256];
        for b in 0u8..=255 {
            byte_to_token[b as usize] = byte_to_token_opt[b as usize].unwrap_or(b as u32);
        }

        // Debug: show mapping for a few bytes
        println!("      Byte-to-token mapping samples:");
        for &b in &[b' ', b'a', b'e', b't', b'0', b'\n'] {
            let id = byte_to_token[b as usize];
            let token_bytes = &loaded_tokens[id as usize].bytes;
            println!("        byte {:?} ({}) -> token ID {} (bytes: {:?})",
                b as char, b, id,
                String::from_utf8_lossy(token_bytes));
        }

        // Build pair_to_id by finding which tokens can be formed by merging two others
        // This is needed for the hardcoded merge application during data loading
        let pair_to_id = build_pair_to_id(&loaded_tokens, &bytes_to_id);
        println!("      Built merge table with {} pairs", pair_to_id.len());

        Ok(Self {
            config,
            mode: TrainerMode::WarmStart { loaded_tokens },
            bytes_to_id,
            pair_to_id,
            byte_to_token,
            trained_merges: Vec::new(),
            vocab_size,
        })
    }

    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let special_tokens = self.config.special_tokens.clone().unwrap_or_default();
        let working_set_bytes = self.config.get_working_set_bytes();
        let verbose = self.config.verbose;
        let num_phases = self.config.phases.len();

        let is_warm_start = matches!(self.mode, TrainerMode::WarmStart { .. });
        let phase_offset = if is_warm_start { 2 } else { 2 };

        // Clone phase info to avoid borrow checker issues
        let phases: Vec<(String, usize, Vec<Source>)> = self.config.phases
            .iter()
            .map(|p| (p.name.clone(), p.merges, p.sources.clone()))
            .collect();

        for (phase_idx, (name, merges, sources)) in phases.into_iter().enumerate() {
            println!(
                "[{}/{}] Phase: \"{}\" - {} merges (vocab {} -> {})",
                phase_idx + phase_offset,
                num_phases + phase_offset,
                name,
                merges,
                self.vocab_size,
                self.vocab_size + merges as u32
            );

            // Load data for this phase
            println!("      Loading training data...");
            let mut reservoir = fill_reservoir(
                &sources,
                working_set_bytes,
                &self.pair_to_id,
                &self.byte_to_token,
                &special_tokens,
                verbose,
            );

            if reservoir.chunks.is_empty() {
                println!("      Warning: No data loaded for this phase, skipping");
                continue;
            }

            // Train merges for this phase
            self.train_phase(&mut reservoir, merges, verbose)?;
        }

        // Write output
        let output_phase = num_phases + phase_offset;
        println!("[{}/{}] Writing output...", output_phase, output_phase);

        self.write_output(&special_tokens)?;

        Ok(())
    }

    fn write_output(&self, special_tokens: &[String]) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(&self.config.output)?;
        let mut writer = BufWriter::new(file);

        match &self.mode {
            TrainerMode::ColdStart { hardcoded_merges } => {
                // Write base bytes (0-255)
                for byte in 0u8..=255 {
                    let b64 = STANDARD.encode([byte]);
                    writeln!(writer, "{} {}", b64, byte as u32)?;
                }

                // Write hardcoded merges
                for (idx, merge) in hardcoded_merges.iter().enumerate() {
                    let b64 = STANDARD.encode(&merge.bytes);
                    writeln!(writer, "{} {}", b64, 256 + idx as u32)?;
                }

                // Write trained merges
                for merge in &self.trained_merges {
                    let bytes = reconstruct_token_bytes_cold(
                        merge.new_token,
                        hardcoded_merges,
                        &self.trained_merges,
                    );
                    let b64 = STANDARD.encode(&bytes);
                    writeln!(writer, "{} {}", b64, merge.new_token)?;
                }

                let total_vocab = 256 + hardcoded_merges.len() + self.trained_merges.len() + special_tokens.len();
                println!("      Output: {}", self.config.output);
                println!("      Total vocabulary size: {}", total_vocab);
                println!("        - Base bytes: 256");
                println!("        - Hardcoded merges: {}", hardcoded_merges.len());
                println!("        - Trained merges: {}", self.trained_merges.len());
                println!("        - Special tokens: {}", special_tokens.len());
            }

            TrainerMode::WarmStart { loaded_tokens } => {
                // Write all loaded tokens first
                for token in loaded_tokens {
                    let b64 = STANDARD.encode(&token.bytes);
                    writeln!(writer, "{} {}", b64, token.rank)?;
                }

                // Write new trained merges
                let base_vocab = loaded_tokens.len() as u32;
                for merge in &self.trained_merges {
                    let bytes = reconstruct_token_bytes_warm(
                        merge.new_token,
                        loaded_tokens,
                        &self.trained_merges,
                        base_vocab,
                    );
                    let b64 = STANDARD.encode(&bytes);
                    writeln!(writer, "{} {}", b64, merge.new_token)?;
                }

                let total_vocab = loaded_tokens.len() + self.trained_merges.len() + special_tokens.len();
                println!("      Output: {}", self.config.output);
                println!("      Total vocabulary size: {}", total_vocab);
                println!("        - Warm start tokens: {}", loaded_tokens.len());
                println!("        - New trained merges: {}", self.trained_merges.len());
                println!("        - Special tokens: {}", special_tokens.len());
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Get bytes for a token ID (for debug logging)
    fn get_token_bytes(&self, token_id: u32) -> Vec<u8> {
        match &self.mode {
            TrainerMode::ColdStart { hardcoded_merges } => {
                reconstruct_token_bytes_cold(token_id, hardcoded_merges, &self.trained_merges)
            }
            TrainerMode::WarmStart { loaded_tokens } => {
                let base_vocab = loaded_tokens.len() as u32;
                reconstruct_token_bytes_warm(token_id, loaded_tokens, &self.trained_merges, base_vocab)
            }
        }
    }

    fn train_phase(&mut self, reservoir: &mut Reservoir, num_merges: usize, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
        let pb = ProgressBar::new(num_merges as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("      [{elapsed_precise}] [{bar:40.green/white}] {pos}/{len} merges ({per_sec}, eta: {eta})")
                .unwrap()
                .progress_chars("█▓░"),
        );

        let start = Instant::now();
        let mut merges_done = 0;

        let mut skipped_duplicates = 0u64;

        while merges_done < num_merges {
            // Phase 1: Parallel pair counting
            let pair_counts = self.count_pairs_parallel(&reservoir.chunks);

            if pair_counts.is_empty() {
                println!("\n      No more pairs to merge");
                break;
            }

            // Phase 2: Find best pair that doesn't already exist in vocabulary
            // Sort pairs by count descending
            let mut sorted_pairs: Vec<_> = pair_counts.into_iter().collect();
            sorted_pairs.sort_by(|a, b| b.1.cmp(&a.1));

            let mut found_valid_pair = false;
            let mut best_pair = (0u32, 0u32);
            let mut best_count = 0u64;
            let mut left_bytes = Vec::new();
            let mut right_bytes = Vec::new();

            for ((left_token, right_token), count) in sorted_pairs {
                if count < 2 {
                    break; // No more pairs worth merging
                }

                // Get bytes for this pair
                let lb = self.get_token_bytes(left_token);
                let rb = self.get_token_bytes(right_token);

                // Combine bytes and check if this token already exists
                let mut combined = lb.clone();
                combined.extend(&rb);

                if self.bytes_to_id.contains_key(&combined) {
                    // This merge would create a duplicate token - skip it
                    skipped_duplicates += 1;
                    if skipped_duplicates <= 5 {
                        pb.println(format!(
                            "        [SKIP] Pair ({}, {}) -> \"{}\" already exists in vocabulary",
                            left_token, right_token, format_bytes_for_display(&combined)
                        ));
                    }
                    continue;
                }

                // Found a valid pair
                best_pair = (left_token, right_token);
                best_count = count;
                left_bytes = lb;
                right_bytes = rb;
                found_valid_pair = true;
                break;
            }

            if !found_valid_pair {
                println!("\n      No more valid pairs to merge (skipped {} duplicates)", skipped_duplicates);
                break;
            }

            let (left_token, right_token) = best_pair;
            let new_token = self.vocab_size;

            // Add combined bytes to bytes_to_id to prevent future duplicates
            let mut combined = left_bytes.clone();
            combined.extend(&right_bytes);
            self.bytes_to_id.insert(combined, new_token);

            // Also add to pair_to_id for use in subsequent training phases
            self.pair_to_id.insert((left_token, right_token), new_token);

            // Record the merge
            self.trained_merges.push(TrainedMerge {
                left_token,
                right_token,
                new_token,
            });
            self.vocab_size += 1;

            // Phase 3: Apply merge in parallel
            self.apply_merge_parallel(&mut reservoir.chunks, left_token, right_token, new_token);

            merges_done += 1;
            pb.inc(1);

            // Always log first 10 merges, then periodically if verbose
            let should_log = merges_done <= 10 || (verbose && merges_done % 100 == 0);
            if should_log {
                let left_display = format_bytes_for_display(&left_bytes);
                let right_display = format_bytes_for_display(&right_bytes);
                let combined: Vec<u8> = left_bytes.iter().chain(right_bytes.iter()).copied().collect();
                let combined_display = format_bytes_for_display(&combined);
                pb.println(format!(
                    "        Merge {}: \"{}\" + \"{}\" -> \"{}\" (id {} + {} -> {}, count: {})",
                    merges_done,
                    left_display,
                    right_display,
                    combined_display,
                    left_token,
                    right_token,
                    new_token,
                    best_count
                ));
            }
        }

        pb.finish();

        let elapsed = start.elapsed();
        let merges_per_sec = merges_done as f64 / elapsed.as_secs_f64();
        println!(
            "      Completed {} merges in {:.2}s ({:.1} merges/sec)",
            merges_done,
            elapsed.as_secs_f64(),
            merges_per_sec
        );
        if skipped_duplicates > 0 {
            println!(
                "      Skipped {} pairs that would have created duplicate tokens",
                skipped_duplicates
            );
        }

        Ok(())
    }

    /// Count all adjacent pairs in parallel across chunks
    fn count_pairs_parallel(&self, chunks: &[TokenChunk]) -> AHashMap<(u32, u32), u64> {
        chunks
            .par_iter()
            .fold(
                || AHashMap::with_capacity(100_000),
                |mut counts, chunk| {
                    for (left, right, _idx) in chunk.iter_pairs() {
                        *counts.entry((left, right)).or_insert(0) += 1;
                    }
                    counts
                },
            )
            .reduce(
                || AHashMap::new(),
                |mut a, b| {
                    for (pair, count) in b {
                        *a.entry(pair).or_insert(0) += count;
                    }
                    a
                },
            )
    }

    /// Apply a merge operation in parallel across all chunks
    fn apply_merge_parallel(
        &self,
        chunks: &mut [TokenChunk],
        left_token: u32,
        right_token: u32,
        new_token: u32,
    ) {
        chunks.par_iter_mut().for_each(|chunk| {
            self.apply_merge_to_chunk(chunk, left_token, right_token, new_token);
        });
    }

    /// Apply a merge to a single chunk (handles multiple segments)
    fn apply_merge_to_chunk(
        &self,
        chunk: &mut TokenChunk,
        left_token: u32,
        right_token: u32,
        new_token: u32,
    ) {
        for &segment_head in &chunk.segment_heads.clone() {
            let mut current = segment_head;

            while current >= 0 {
                let current_idx = current as usize;
                let next = chunk.next[current_idx];

                if next < 0 {
                    break;
                }

                let next_idx = next as usize;

                if chunk.tokens[current_idx] == left_token && chunk.tokens[next_idx] == right_token {
                    chunk.tokens[current_idx] = new_token;

                    let next_next = chunk.next[next_idx];
                    chunk.next[current_idx] = next_next;

                    if next_next >= 0 {
                        chunk.prev[next_next as usize] = current;
                    }

                    chunk.active_count -= 1;
                }

                current = chunk.next[current_idx];
            }
        }
    }
}

/// Build pair_to_id map from loaded vocabulary
/// This finds all tokens that can be formed by concatenating two shorter tokens
///
/// IMPORTANT: We record ALL valid splits, not just one, because during data loading
/// we might encounter different intermediate representations depending on the order
/// merges are applied. For example, " spider" could come from:
/// - " " + "spider" (if we merged "spider" first)
/// - " sp" + "ider" (if we merged " sp" first)
/// We need to handle all these cases.
fn build_pair_to_id(
    loaded_tokens: &[LoadedToken],
    bytes_to_id: &AHashMap<Vec<u8>, u32>,
) -> AHashMap<(u32, u32), u32> {
    let mut pair_to_id: AHashMap<(u32, u32), u32> = AHashMap::new();

    // For each token, try to find ALL ways it can be split into two existing tokens
    for token in loaded_tokens {
        if token.bytes.len() < 2 {
            continue;
        }

        // Try all possible split points - record ALL valid splits
        for split in 1..token.bytes.len() {
            let left_bytes = &token.bytes[..split];
            let right_bytes = &token.bytes[split..];

            if let (Some(&left_id), Some(&right_id)) =
                (bytes_to_id.get(left_bytes), bytes_to_id.get(right_bytes))
            {
                // Both parts exist - this is a valid merge
                // Only record if left and right have lower rank than this token
                // (this ensures we're recording a valid BPE merge, not a spurious one)
                if left_id < token.rank && right_id < token.rank {
                    // Don't overwrite if we already have a mapping for this pair
                    // (prefer the lowest-rank result, which is the "true" BPE merge)
                    pair_to_id
                        .entry((left_id, right_id))
                        .or_insert(token.rank);
                    // NO break - continue checking other splits!
                }
            }
        }
    }

    pair_to_id
}

/// Reconstruct byte sequence for cold start (with hardcoded merges)
fn reconstruct_token_bytes_cold(
    token_id: u32,
    hardcoded_merges: &[HardcodedMerge],
    trained_merges: &[TrainedMerge],
) -> Vec<u8> {
    if token_id < 256 {
        return vec![token_id as u8];
    }

    if token_id < TRAINED_START {
        let merge = &hardcoded_merges[(token_id - 256) as usize];
        return merge.bytes.clone();
    }

    let merge_idx = (token_id - TRAINED_START) as usize;
    if merge_idx < trained_merges.len() {
        let merge = &trained_merges[merge_idx];
        let mut bytes = reconstruct_token_bytes_cold(merge.left_token, hardcoded_merges, trained_merges);
        bytes.extend(reconstruct_token_bytes_cold(merge.right_token, hardcoded_merges, trained_merges));
        bytes
    } else {
        Vec::new()
    }
}

/// Reconstruct byte sequence for warm start
fn reconstruct_token_bytes_warm(
    token_id: u32,
    loaded_tokens: &[LoadedToken],
    trained_merges: &[TrainedMerge],
    base_vocab: u32,
) -> Vec<u8> {
    if token_id < base_vocab {
        // Token from warm start vocabulary
        return loaded_tokens[token_id as usize].bytes.clone();
    }

    // New trained merge
    let merge_idx = (token_id - base_vocab) as usize;
    if merge_idx < trained_merges.len() {
        let merge = &trained_merges[merge_idx];
        let mut bytes = reconstruct_token_bytes_warm(merge.left_token, loaded_tokens, trained_merges, base_vocab);
        bytes.extend(reconstruct_token_bytes_warm(merge.right_token, loaded_tokens, trained_merges, base_vocab));
        bytes
    } else {
        Vec::new()
    }
}

// Keep the old function for backwards compatibility with output.rs
pub fn reconstruct_token_bytes(
    token_id: u32,
    hardcoded_merges: &[HardcodedMerge],
    trained_merges: &[TrainedMerge],
) -> Vec<u8> {
    reconstruct_token_bytes_cold(token_id, hardcoded_merges, trained_merges)
}

/// Format bytes for human-readable display
/// Shows printable ASCII as-is, non-printable as hex escape
pub fn format_bytes_for_display(bytes: &[u8]) -> String {
    let mut result = String::new();
    for &b in bytes {
        if b >= 0x20 && b < 0x7F {
            // Printable ASCII
            result.push(b as char);
        } else if b == b'\n' {
            result.push_str("\\n");
        } else if b == b'\r' {
            result.push_str("\\r");
        } else if b == b'\t' {
            result.push_str("\\t");
        } else {
            // Non-printable: show hex
            result.push_str(&format!("\\x{:02X}", b));
        }
    }
    result
}
