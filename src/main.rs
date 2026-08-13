mod config;
mod hardcoded;
mod output;
mod reservoir;
mod tokenizer;
mod trainer;

use clap::Parser;
use config::Config;
use std::path::PathBuf;
use std::time::Instant;
use trainer::BpeTrainer;

#[derive(Parser, Debug)]
#[command(name = "toktoktok")]
#[command(about = "High-performance BPE tokenizer trainer compatible with tiktoken")]
#[command(version = "1.0.0")]
struct Args {
    /// Path to YAML configuration file
    #[arg(short, long)]
    config: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           TokTokTok BPE Tokenizer Trainer v1.0               ║");
    println!("║        High-performance tiktoken-compatible trainer          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let start = Instant::now();

    // Load configuration
    println!("[1/5] Loading configuration from {:?}...", args.config);
    let config_content = std::fs::read_to_string(&args.config)?;
    let config: Config = serde_yaml::from_str(&config_content)?;

    println!("      Output: {}", config.output);
    println!("      Working set: {} MB", config.working_set_mb.unwrap_or(1024));
    println!("      Threads: {}", config.threads.map_or("auto".to_string(), |t| t.to_string()));
    println!("      Special tokens: {}", config.special_tokens.as_ref().map_or(0, |v| v.len()));
    println!("      Phases: {}", config.phases.len());

    let total_merges: usize = config.phases.iter().map(|p| p.merges).sum();
    println!("      Total merges to train: {}", total_merges);
    println!();

    // Initialize trainer
    let mut trainer = BpeTrainer::new(config)?;

    // Run training
    trainer.train()?;

    let elapsed = start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("Training complete in {:.2}s", elapsed.as_secs_f64());
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
