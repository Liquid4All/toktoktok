use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    /// Output path for the .tiktoken file
    pub output: String,

    /// Maximum memory usage in MB for the working set
    #[serde(default = "default_working_set")]
    pub working_set_mb: Option<usize>,

    /// Number of threads (-1 for auto)
    pub threads: Option<i32>,

    /// Verbose logging
    #[serde(default)]
    pub verbose: bool,

    /// Special tokens to add at the end of vocabulary
    pub special_tokens: Option<Vec<String>>,

    /// Warm start from existing tokenizer
    pub warm_start: Option<WarmStart>,

    /// Training phases
    pub phases: Vec<Phase>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WarmStart {
    /// Path to existing .tiktoken file
    pub file: PathBuf,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Phase {
    /// Name of this phase (for logging)
    pub name: String,

    /// Number of merges to perform in this phase
    pub merges: usize,

    /// Data sources for this phase
    pub sources: Vec<Source>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Source {
    /// Directory path (recursive scan for .txt/.parquet)
    pub path: Option<PathBuf>,

    /// Single file path
    pub file: Option<PathBuf>,
}

fn default_working_set() -> Option<usize> {
    Some(1024)
}

impl Config {
    pub fn get_thread_count(&self) -> usize {
        match self.threads {
            Some(-1) | None => num_cpus::get(),
            Some(n) if n > 0 => n as usize,
            _ => num_cpus::get(),
        }
    }

    pub fn get_working_set_bytes(&self) -> usize {
        self.working_set_mb.unwrap_or(1024) * 1024 * 1024
    }
}
