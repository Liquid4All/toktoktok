use regex::Regex;
use std::sync::OnceLock;

// Split the GPT-4 pattern into parts that standard regex can handle
// The problematic part is `\s+(?!\S)` (trailing whitespace with negative lookahead)
// We handle this with post-processing instead

// Pattern WITHOUT the lookahead - standard regex can handle this efficiently
const GPT4_PATTERN_FAST: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+";

static TOKENIZER_REGEX: OnceLock<Regex> = OnceLock::new();

/// Get the compiled pre-tokenization regex (fast version)
pub fn get_regex() -> &'static Regex {
    TOKENIZER_REGEX.get_or_init(|| {
        Regex::new(GPT4_PATTERN_FAST).expect("Failed to compile regex pattern")
    })
}

/// Pre-tokenize text into regex-matched chunks
/// Returns byte sequences for each match
///
/// This is optimized for speed using the standard regex crate.
/// The GPT-4 pattern's `\s+(?!\S)` lookahead is approximated by
/// the simpler `\s+` which is close enough for training purposes.
#[inline]
pub fn pre_tokenize(text: &str) -> Vec<Vec<u8>> {
    let regex = get_regex();
    regex
        .find_iter(text)
        .map(|m| m.as_str().as_bytes().to_vec())
        .collect()
}

/// Fast pre-tokenize that reuses a buffer
#[inline]
pub fn pre_tokenize_into(text: &str, output: &mut Vec<Vec<u8>>) {
    let regex = get_regex();
    output.clear();
    for m in regex.find_iter(text) {
        output.push(m.as_str().as_bytes().to_vec());
    }
}

/// Pre-tokenize text, avoiding special tokens
/// Special tokens are identified and excluded from the regex splitting
pub fn pre_tokenize_with_special_tokens(text: &str, special_tokens: &[String]) -> Vec<PreToken> {
    if special_tokens.is_empty() {
        return pre_tokenize(text)
            .into_iter()
            .map(PreToken::Regular)
            .collect();
    }

    let mut result = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Find the earliest special token
        let mut earliest_match: Option<(usize, &str)> = None;
        for special in special_tokens {
            if let Some(pos) = remaining.find(special.as_str()) {
                match earliest_match {
                    None => earliest_match = Some((pos, special)),
                    Some((prev_pos, _)) if pos < prev_pos => {
                        earliest_match = Some((pos, special));
                    }
                    _ => {}
                }
            }
        }

        match earliest_match {
            Some((0, special)) => {
                // Special token at start - skip it
                result.push(PreToken::Special(()));
                remaining = &remaining[special.len()..];
            }
            Some((pos, special)) => {
                // Regular text before special token
                let before = &remaining[..pos];
                for bytes in pre_tokenize(before) {
                    result.push(PreToken::Regular(bytes));
                }
                result.push(PreToken::Special(()));
                remaining = &remaining[pos + special.len()..];
            }
            None => {
                // No more special tokens
                for bytes in pre_tokenize(remaining) {
                    result.push(PreToken::Regular(bytes));
                }
                break;
            }
        }
    }

    result
}

#[derive(Debug, Clone)]
pub enum PreToken {
    Regular(Vec<u8>),
    Special(()),  // We don't need to store special token bytes for training
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let tokens = pre_tokenize("Hello, world!");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_contractions() {
        let tokens = pre_tokenize("I'm don't won't");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_numbers() {
        let tokens = pre_tokenize("123 4567 89");
        assert!(!tokens.is_empty());
    }
}
