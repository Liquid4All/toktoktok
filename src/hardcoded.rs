use ahash::AHashMap;

/// Hardcoded merge information
/// Contains the byte sequence and its assigned token ID
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HardcodedMerge {
    pub bytes: Vec<u8>,
    pub token_id: u32,
    pub left_token: u32,
    pub right_token: u32,
}

/// Number of base byte tokens (0-255)
pub const BASE_VOCAB_SIZE: u32 = 256;

/// Starting ID for hardcoded merges
pub const HARDCODED_START: u32 = 256;

/// Total number of hardcoded merges
pub const HARDCODED_COUNT: u32 = 1161;

/// First trained merge ID
pub const TRAINED_START: u32 = HARDCODED_START + HARDCODED_COUNT; // 1417

/// Generate all hardcoded merges
/// Returns:
/// - Vector of all hardcoded merges in order
/// - Map from byte sequence to token ID (includes base bytes)
/// - Map from (left, right) pair to new token ID
pub fn generate_hardcoded_merges() -> (Vec<HardcodedMerge>, AHashMap<Vec<u8>, u32>, AHashMap<(u32, u32), u32>) {
    let mut merges = Vec::with_capacity(HARDCODED_COUNT as usize);
    let mut bytes_to_id: AHashMap<Vec<u8>, u32> = AHashMap::with_capacity(BASE_VOCAB_SIZE as usize + HARDCODED_COUNT as usize);
    let mut pair_to_id: AHashMap<(u32, u32), u32> = AHashMap::with_capacity(HARDCODED_COUNT as usize);

    // Initialize base bytes (0-255)
    for i in 0..256u32 {
        bytes_to_id.insert(vec![i as u8], i);
    }

    let mut current_id = HARDCODED_START;

    // === Two-digit numbers (100 merges, IDs 256-355) ===
    // "00" through "99"
    for tens in 0..10u8 {
        for ones in 0..10u8 {
            let left = (b'0' + tens) as u32;
            let right = (b'0' + ones) as u32;
            let bytes = vec![b'0' + tens, b'0' + ones];

            merges.push(HardcodedMerge {
                bytes: bytes.clone(),
                token_id: current_id,
                left_token: left,
                right_token: right,
            });
            bytes_to_id.insert(bytes, current_id);
            pair_to_id.insert((left, right), current_id);
            current_id += 1;
        }
    }

    // === Three-digit numbers (1000 merges, IDs 356-1355) ===
    // "000" through "999"
    // Each is formed by merging a two-digit token with a single digit
    for hundreds in 0..10u8 {
        for tens in 0..10u8 {
            for ones in 0..10u8 {
                // Two-digit prefix token ID
                let two_digit_id = HARDCODED_START + (hundreds as u32 * 10) + (tens as u32);
                let right = (b'0' + ones) as u32;
                let bytes = vec![b'0' + hundreds, b'0' + tens, b'0' + ones];

                merges.push(HardcodedMerge {
                    bytes: bytes.clone(),
                    token_id: current_id,
                    left_token: two_digit_id,
                    right_token: right,
                });
                bytes_to_id.insert(bytes, current_id);
                pair_to_id.insert((two_digit_id, right), current_id);
                current_id += 1;
            }
        }
    }

    // === Multi-space tokens (31 merges, IDs 1356-1386) ===
    // 2 to 32 consecutive spaces, built hierarchically
    let space = b' ' as u32;
    let mut space_tokens: Vec<(u32, Vec<u8>)> = vec![(space, vec![b' '])]; // 1 space

    // 2 spaces: ' ' + ' '
    {
        let bytes = vec![b' ', b' '];
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token: space,
            right_token: space,
        });
        bytes_to_id.insert(bytes.clone(), current_id);
        pair_to_id.insert((space, space), current_id);
        space_tokens.push((current_id, bytes));
        current_id += 1;
    }

    // 3-32 spaces: build hierarchically
    for n in 3..=32usize {
        // Find best split point (closest to n/2 for balanced tree)
        let half = n / 2;
        let left_idx = half; // number of spaces in left part
        let right_idx = n - half; // number of spaces in right part

        let left_token = space_tokens[left_idx - 1].0;
        let right_token = space_tokens[right_idx - 1].0;

        let bytes = vec![b' '; n];
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token,
            right_token,
        });
        bytes_to_id.insert(bytes.clone(), current_id);
        pair_to_id.insert((left_token, right_token), current_id);
        space_tokens.push((current_id, bytes));
        current_id += 1;
    }

    // === Multi-newline tokens (7 merges, IDs 1387-1393) ===
    // 2 to 8 consecutive newlines, built hierarchically
    let newline = b'\n' as u32;
    let mut newline_tokens: Vec<(u32, Vec<u8>)> = vec![(newline, vec![b'\n'])]; // 1 newline

    // 2 newlines: '\n' + '\n'
    {
        let bytes = vec![b'\n', b'\n'];
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token: newline,
            right_token: newline,
        });
        bytes_to_id.insert(bytes.clone(), current_id);
        pair_to_id.insert((newline, newline), current_id);
        newline_tokens.push((current_id, bytes));
        current_id += 1;
    }

    // 3-8 newlines: build hierarchically
    for n in 3..=8usize {
        let half = n / 2;
        let left_idx = half;
        let right_idx = n - half;

        let left_token = newline_tokens[left_idx - 1].0;
        let right_token = newline_tokens[right_idx - 1].0;

        let bytes = vec![b'\n'; n];
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token,
            right_token,
        });
        bytes_to_id.insert(bytes.clone(), current_id);
        pair_to_id.insert((left_token, right_token), current_id);
        newline_tokens.push((current_id, bytes));
        current_id += 1;
    }

    // === Windows line ending (1 merge, ID 1394) ===
    {
        let bytes = vec![b'\r', b'\n'];
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token: b'\r' as u32,
            right_token: b'\n' as u32,
        });
        bytes_to_id.insert(bytes, current_id);
        pair_to_id.insert((b'\r' as u32, b'\n' as u32), current_id);
        current_id += 1;
    }

    // === Programming operators (20 merges, IDs 1395-1414) ===
    let operators = [
        ("==", b'=' as u32, b'=' as u32),
        ("!=", b'!' as u32, b'=' as u32),
        ("<=", b'<' as u32, b'=' as u32),
        (">=", b'>' as u32, b'=' as u32),
        ("+=", b'+' as u32, b'=' as u32),
        ("-=", b'-' as u32, b'=' as u32),
        ("*=", b'*' as u32, b'=' as u32),
        ("/=", b'/' as u32, b'=' as u32),
        ("->", b'-' as u32, b'>' as u32),
        ("=>", b'=' as u32, b'>' as u32),
        ("::", b':' as u32, b':' as u32),
        ("//", b'/' as u32, b'/' as u32),
        ("/*", b'/' as u32, b'*' as u32),
        ("*/", b'*' as u32, b'/' as u32),
        ("&&", b'&' as u32, b'&' as u32),
        ("||", b'|' as u32, b'|' as u32),
        ("++", b'+' as u32, b'+' as u32),
        ("--", b'-' as u32, b'-' as u32),
        ("<<", b'<' as u32, b'<' as u32),
        (">>", b'>' as u32, b'>' as u32),
    ];

    for (op, left, right) in operators {
        let bytes = op.as_bytes().to_vec();
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token: left,
            right_token: right,
        });
        bytes_to_id.insert(bytes, current_id);
        pair_to_id.insert((left, right), current_id);
        current_id += 1;
    }

    // === Ellipsis (2 merges, IDs 1415-1416) ===
    // ".." = '.' + '.'
    {
        let bytes = vec![b'.', b'.'];
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token: b'.' as u32,
            right_token: b'.' as u32,
        });
        bytes_to_id.insert(bytes, current_id);
        pair_to_id.insert((b'.' as u32, b'.' as u32), current_id);
        let double_dot_id = current_id;
        current_id += 1;

        // "..." = ".." + '.'
        let bytes = vec![b'.', b'.', b'.'];
        merges.push(HardcodedMerge {
            bytes: bytes.clone(),
            token_id: current_id,
            left_token: double_dot_id,
            right_token: b'.' as u32,
        });
        bytes_to_id.insert(bytes, current_id);
        pair_to_id.insert((double_dot_id, b'.' as u32), current_id);
        // current_id += 1; // Not needed, this is the last
    }

    assert_eq!(merges.len(), HARDCODED_COUNT as usize, "Hardcoded merge count mismatch");

    (merges, bytes_to_id, pair_to_id)
}

/// Apply hardcoded merges to a byte sequence
/// Returns the token IDs after applying all applicable hardcoded merges
///
/// Arguments:
/// - bytes: Raw UTF-8 bytes from pre-tokenization
/// - pair_to_id: Map from (left_token, right_token) to merged token ID
/// - byte_to_token: Map from raw byte value to token ID (handles warm start vocabularies
///   where byte 0 might not be token 0)
pub fn apply_hardcoded_merges(
    bytes: &[u8],
    pair_to_id: &AHashMap<(u32, u32), u32>,
    byte_to_token: &[u32; 256],
) -> Vec<u32> {
    if bytes.is_empty() {
        return Vec::new();
    }

    // Convert raw bytes to token IDs using the byte_to_token mapping
    // This is critical for warm start where byte X may not be token X
    let mut tokens: Vec<u32> = bytes.iter().map(|&b| byte_to_token[b as usize]).collect();

    // Keep merging until no more merges are possible
    let mut changed = true;
    while changed {
        changed = false;
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut i = 0;

        while i < tokens.len() {
            if i + 1 < tokens.len() {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&new_id) = pair_to_id.get(&pair) {
                    new_tokens.push(new_id);
                    i += 2;
                    changed = true;
                    continue;
                }
            }
            new_tokens.push(tokens[i]);
            i += 1;
        }

        tokens = new_tokens;
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardcoded_count() {
        let (merges, _, _) = generate_hardcoded_merges();
        assert_eq!(merges.len(), HARDCODED_COUNT as usize);
    }

    #[test]
    fn test_two_digit_numbers() {
        let (_, bytes_to_id, _) = generate_hardcoded_merges();

        // Check "00" has ID 256
        assert_eq!(bytes_to_id.get(&vec![b'0', b'0']), Some(&256));
        // Check "99" has ID 355
        assert_eq!(bytes_to_id.get(&vec![b'9', b'9']), Some(&355));
    }

    #[test]
    fn test_three_digit_numbers() {
        let (_, bytes_to_id, _) = generate_hardcoded_merges();

        // Check "000" has ID 356
        assert_eq!(bytes_to_id.get(&vec![b'0', b'0', b'0']), Some(&356));
        // Check "999" has ID 1355
        assert_eq!(bytes_to_id.get(&vec![b'9', b'9', b'9']), Some(&1355));
    }

    #[test]
    fn test_apply_merges() {
        let (_, _, pair_to_id) = generate_hardcoded_merges();

        // Cold start: byte X maps to token X
        let mut byte_to_token = [0u32; 256];
        for i in 0..256 {
            byte_to_token[i] = i as u32;
        }

        // Test merging "12" -> should become single token
        let result = apply_hardcoded_merges(b"12", &pair_to_id, &byte_to_token);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], 256 + 12); // ID for "12"
    }
}
