//! Shared regex patterns for Hugging Face tokenizers.

use crate::{
    join_patterns, spanners::span_lexers::accelerators::RegexAutomataTransformHook,
    support::regex::ConstRegexPattern,
};

/// The Qwen2 pretrained vocabulary word pattern.
///
/// Shared by the Qwen2-family GGUF and Hugging Face tokenizers.
pub(crate) const QWEN2_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy(join_patterns!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?\p{L}+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
));

/// The Qwen3.5 pretrained vocabulary word pattern.
///
/// Shared by the Qwen3.5 tokenizer family loaded via Hugging Face.
pub(crate) const QWEN35_PATTERN: ConstRegexPattern = ConstRegexPattern::Fancy(join_patterns!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
));

/// The Gemma4 pretrained vocabulary word pattern.
///
/// Gemma4 applies a metaspace-like normalizer and then runs BPE across whole
/// lines, so splitting only needs to preserve newline boundaries.
pub(crate) const GEMMA4_PATTERN: ConstRegexPattern =
    ConstRegexPattern::Basic(join_patterns!(r"[^\n]+", r"[\n]+",));

/// Transformed Qwen2 pattern for `regex-automata` (lookahead removed).
pub(crate) const QWEN2_PATTERN_RA: &str = join_patterns!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?\p{L}+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+",
);

inventory::submit! {
    RegexAutomataTransformHook::new(QWEN2_PATTERN, QWEN2_PATTERN_RA, true)
}

/// Transformed Qwen3.5 pattern for `regex-automata` (lookahead removed).
///
/// The `\s+(?!\S)` branch is collapsed to `\s+`; post-processing restores
/// the original end-of-whitespace semantics.
pub(crate) const QWEN35_PATTERN_RA: &str = join_patterns!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"\p{N}",
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"\s*[\r\n]+",
    r"\s+",
);

inventory::submit! {
    RegexAutomataTransformHook::new(QWEN35_PATTERN, QWEN35_PATTERN_RA, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns_compile() {
        assert!(QWEN2_PATTERN.compile().is_ok());
        assert!(QWEN35_PATTERN.compile().is_ok());
        assert!(GEMMA4_PATTERN.compile().is_ok());
    }
}
