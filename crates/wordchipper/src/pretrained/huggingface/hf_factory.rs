use std::path::Path;

use gguf::{
    GGUFFile,
    GGUFMetadata,
    GGUFMetadataValue,
    GGUfMetadataValueType,
};
use serde_json::Value;
use tokenizers::{
    ModelWrapper::BPE,
    PreTokenizerWrapper,
    PreTokenizerWrapper::{
        ByteLevel,
        Sequence,
        Split,
    },
    pre_tokenizers::split::SplitPattern,
    tokenizer::{
        NormalizerWrapper,
        SplitDelimiterBehavior,
        Tokenizer,
    },
};

use crate::{
    LabeledVocab,
    UnifiedTokenVocab,
    VocabDescription,
    VocabIndex,
    VocabQuery,
    WCError,
    WCHashMap,
    WCHashSet,
    WCResult,
    alloc::sync::Arc,
    prelude::*,
    pretrained::{
        factory::{
            VocabProvider,
            VocabProviderInventoryHook,
        },
        huggingface::patterns::{
            GEMMA4_PATTERN,
            QWEN2_PATTERN,
            QWEN35_PATTERN,
        },
        openai::{
            OA_CL100K_BASE_PATTERN,
            OA_GPT2_PATTERN,
        },
    },
    spanners::TextSpanningConfig,
    support::{
        normalization::TextNormalizer,
        regex::RegexPattern,
        resources::ResourceLoader,
    },
    vocab::{
        ByteMapVocab,
        PairMapVocab,
        PairRankMap,
        PairTokenMap,
        SpanMapVocab,
        SpanTokenMap,
        TokenSpanMap,
    },
};

const HF_WHOLE_SEGMENT_PATTERN: &str = r"[\s\S]+";
const GGUF_TOKENIZER_HF_JSON_KEY: &str = "tokenizer.huggingface.json";
const GGUF_TOKENIZER_MODEL_KEY: &str = "tokenizer.ggml.model";
const GGUF_TOKENIZER_PRE_KEY: &str = "tokenizer.ggml.pre";
const GGUF_TOKENIZER_TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const GGUF_TOKENIZER_TOKEN_TYPE_KEY: &str = "tokenizer.ggml.token_type";
const GGUF_TOKENIZER_MERGES_KEY: &str = "tokenizer.ggml.merges";
const GGUF_TOKEN_TYPE_UNKNOWN: i32 = 2;
const GGUF_TOKEN_TYPE_CONTROL: i32 = 3;
const GGUF_TOKEN_TYPE_USER_DEFINED: i32 = 4;
const GGUF_SPECIAL_TOKEN_ID_KEYS: &[&str] = &[
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eot_token_id",
    "tokenizer.ggml.eom_token_id",
    "tokenizer.ggml.unk_token_id",
    "tokenizer.ggml.sep_token_id",
    "tokenizer.ggml.pad_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.fim_pre_token_id",
    "tokenizer.ggml.fim_suf_token_id",
    "tokenizer.ggml.fim_mid_token_id",
    "tokenizer.ggml.fim_pad_token_id",
    "tokenizer.ggml.fim_rep_token_id",
    "tokenizer.ggml.fim_sep_token_id",
    "tokenizer.ggml.prefix_token_id",
    "tokenizer.ggml.suffix_token_id",
    "tokenizer.ggml.middle_token_id",
];

enum HFBpeEncoding {
    ByteLevel(WCHashMap<char, u8>),
    ByteFallback,
}

fn extract_pattern(
    pt: Option<&PreTokenizerWrapper>,
    normalizer: Option<&TextNormalizer>,
) -> Result<RegexPattern, WCError> {
    fn split_pattern(
        s: &tokenizers::pre_tokenizers::split::Split,
        normalizer: Option<&TextNormalizer>,
    ) -> Result<RegexPattern, WCError> {
        match &s.pattern {
            SplitPattern::Regex(r) => Ok(r.clone().into()),
            SplitPattern::String(pattern)
                if !s.invert
                    && s.behavior == SplitDelimiterBehavior::MergedWithPrevious
                    && matches!(
                        normalizer,
                        Some(TextNormalizer::Replace {
                            pattern: replace_pattern,
                            ..
                        }) if replace_pattern == pattern
                    ) =>
            {
                Ok(HF_WHOLE_SEGMENT_PATTERN.into())
            }
            SplitPattern::String(pattern) => Err(WCError::External(crate::alloc::format!(
                "unsupported string Split pre-tokenizer: pattern={pattern:?}, behavior={:?}, invert={}",
                s.behavior,
                s.invert
            ))),
        }
    }
    match pt {
        Some(Split(s)) => split_pattern(s, normalizer),
        Some(ByteLevel(bl)) if bl.use_regex => Ok(OA_GPT2_PATTERN.into()),
        Some(ByteLevel(_)) => Err(WCError::External(
            "ByteLevel with use_regex=false has no splitting regex".into(),
        )),
        Some(Sequence(seq)) => {
            let mut found = None;
            for sub in seq.as_ref() {
                match &sub {
                    Split(s) => {
                        if found.is_some() {
                            return Err(WCError::External("Sequence has multiple Splits".into()));
                        }
                        found = Some(split_pattern(s, normalizer)?);
                    }
                    ByteLevel(_) => {} // sibling byte-encoder, fine
                    _ => return Err(WCError::External("unsupported member in Sequence".into())),
                }
            }
            found.ok_or_else(|| WCError::External("Sequence has no Split regex".into()))
        }
        Some(_) => Err(WCError::External("unsupported pre-tokenizer".into())),
        None => Err(WCError::External("no pre-tokenizer".into())),
    }
}

fn extract_replace_normalizer(
    replace: &tokenizers::normalizers::Replace
) -> WCResult<TextNormalizer> {
    let value = serde_json::to_value(replace).map_err(|error| {
        WCError::External(crate::alloc::format!(
            "failed to serialize huggingface Replace normalizer: {error}"
        ))
    })?;

    let pattern = value
        .get("pattern")
        .and_then(|pattern| pattern.get("String"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            WCError::External(crate::alloc::format!(
                "unsupported huggingface Replace normalizer pattern: {value:?}"
            ))
        })?;

    if pattern.is_empty() {
        return Err(WCError::External(
            "unsupported huggingface Replace normalizer with empty pattern".into(),
        ));
    }

    let replacement = value
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            WCError::External(crate::alloc::format!(
                "unsupported huggingface Replace normalizer content: {value:?}"
            ))
        })?;

    Ok(TextNormalizer::Replace {
        pattern: pattern.into(),
        replacement: replacement.into(),
    })
}

fn extract_text_normalizer(normalizer: &NormalizerWrapper) -> WCResult<TextNormalizer> {
    match normalizer {
        NormalizerWrapper::NFC(_) => Ok(TextNormalizer::NFC),
        NormalizerWrapper::NFD(_) => Ok(TextNormalizer::NFD),
        NormalizerWrapper::NFKC(_) => Ok(TextNormalizer::NFKC),
        NormalizerWrapper::NFKD(_) => Ok(TextNormalizer::NFKD),
        NormalizerWrapper::Replace(replace) => extract_replace_normalizer(replace),
        NormalizerWrapper::Sequence(sequence) => sequence
            .as_ref()
            .iter()
            .map(extract_text_normalizer)
            .collect::<WCResult<Vec<_>>>()
            .map(TextNormalizer::Sequence),
        _ => Err(WCError::External(crate::alloc::format!(
            "unsupported huggingface normalizer: {normalizer:?}"
        ))),
    }
}

fn extract_normalizer(normalizer: Option<&NormalizerWrapper>) -> WCResult<Option<TextNormalizer>> {
    normalizer.map(extract_text_normalizer).transpose()
}

fn normalize_input<'a>(
    normalizer: Option<&TextNormalizer>,
    text: &'a str,
) -> crate::alloc::borrow::Cow<'a, str> {
    normalizer
        .map(|normalizer| normalizer.normalize(text))
        .unwrap_or_else(|| crate::alloc::borrow::Cow::Borrowed(text))
}

/// Converts bytes to Unicode characters.
/// See <https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9>
///
/// This is from tokenizers; but is private in that crate.
///
/// TODO: Workout what this is doing, relative to the bytemap.
/// This seems to be some default map for gpt2; and might be shared
/// with the `BytMap` code for loading datagym.
fn bytes_char() -> WCHashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }

    // Safety: cs contains all values from bs (between 0 and 255),
    // and some values of value 2⁸ + n, where n is between 0 and 255. This is
    // between 255 and 512. Both ranges are valid UTF-32 values (which is fully
    // saturated until 0xD000)
    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}

fn byte_fallback_token(byte: u8) -> String {
    crate::alloc::format!("<0x{byte:02X}>")
}

fn byte_fallback_value(token: &str) -> Option<u8> {
    if token.len() != 6 || !token.starts_with("<0x") || !token.ends_with('>') {
        return None;
    }

    u8::from_str_radix(&token[3..5], 16).ok()
}

fn extract_bpe_encoding(
    hf_vocab: &std::collections::HashMap<String, u32>
) -> WCResult<HFBpeEncoding> {
    let byte_chars = bytes_char();
    let byte_level = (0u8..=255).all(|byte| {
        let key: String = std::iter::once(byte_chars[&byte]).collect();
        hf_vocab.contains_key(&key)
    });
    if byte_level {
        return Ok(HFBpeEncoding::ByteLevel(
            byte_chars.iter().map(|(&byte, &ch)| (ch, byte)).collect(),
        ));
    }

    let fallback = (0u8..=255).all(|byte| hf_vocab.contains_key(&byte_fallback_token(byte)));
    if fallback {
        return Ok(HFBpeEncoding::ByteFallback);
    }

    Err(WCError::External(
        "unsupported BPE byte encoding: vocab must expose either GPT-2 byte-level tokens or the full <0xXX> byte fallback set"
            .into(),
    ))
}

fn token_to_bytes(
    token: &str,
    encoding: &HFBpeEncoding,
) -> WCResult<Option<Vec<u8>>> {
    match encoding {
        HFBpeEncoding::ByteLevel(char_to_byte) => {
            let mut bytes = Vec::with_capacity(token.len());

            for ch in token.chars() {
                match char_to_byte.get(&ch) {
                    Some(&byte) => bytes.push(byte),
                    None => {
                        return Err(WCError::External(crate::alloc::format!(
                            "token {token:?} has non-byte-level codepoint {ch:?}"
                        )));
                    }
                }
            }

            Ok(Some(bytes))
        }
        HFBpeEncoding::ByteFallback => {
            if byte_fallback_value(token).is_some() {
                Ok(None)
            } else {
                let bytes = token.as_bytes().to_vec();
                if bytes.len() == 1 {
                    Ok(None)
                } else {
                    Ok(Some(bytes))
                }
            }
        }
    }
}

fn extract_byte_tokens(
    hf_vocab: &std::collections::HashMap<String, u32>,
    encoding: &HFBpeEncoding,
) -> WCResult<Vec<u32>> {
    (0u8..=255)
        .map(|byte| {
            let key = match encoding {
                HFBpeEncoding::ByteLevel(_) => {
                    let byte_chars = bytes_char();
                    std::iter::once(byte_chars[&byte]).collect::<String>()
                }
                HFBpeEncoding::ByteFallback => {
                    if byte.is_ascii() {
                        let ch = char::from(byte);
                        let single = std::iter::once(ch).collect::<String>();
                        if hf_vocab.contains_key(&single) {
                            single
                        } else {
                            byte_fallback_token(byte)
                        }
                    } else {
                        byte_fallback_token(byte)
                    }
                }
            };

            hf_vocab.get(&key).copied().ok_or(byte)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|byte| {
            WCError::External(crate::alloc::format!("missing byte token for 0x{byte:02x}"))
        })
}

fn extract_serialized_merges(
    bpe: &tokenizers::models::bpe::BPE
) -> WCResult<Vec<(String, String)>> {
    let value = serde_json::to_value(bpe).map_err(|error| {
        WCError::External(crate::alloc::format!(
            "failed to serialize huggingface BPE model: {error}"
        ))
    })?;

    value
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| WCError::External("huggingface BPE model is missing merges".into()))?
        .iter()
        .map(|entry| match entry {
            Value::Array(parts) if parts.len() == 2 => {
                let a = parts[0].as_str().ok_or_else(|| {
                    WCError::External(crate::alloc::format!(
                        "invalid first merge token in {entry:?}"
                    ))
                })?;
                let b = parts[1].as_str().ok_or_else(|| {
                    WCError::External(crate::alloc::format!(
                        "invalid second merge token in {entry:?}"
                    ))
                })?;
                Ok((a.to_string(), b.to_string()))
            }
            Value::String(line) => line
                .split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| {
                    WCError::External(crate::alloc::format!("invalid legacy merge entry {line:?}"))
                }),
            _ => Err(WCError::External(crate::alloc::format!(
                "invalid merge entry {entry:?}"
            ))),
        })
        .collect()
}

fn extract_unicode_scalar_seed_tokens(
    hf_vocab: &std::collections::HashMap<String, u32>,
    special_tokens: &WCHashSet<u32>,
) -> SpanTokenMap<u32> {
    hf_vocab
        .iter()
        .filter(|(token, id)| {
            !special_tokens.contains(id)
                && byte_fallback_value(token).is_none()
                && token.chars().count() == 1
        })
        .map(|(token, &id)| (token.as_bytes().to_vec(), id))
        .collect()
}

struct SerializedBpeConfig {
    pattern: RegexPattern,
    input_normalizer: Option<TextNormalizer>,
    hf_vocab: std::collections::HashMap<String, u32>,
    merges: Vec<(String, String)>,
    special_tokens: Vec<(u32, String)>,
    continuing_subword_prefix: Option<String>,
    ignore_merges: bool,
    unk_token: Option<String>,
}

struct GGUFBpePreset {
    pattern: RegexPattern,
    input_normalizer: Option<TextNormalizer>,
    ignore_merges: bool,
}

fn build_exact_pair_vocab_from_parts(
    hf_vocab: &std::collections::HashMap<String, u32>,
    encoding: &HFBpeEncoding,
    byte_map: ByteMapVocab<u32>,
    merges: &[(String, String)],
    continuing_subword_prefix: Option<&str>,
    special_tokens: &WCHashSet<u32>,
) -> WCResult<PairMapVocab<u32>> {
    let mut primitive_spans: TokenSpanMap<u32> = byte_map
        .span_pairs()
        .map(|(span, token)| (token, span))
        .collect();

    if matches!(encoding, HFBpeEncoding::ByteFallback) {
        if continuing_subword_prefix.is_some() {
            return Err(WCError::External(
                "byte-fallback BPE with prefix/suffix affixes is not yet supported".into(),
            ));
        }

        primitive_spans.extend(
            extract_unicode_scalar_seed_tokens(hf_vocab, special_tokens)
                .into_iter()
                .map(|(span, token)| (token, span)),
        );
    }

    let mut pair_map: PairTokenMap<u32> = PairTokenMap::default();
    let mut pair_ranks: PairRankMap<u32> = PairRankMap::default();
    let prefix = continuing_subword_prefix.unwrap_or("");

    for (rank, (a, b)) in merges.iter().enumerate() {
        let a_id = hf_vocab.get(a).copied().ok_or_else(|| {
            WCError::External(crate::alloc::format!(
                "merge token {a:?} missing from huggingface vocab"
            ))
        })?;
        let b_id = hf_vocab.get(b).copied().ok_or_else(|| {
            WCError::External(crate::alloc::format!(
                "merge token {b:?} missing from huggingface vocab"
            ))
        })?;

        if special_tokens.contains(&a_id) || special_tokens.contains(&b_id) {
            return Err(WCError::External(crate::alloc::format!(
                "special token found in merge pair ({a:?}, {b:?})"
            )));
        }

        let suffix = b.strip_prefix(prefix).ok_or_else(|| {
            WCError::External(crate::alloc::format!(
                "merge token {b:?} is missing expected continuing_subword_prefix {prefix:?}"
            ))
        })?;
        let merged = crate::alloc::format!("{a}{suffix}");
        let merged_id = hf_vocab.get(&merged).copied().ok_or_else(|| {
            WCError::External(crate::alloc::format!(
                "merge target token {merged:?} missing from huggingface vocab"
            ))
        })?;

        if special_tokens.contains(&merged_id) {
            return Err(WCError::External(crate::alloc::format!(
                "special token found as merge target {merged:?}"
            )));
        }

        pair_map.insert((a_id, b_id), merged_id);
        pair_ranks.insert((a_id, b_id), rank as u32);
    }

    PairMapVocab::new_with_parts(byte_map, primitive_spans, pair_map, pair_ranks)
}

fn build_vocab_from_serialized_bpe(
    config: SerializedBpeConfig
) -> WCResult<Arc<UnifiedTokenVocab<u32>>> {
    type T = u32;

    let SerializedBpeConfig {
        pattern,
        input_normalizer,
        hf_vocab,
        merges,
        special_tokens: serialized_special_tokens,
        continuing_subword_prefix,
        ignore_merges,
        unk_token,
    } = config;

    let mut span_config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(pattern);
    let encoding = extract_bpe_encoding(&hf_vocab)?;

    let mut special_tokens: WCHashSet<T> = Default::default();
    for (token, special_content) in serialized_special_tokens {
        span_config
            .specials_mut()
            .add_str_word(&special_content, token);
        special_tokens.insert(token);
    }

    let mut span_map: SpanTokenMap<T> = SpanTokenMap::default();
    for (token_text, id) in &hf_vocab {
        if special_tokens.contains(id) {
            continue;
        }

        if let Some(bytes) = token_to_bytes(token_text, &encoding)? {
            span_map.insert(bytes, *id);
        }
    }

    if span_config.specials().len() != special_tokens.len() {
        return Err(WCError::External(format!(
            "hf vocab identifies {} special tokens, but only {} special tokens found in span_config",
            special_tokens.len(),
            span_config.specials().len()
        )));
    }

    let byte_tokens: Vec<T> = extract_byte_tokens(&hf_vocab, &encoding)?;

    let byte_map = ByteMapVocab::<T>::from_byte_to_token(&byte_tokens);
    let span_vocab = SpanMapVocab::<T>::new(byte_map, span_map)?;
    let pair_vocab = build_exact_pair_vocab_from_parts(
        &hf_vocab,
        &encoding,
        span_vocab.byte_vocab().clone(),
        &merges,
        continuing_subword_prefix.as_deref(),
        &special_tokens,
    )?;

    let expected_len = span_vocab.len() + span_config.specials().len();

    let mut vocab = UnifiedTokenVocab::new(span_config, span_vocab, pair_vocab)?;
    vocab = vocab.with_direct_word_lookup(ignore_merges);
    if matches!(encoding, HFBpeEncoding::ByteFallback) {
        let unk_token = unk_token
            .as_ref()
            .and_then(|token| hf_vocab.get(token))
            .copied();
        vocab = vocab.with_unicode_scalar_seeding(
            extract_unicode_scalar_seed_tokens(&hf_vocab, &special_tokens),
            byte_tokens.clone(),
            unk_token,
        );
    }

    let vocab = if let Some(normalizer) = input_normalizer {
        vocab.with_input_normalizer(normalizer)
    } else {
        vocab
    };
    let vocab: Arc<UnifiedTokenVocab<T>> = Arc::new(vocab);

    if vocab.len() + vocab.special_vocab().len() != expected_len {
        return Err(WCError::External(format!(
            "Expected {} tokens, got {}",
            expected_len,
            vocab.len()
        )));
    }

    Ok(vocab)
}

fn gguf_metadata_value<'a>(
    metadata: &'a [GGUFMetadata],
    key: &str,
) -> Option<&'a GGUFMetadataValue> {
    metadata
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| &entry.value)
}

fn gguf_optional_string(
    metadata: &[GGUFMetadata],
    key: &str,
) -> WCResult<Option<String>> {
    match gguf_metadata_value(metadata, key) {
        Some(GGUFMetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(other) => Err(WCError::External(crate::alloc::format!(
            "GGUF metadata key {key:?} must be a string, found {other:?}"
        ))),
        None => Ok(None),
    }
}

fn gguf_required_string(
    metadata: &[GGUFMetadata],
    key: &str,
) -> WCResult<String> {
    gguf_optional_string(metadata, key)?.ok_or_else(|| {
        WCError::External(crate::alloc::format!(
            "GGUF metadata is missing required key {key:?}"
        ))
    })
}

fn gguf_optional_u32(
    metadata: &[GGUFMetadata],
    key: &str,
) -> WCResult<Option<u32>> {
    match gguf_metadata_value(metadata, key) {
        Some(GGUFMetadataValue::Uint32(value)) => Ok(Some(*value)),
        Some(GGUFMetadataValue::Int32(value)) if *value >= 0 => Ok(Some(*value as u32)),
        Some(GGUFMetadataValue::Uint64(value)) if *value <= u32::MAX as u64 => {
            Ok(Some(*value as u32))
        }
        Some(GGUFMetadataValue::Int64(value)) if (0..=u32::MAX as i64).contains(value) => {
            Ok(Some(*value as u32))
        }
        Some(other) => Err(WCError::External(crate::alloc::format!(
            "GGUF metadata key {key:?} must be an integer token id, found {other:?}"
        ))),
        None => Ok(None),
    }
}

fn gguf_required_string_array(
    metadata: &[GGUFMetadata],
    key: &str,
) -> WCResult<Vec<String>> {
    match gguf_metadata_value(metadata, key) {
        Some(GGUFMetadataValue::Array(array))
            if array.value_type == GGUfMetadataValueType::String =>
        {
            array
                .value
                .iter()
                .map(|entry| match entry {
                    GGUFMetadataValue::String(value) => Ok(value.clone()),
                    other => Err(WCError::External(crate::alloc::format!(
                        "GGUF metadata key {key:?} contains non-string array value {other:?}"
                    ))),
                })
                .collect()
        }
        Some(GGUFMetadataValue::Array(array)) => Err(WCError::External(crate::alloc::format!(
            "GGUF metadata key {key:?} must be a string array, found array of {:?}",
            array.value_type
        ))),
        Some(other) => Err(WCError::External(crate::alloc::format!(
            "GGUF metadata key {key:?} must be a string array, found {other:?}"
        ))),
        None => Err(WCError::External(crate::alloc::format!(
            "GGUF metadata is missing required key {key:?}"
        ))),
    }
}

fn gguf_optional_i32_array(
    metadata: &[GGUFMetadata],
    key: &str,
) -> WCResult<Option<Vec<i32>>> {
    match gguf_metadata_value(metadata, key) {
        Some(GGUFMetadataValue::Array(array))
            if matches!(
                array.value_type,
                GGUfMetadataValueType::Int32 | GGUfMetadataValueType::Uint32
            ) =>
        {
            array
                .value
                .iter()
                .map(|entry| match entry {
                    GGUFMetadataValue::Int32(value) => Ok(*value),
                    GGUFMetadataValue::Uint32(value) if *value <= i32::MAX as u32 => {
                        Ok(*value as i32)
                    }
                    other => Err(WCError::External(crate::alloc::format!(
                        "GGUF metadata key {key:?} contains non-int32 array value {other:?}"
                    ))),
                })
                .collect::<WCResult<Vec<_>>>()
                .map(Some)
        }
        Some(GGUFMetadataValue::Array(array)) => Err(WCError::External(crate::alloc::format!(
            "GGUF metadata key {key:?} must be an int32 array, found array of {:?}",
            array.value_type
        ))),
        Some(other) => Err(WCError::External(crate::alloc::format!(
            "GGUF metadata key {key:?} must be an int32 array, found {other:?}"
        ))),
        None => Ok(None),
    }
}

fn parse_serialized_merges<I>(entries: I) -> WCResult<Vec<(String, String)>>
where
    I: IntoIterator<Item = String>,
{
    entries
        .into_iter()
        .map(|entry| {
            entry
                .split_once(' ')
                .map(|(left, right)| (left.to_string(), right.to_string()))
                .ok_or_else(|| {
                    WCError::External(crate::alloc::format!(
                        "invalid legacy merge entry {entry:?}"
                    ))
                })
        })
        .collect()
}

fn gguf_token_text(
    tokens: &[String],
    key: &str,
    id: u32,
) -> WCResult<String> {
    tokens.get(id as usize).cloned().ok_or_else(|| {
        WCError::External(crate::alloc::format!(
            "GGUF metadata key {key:?} references out-of-range token id {id}"
        ))
    })
}

fn gguf_special_tokens(
    tokens: &[String],
    metadata: &[GGUFMetadata],
    token_types: Option<&[i32]>,
) -> WCResult<Vec<(u32, String)>> {
    let mut special_ids: WCHashSet<u32> = WCHashSet::default();

    if let Some(token_types) = token_types {
        for (id, token_type) in token_types.iter().enumerate() {
            if matches!(
                *token_type,
                GGUF_TOKEN_TYPE_UNKNOWN | GGUF_TOKEN_TYPE_CONTROL | GGUF_TOKEN_TYPE_USER_DEFINED
            ) {
                special_ids.insert(id as u32);
            }
        }
    }

    for key in GGUF_SPECIAL_TOKEN_ID_KEYS {
        if let Some(id) = gguf_optional_u32(metadata, key)? {
            special_ids.insert(id);
        }
    }

    let mut special_entries = special_ids
        .into_iter()
        .map(|id| gguf_token_text(tokens, "tokenizer special token", id).map(|token| (id, token)))
        .collect::<WCResult<Vec<_>>>()?;
    special_entries.sort_unstable_by_key(|(id, _)| *id);
    Ok(special_entries)
}

fn gguf_unk_token(
    tokens: &[String],
    metadata: &[GGUFMetadata],
    token_types: Option<&[i32]>,
) -> WCResult<Option<String>> {
    if let Some(id) = gguf_optional_u32(metadata, "tokenizer.ggml.unk_token_id")? {
        return gguf_token_text(tokens, "tokenizer.ggml.unk_token_id", id).map(Some);
    }

    if let Some(token_types) = token_types
        && let Some((id, _)) = token_types
            .iter()
            .enumerate()
            .find(|(_, token_type)| **token_type == GGUF_TOKEN_TYPE_UNKNOWN)
    {
        return Ok(Some(tokens[id].clone()));
    }

    Ok(None)
}

fn gguf_bpe_preset(
    model: &str,
    pre: Option<&str>,
) -> WCResult<GGUFBpePreset> {
    let family = pre.unwrap_or(model);

    let preset = match family {
        "gpt2" | "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "gigachat" | "jina-v2-es"
        | "jina-v2-de" | "a.x-4.0" | "mellum" | "modern-bert" => GGUFBpePreset {
            pattern: OA_GPT2_PATTERN.into(),
            input_normalizer: None,
            ignore_merges: false,
        },
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "stablelm2" | "hunyuan"
        | "solar-open" | "megrez" | "grok-2" => GGUFBpePreset {
            pattern: QWEN2_PATTERN.into(),
            input_normalizer: None,
            ignore_merges: false,
        },
        "qwen35" => GGUFBpePreset {
            pattern: QWEN35_PATTERN.into(),
            input_normalizer: None,
            ignore_merges: false,
        },
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral" | "midm-2.0"
        | "lfm2" | "jina-v5-nano" | "dbrx" | "smaug-bpe" => GGUFBpePreset {
            pattern: OA_CL100K_BASE_PATTERN.into(),
            input_normalizer: None,
            ignore_merges: true,
        },
        "gemma4" | "sarvam-moe" => GGUFBpePreset {
            pattern: GEMMA4_PATTERN.into(),
            input_normalizer: Some(TextNormalizer::Replace {
                pattern: " ".into(),
                replacement: "▁".into(),
            }),
            ignore_merges: false,
        },
        _ => {
            return Err(WCError::NotImplemented(crate::alloc::format!(
                "unsupported GGUF BPE tokenizer family model={model:?} pre={pre:?}"
            )));
        }
    };

    Ok(preset)
}

fn vocab_from_gguf_metadata(metadata: &[GGUFMetadata]) -> WCResult<Arc<UnifiedTokenVocab<u32>>> {
    let mut embedded_hf_error = None;

    if let Some(tokenizer_json) = gguf_optional_string(metadata, GGUF_TOKENIZER_HF_JSON_KEY)? {
        match Tokenizer::from_bytes(tokenizer_json.as_bytes()) {
            Ok(tokenizer) => return vocab_from_hf_tokenizer(&tokenizer),
            Err(error) => embedded_hf_error = Some(error.to_string()),
        }
    }

    let fallback_result = (|| {
        let model = gguf_required_string(metadata, GGUF_TOKENIZER_MODEL_KEY)?;
        let pre = gguf_optional_string(metadata, GGUF_TOKENIZER_PRE_KEY)?;
        let preset = gguf_bpe_preset(&model, pre.as_deref())?;

        let tokens = gguf_required_string_array(metadata, GGUF_TOKENIZER_TOKENS_KEY)?;
        let token_types = gguf_optional_i32_array(metadata, GGUF_TOKENIZER_TOKEN_TYPE_KEY)?;

        if let Some(token_types) = token_types.as_ref()
            && token_types.len() != tokens.len()
        {
            return Err(WCError::External(crate::alloc::format!(
                "GGUF metadata key {:?} has {} entries, but {:?} has {} entries",
                GGUF_TOKENIZER_TOKEN_TYPE_KEY,
                token_types.len(),
                GGUF_TOKENIZER_TOKENS_KEY,
                tokens.len()
            )));
        }

        let merges = parse_serialized_merges(gguf_required_string_array(
            metadata,
            GGUF_TOKENIZER_MERGES_KEY,
        )?)?;

        let special_tokens = gguf_special_tokens(&tokens, metadata, token_types.as_deref())?;
        let unk_token = gguf_unk_token(&tokens, metadata, token_types.as_deref())?;

        let token_count = tokens.len();
        let hf_vocab = tokens
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token, id as u32))
            .collect::<std::collections::HashMap<_, _>>();

        if hf_vocab.len() != token_count {
            return Err(WCError::External(
                "GGUF tokenizer contains duplicate token strings".into(),
            ));
        }

        build_vocab_from_serialized_bpe(SerializedBpeConfig {
            pattern: preset.pattern,
            input_normalizer: preset.input_normalizer,
            hf_vocab,
            merges,
            special_tokens,
            continuing_subword_prefix: None,
            ignore_merges: preset.ignore_merges,
            unk_token,
        })
    })();

    match (embedded_hf_error, fallback_result) {
        (_, Ok(vocab)) => Ok(vocab),
        (Some(embedded_hf_error), Err(fallback_error)) => {
            Err(WCError::External(crate::alloc::format!(
                "failed to load embedded tokenizer.huggingface.json: {embedded_hf_error}; fallback GGUF tokenizer extraction failed: {fallback_error}"
            )))
        }
        (None, Err(fallback_error)) => Err(fallback_error),
    }
}

/// Attempt to convert a `HuggingFace` tokenizer to a `WordChipper` vocabulary.
pub fn vocab_from_hf_tokenizer(tok: &Tokenizer) -> WCResult<Arc<UnifiedTokenVocab<u32>>> {
    let input_normalizer = extract_normalizer(tok.get_normalizer())?;
    let pattern = extract_pattern(tok.get_pre_tokenizer(), input_normalizer.as_ref())?;

    let BPE(bpe) = tok.get_model() else {
        return Err(WCError::External(
            "Tokenizer is not BPE compatible".to_string(),
        ));
    };

    let decoder = tok.get_added_tokens_decoder();
    let mut special_tokens = Vec::with_capacity(decoder.len());
    for (t, at) in decoder.iter() {
        let special_content = if at.normalized {
            normalize_input(input_normalizer.as_ref(), &at.content).into_owned()
        } else {
            let normalized = normalize_input(input_normalizer.as_ref(), &at.content);
            if normalized.as_ref() != at.content {
                return Err(WCError::External(crate::alloc::format!(
                    "unsupported non-normalized special token under text normalizer: {:?}",
                    at.content
                )));
            }
            at.content.clone()
        };

        special_tokens.push((*t, special_content));
    }

    build_vocab_from_serialized_bpe(SerializedBpeConfig {
        pattern,
        input_normalizer,
        hf_vocab: bpe.get_vocab(),
        merges: extract_serialized_merges(bpe)?,
        special_tokens,
        continuing_subword_prefix: bpe.continuing_subword_prefix.clone(),
        ignore_merges: bpe.ignore_merges,
        unk_token: bpe.unk_token.clone(),
    })
}

/// Load a `WordChipper` vocabulary from raw GGUF bytes.
pub fn vocab_from_gguf_bytes(bytes: &[u8]) -> WCResult<Arc<UnifiedTokenVocab<u32>>> {
    let file = GGUFFile::read(bytes)
        .map_err(|error| {
            WCError::External(crate::alloc::format!("failed to parse GGUF file: {error}"))
        })?
        .ok_or_else(|| WCError::External("incomplete GGUF file".into()))?;

    vocab_from_gguf_metadata(&file.header.metadata)
}

/// Load a `WordChipper` vocabulary from a local GGUF file.
pub fn vocab_from_gguf_file<P: AsRef<Path>>(path: P) -> WCResult<Arc<UnifiedTokenVocab<u32>>> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|error| {
        WCError::External(crate::alloc::format!(
            "failed to read GGUF file {}: {error}",
            path.display()
        ))
    })?;

    vocab_from_gguf_bytes(&bytes)
}

fn normalize_gguf_query(query: &VocabQuery) -> WCResult<VocabQuery> {
    if let Some(schema) = query.schema()
        && schema != "gguf"
    {
        return Err(WCError::ResourceNotFound(query.to_string()));
    }

    let raw_path = match query.path() {
        Some(path) => crate::alloc::format!("{path}/{}", query.name()),
        None => query.name().to_string(),
    };

    let normalized: VocabQuery = raw_path.parse()?;
    if normalized.schema() != Some("gguf") {
        return Err(WCError::ResourceNotFound(query.to_string()));
    }

    Ok(normalized)
}

fn resolve_gguf_description(query: &VocabQuery) -> WCResult<VocabDescription> {
    let normalized = normalize_gguf_query(query)?;
    let path = normalized.clone().with_schema(None).to_string();
    if !Path::new(&path).is_file() {
        return Err(WCError::ResourceNotFound(query.to_string()));
    }

    let context = normalized.to_context();
    Ok(VocabDescription::new(
        normalized,
        &context,
        "Local GGUF tokenizer file",
    ))
}

pub struct GGUFVocabProvider {}

inventory::submit! {
    VocabProviderInventoryHook::new(|| Arc::new(GGUFVocabProvider{}))
}

impl VocabProvider for GGUFVocabProvider {
    fn name(&self) -> String {
        "gguf".to_string()
    }

    fn description(&self) -> String {
        "Local GGUF tokenizer files".to_string()
    }

    fn list_vocabs(&self) -> Vec<VocabDescription> {
        vec![]
    }

    fn resolve_vocab(
        &self,
        query: &VocabQuery,
    ) -> WCResult<VocabDescription> {
        resolve_gguf_description(query)
    }

    fn load_vocab(
        &self,
        query: &VocabQuery,
        _loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>> {
        let descr = resolve_gguf_description(query)?;
        let path = descr.id().clone().with_schema(None).to_string();
        let vocab = vocab_from_gguf_file(&path)?;

        Ok(LabeledVocab::new(descr, vocab))
    }
}

pub struct HFVocabProvider {}

inventory::submit! {
    VocabProviderInventoryHook::new(|| Arc::new(HFVocabProvider{}))
}

impl VocabProvider for HFVocabProvider {
    fn name(&self) -> String {
        "hf".to_string()
    }

    fn description(&self) -> String {
        "HuggingFace vocabularies".to_string()
    }

    fn list_vocabs(&self) -> Vec<VocabDescription> {
        vec![]
    }

    fn load_vocab(
        &self,
        query: &VocabQuery,
        _loader: &mut dyn ResourceLoader,
    ) -> WCResult<LabeledVocab<u32>> {
        if let Some(schema) = query.schema()
            && schema != "hf"
        {
            return Err(WCError::ResourceNotFound(query.to_string()));
        }

        if normalize_gguf_query(query).is_ok() {
            return Err(WCError::ResourceNotFound(query.to_string()));
        }

        match Tokenizer::from_pretrained(query.clone().with_schema(None).to_string(), None) {
            Ok(tok) => {
                let vocab = vocab_from_hf_tokenizer(&tok)?;

                let mut context = vec!["hf"];
                if query.path().is_some() {
                    context.push(query.path().unwrap());
                }
                context.push(query.name());

                let id = query.clone().with_schema(Some("hf"));
                let context = id.to_context();

                let descr: VocabDescription =
                    VocabDescription::new(id, &context, "Model loaded from hf");

                Ok(LabeledVocab::new(descr, vocab))
            }
            Err(_) => Err(WCError::ResourceNotFound(query.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use gguf::{
        GGUFMetadata,
        GGUFMetadataArrayValue,
        GGUFMetadataValue,
        GGUfMetadataValueType,
    };
    use tokenizers::{
        models::bpe::{
            BPE as HfBpe,
            Vocab as HfVocab,
        },
        normalizers::{
            Lowercase,
            NFC,
            NormalizerWrapper,
            Replace,
            Sequence,
        },
        tokenizer::SplitDelimiterBehavior,
    };

    use super::*;
    use crate::{
        TokenEncoder,
        TokenizerOptions,
        load_vocab,
        resolve_vocab,
    };

    struct NoopResourceLoader;

    impl ResourceLoader for NoopResourceLoader {
        fn load_resource_path(
            &mut self,
            _resource: &crate::support::resources::KeyedResource,
        ) -> WCResult<PathBuf> {
            unreachable!("GGUF provider should not request remote resources")
        }
    }

    #[test]
    fn test_extract_normalizer_maps_nfc() {
        assert_eq!(
            extract_normalizer(Some(&NormalizerWrapper::NFC(NFC))).unwrap(),
            Some(TextNormalizer::NFC)
        );
    }

    #[test]
    fn test_extract_normalizer_maps_sequence() {
        let sequence = Sequence::new(vec![NormalizerWrapper::NFC(NFC)]);

        assert_eq!(
            extract_normalizer(Some(&NormalizerWrapper::Sequence(sequence))).unwrap(),
            Some(TextNormalizer::Sequence(vec![TextNormalizer::NFC]))
        );
    }

    #[test]
    fn test_extract_normalizer_rejects_unsupported_wrapper() {
        let error = extract_normalizer(Some(&NormalizerWrapper::Lowercase(Lowercase))).unwrap_err();

        assert!(matches!(error, WCError::External(_)));
    }

    #[test]
    fn test_extract_normalizer_maps_replace_string() {
        let replace = Replace::new(" ", "▁").unwrap();

        assert_eq!(
            extract_normalizer(Some(&NormalizerWrapper::Replace(replace))).unwrap(),
            Some(TextNormalizer::Replace {
                pattern: " ".into(),
                replacement: "▁".into(),
            })
        );
    }

    #[test]
    fn test_extract_pattern_maps_metaspace_split() {
        let split = tokenizers::pre_tokenizers::split::Split::new(
            " ",
            SplitDelimiterBehavior::MergedWithPrevious,
            false,
        )
        .unwrap();

        let pattern = extract_pattern(
            Some(&PreTokenizerWrapper::Split(split)),
            Some(&TextNormalizer::Replace {
                pattern: " ".into(),
                replacement: "▁".into(),
            }),
        )
        .unwrap();

        assert_eq!(pattern.as_str(), HF_WHOLE_SEGMENT_PATTERN);
    }

    #[test]
    fn test_extract_pattern_rejects_unpaired_string_split() {
        let split = tokenizers::pre_tokenizers::split::Split::new(
            " ",
            SplitDelimiterBehavior::MergedWithPrevious,
            false,
        )
        .unwrap();

        let error = extract_pattern(Some(&PreTokenizerWrapper::Split(split)), None).unwrap_err();

        assert!(matches!(error, WCError::External(_)));
    }

    #[test]
    fn test_extract_byte_tokens_maps_byte_fallback_vocab() {
        let hf_vocab = (0u8..=255)
            .map(|byte| (byte_fallback_token(byte), byte as u32))
            .collect::<std::collections::HashMap<_, _>>();

        let tokens = extract_byte_tokens(&hf_vocab, &HFBpeEncoding::ByteFallback).unwrap();

        assert_eq!(tokens.len(), 256);
        assert_eq!(tokens[0], 0);
        assert_eq!(tokens[255], 255);
    }

    #[test]
    fn test_extract_byte_tokens_prefers_single_byte_chars_when_available() {
        let mut hf_vocab = (0u8..=255)
            .map(|byte| (byte_fallback_token(byte), byte as u32))
            .collect::<std::collections::HashMap<_, _>>();
        hf_vocab.insert("a".into(), 9999);

        let tokens = extract_byte_tokens(&hf_vocab, &HFBpeEncoding::ByteFallback).unwrap();

        assert_eq!(tokens[b'a' as usize], 9999);
    }

    #[test]
    fn test_vocab_from_hf_tokenizer_preserves_scalar_seeding_and_merge_rank() {
        let mut hf_vocab = (0u8..=255)
            .map(|byte| (byte_fallback_token(byte), byte as u32))
            .collect::<std::collections::HashMap<_, _>>();
        hf_vocab.insert("a".into(), 1000);
        hf_vocab.insert("b".into(), 1001);
        hf_vocab.insert("ab".into(), 2000);
        hf_vocab.insert("め".into(), 3000);
        hf_vocab.insert("めa".into(), 1500);

        let bpe = HfBpe::builder()
            .vocab_and_merges(
                hf_vocab.clone().into_iter().collect::<HfVocab>(),
                vec![
                    ("a".to_string(), "b".to_string()),
                    ("め".to_string(), "a".to_string()),
                ],
            )
            .byte_fallback(true)
            .build()
            .unwrap();

        let mut tok = Tokenizer::new(bpe);
        tok.with_normalizer(Some(NormalizerWrapper::Replace(
            Replace::new(" ", "▁").unwrap(),
        )));
        tok.with_pre_tokenizer(Some(PreTokenizerWrapper::Split(
            tokenizers::pre_tokenizers::split::Split::new(
                " ",
                SplitDelimiterBehavior::MergedWithPrevious,
                false,
            )
            .unwrap(),
        )));

        let vocab = vocab_from_hf_tokenizer(&tok).unwrap();
        assert_eq!(vocab.lookup_pair_merge(&(1000, 1001)), Some((0, 2000)));
        assert_eq!(vocab.lookup_pair_merge(&(3000, 1000)), Some((1, 1500)));
        assert!(!vocab.direct_word_lookup());

        let tokenizer = TokenizerOptions::default().build(vocab.clone());
        let wc_tokens = tokenizer.try_encode("めab", None).unwrap();
        let hf_tokens = tok.encode("めab", true).unwrap().get_ids().to_vec();

        assert_eq!(wc_tokens, hf_tokens);
        assert_eq!(wc_tokens, vec![3000, 2000]);
    }

    fn gguf_string_entry(
        key: &str,
        value: impl Into<String>,
    ) -> GGUFMetadata {
        GGUFMetadata {
            key: key.into(),
            value_type: GGUfMetadataValueType::String,
            value: GGUFMetadataValue::String(value.into()),
        }
    }

    fn gguf_u32_entry(
        key: &str,
        value: u32,
    ) -> GGUFMetadata {
        GGUFMetadata {
            key: key.into(),
            value_type: GGUfMetadataValueType::Uint32,
            value: GGUFMetadataValue::Uint32(value),
        }
    }

    fn gguf_string_array_entry(
        key: &str,
        values: Vec<String>,
    ) -> GGUFMetadata {
        let len = values.len() as u64;
        GGUFMetadata {
            key: key.into(),
            value_type: GGUfMetadataValueType::Array,
            value: GGUFMetadataValue::Array(GGUFMetadataArrayValue {
                value_type: GGUfMetadataValueType::String,
                len,
                value: values.into_iter().map(GGUFMetadataValue::String).collect(),
            }),
        }
    }

    fn gguf_i32_array_entry(
        key: &str,
        values: Vec<i32>,
    ) -> GGUFMetadata {
        let len = values.len() as u64;
        GGUFMetadata {
            key: key.into(),
            value_type: GGUfMetadataValueType::Array,
            value: GGUFMetadataValue::Array(GGUFMetadataArrayValue {
                value_type: GGUfMetadataValueType::Int32,
                len,
                value: values.into_iter().map(GGUFMetadataValue::Int32).collect(),
            }),
        }
    }

    fn push_u32(
        bytes: &mut Vec<u8>,
        value: u32,
    ) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_i32(
        bytes: &mut Vec<u8>,
        value: i32,
    ) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(
        bytes: &mut Vec<u8>,
        value: u64,
    ) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(
        bytes: &mut Vec<u8>,
        value: &str,
    ) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_metadata_scalar(
        bytes: &mut Vec<u8>,
        value: &GGUFMetadataValue,
    ) {
        match value {
            GGUFMetadataValue::Uint32(value) => push_u32(bytes, *value),
            GGUFMetadataValue::Int32(value) => push_i32(bytes, *value),
            GGUFMetadataValue::String(value) => push_string(bytes, value),
            GGUFMetadataValue::Array(_) => panic!("nested GGUF arrays are not supported in tests"),
            other => panic!("unsupported GGUF metadata test value {other:?}"),
        }
    }

    fn push_metadata_value(
        bytes: &mut Vec<u8>,
        value: &GGUFMetadataValue,
    ) {
        match value {
            GGUFMetadataValue::Array(array) => {
                push_u32(bytes, array.value_type as u32);
                push_u64(bytes, array.len);
                for entry in &array.value {
                    push_metadata_scalar(bytes, entry);
                }
            }
            _ => push_metadata_scalar(bytes, value),
        }
    }

    fn build_test_gguf(metadata: &[GGUFMetadata]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, metadata.len() as u64);

        for entry in metadata {
            push_string(&mut bytes, &entry.key);
            push_u32(&mut bytes, entry.value_type as u32);
            push_metadata_value(&mut bytes, &entry.value);
        }

        bytes
    }

    #[test]
    fn test_vocab_from_gguf_bytes_prefers_embedded_hf_tokenizer_json() {
        let mut hf_vocab = (0u8..=255)
            .map(|byte| (byte_fallback_token(byte), byte as u32))
            .collect::<std::collections::HashMap<_, _>>();
        hf_vocab.insert("a".into(), 1000);
        hf_vocab.insert("b".into(), 1001);
        hf_vocab.insert("ab".into(), 2000);
        hf_vocab.insert("め".into(), 3000);
        hf_vocab.insert("めa".into(), 1500);

        let bpe = HfBpe::builder()
            .vocab_and_merges(
                hf_vocab.clone().into_iter().collect::<HfVocab>(),
                vec![
                    ("a".to_string(), "b".to_string()),
                    ("め".to_string(), "a".to_string()),
                ],
            )
            .byte_fallback(true)
            .build()
            .unwrap();

        let mut tok = Tokenizer::new(bpe);
        tok.with_normalizer(Some(NormalizerWrapper::Replace(
            Replace::new(" ", "▁").unwrap(),
        )));
        tok.with_pre_tokenizer(Some(PreTokenizerWrapper::Split(
            tokenizers::pre_tokenizers::split::Split::new(
                " ",
                SplitDelimiterBehavior::MergedWithPrevious,
                false,
            )
            .unwrap(),
        )));

        let metadata = vec![gguf_string_entry(
            GGUF_TOKENIZER_HF_JSON_KEY,
            tok.to_string(false).unwrap(),
        )];

        let vocab = vocab_from_gguf_bytes(&build_test_gguf(&metadata)).unwrap();
        assert_eq!(vocab.lookup_pair_merge(&(1000, 1001)), Some((0, 2000)));

        let tokenizer = TokenizerOptions::default().build(vocab.clone());
        let wc_tokens = tokenizer.try_encode("めab", None).unwrap();
        let hf_tokens = tok.encode("めab", true).unwrap().get_ids().to_vec();

        assert_eq!(wc_tokens, hf_tokens);
        assert_eq!(wc_tokens, vec![3000, 2000]);
    }

    #[test]
    fn test_vocab_from_gguf_bytes_rebuilds_bpe_from_ggml_metadata() {
        let mut tokens = (0u8..=255).map(byte_fallback_token).collect::<Vec<_>>();
        tokens.push("a".into());
        tokens.push("b".into());
        tokens.push("ab".into());
        tokens.push("<bos>".into());

        let mut token_types = vec![6; 256];
        token_types.extend([1, 1, 1, GGUF_TOKEN_TYPE_CONTROL]);

        let metadata = vec![
            gguf_string_entry(GGUF_TOKENIZER_MODEL_KEY, "gpt2"),
            gguf_string_entry(GGUF_TOKENIZER_PRE_KEY, "qwen35"),
            gguf_string_array_entry(GGUF_TOKENIZER_TOKENS_KEY, tokens),
            gguf_i32_array_entry(GGUF_TOKENIZER_TOKEN_TYPE_KEY, token_types),
            gguf_string_array_entry(GGUF_TOKENIZER_MERGES_KEY, vec!["a b".into()]),
            gguf_u32_entry("tokenizer.ggml.bos_token_id", 259),
        ];

        let vocab = vocab_from_gguf_bytes(&build_test_gguf(&metadata)).unwrap();
        assert_eq!(vocab.lookup_pair_merge(&(256, 257)), Some((0, 258)));
        assert_eq!(vocab.special_vocab().lookup_token(b"<bos>"), Some(259));
        assert!(!vocab.direct_word_lookup());

        let tokenizer = TokenizerOptions::default().build(vocab.clone());
        assert_eq!(tokenizer.try_encode("ab", None).unwrap(), vec![258]);
    }

    #[test]
    fn test_vocab_from_gguf_file_reads_local_file() {
        let mut tokens = (0u8..=255).map(byte_fallback_token).collect::<Vec<_>>();
        tokens.push("a".into());
        tokens.push("b".into());
        tokens.push("ab".into());

        let mut token_types = vec![6; 256];
        token_types.extend([1, 1, 1]);

        let metadata = vec![
            gguf_string_entry(GGUF_TOKENIZER_MODEL_KEY, "gpt2"),
            gguf_string_entry(GGUF_TOKENIZER_PRE_KEY, "qwen35"),
            gguf_string_array_entry(GGUF_TOKENIZER_TOKENS_KEY, tokens),
            gguf_i32_array_entry(GGUF_TOKENIZER_TOKEN_TYPE_KEY, token_types),
            gguf_string_array_entry(GGUF_TOKENIZER_MERGES_KEY, vec!["a b".into()]),
        ];

        let dir = tempdir::TempDir::new("wordchipper-gguf").unwrap();
        let path = dir.path().join("tokenizer.gguf");
        std::fs::write(&path, build_test_gguf(&metadata)).unwrap();

        let vocab = vocab_from_gguf_file(&path).unwrap();
        assert_eq!(vocab.lookup_pair_merge(&(256, 257)), Some((0, 258)));
    }

    #[test]
    fn test_load_vocab_accepts_local_gguf_path() {
        let mut tokens = (0u8..=255).map(byte_fallback_token).collect::<Vec<_>>();
        tokens.push("a".into());
        tokens.push("b".into());
        tokens.push("ab".into());

        let mut token_types = vec![6; 256];
        token_types.extend([1, 1, 1]);

        let metadata = vec![
            gguf_string_entry(GGUF_TOKENIZER_MODEL_KEY, "gpt2"),
            gguf_string_entry(GGUF_TOKENIZER_PRE_KEY, "qwen35"),
            gguf_string_array_entry(GGUF_TOKENIZER_TOKENS_KEY, tokens),
            gguf_i32_array_entry(GGUF_TOKENIZER_TOKEN_TYPE_KEY, token_types),
            gguf_string_array_entry(GGUF_TOKENIZER_MERGES_KEY, vec!["a b".into()]),
        ];

        let dir = tempdir::TempDir::new("wordchipper-gguf-provider").unwrap();
        let path = dir.path().join("tokenizer.gguf");
        std::fs::write(&path, build_test_gguf(&metadata)).unwrap();

        let query = path.display().to_string();
        let resolved = resolve_vocab(&query).unwrap();
        assert_eq!(resolved.id().schema(), Some("gguf"));
        assert_eq!(resolved.id().name(), "tokenizer.gguf");

        let mut loader = NoopResourceLoader;
        let loaded = load_vocab(&query, &mut loader).unwrap();
        assert_eq!(loaded.description().id().schema(), Some("gguf"));
        assert_eq!(
            loaded.vocab().lookup_pair_merge(&(256, 257)),
            Some((0, 258))
        );
    }
}
