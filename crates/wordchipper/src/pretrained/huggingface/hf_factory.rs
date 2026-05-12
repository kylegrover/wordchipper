use tokenizers::{
    ModelWrapper::BPE,
    PreTokenizerWrapper,
    PreTokenizerWrapper::{
        ByteLevel,
        Sequence,
        Split,
    },
    tokenizer::SplitDelimiterBehavior,
    tokenizer::NormalizerWrapper,
    pre_tokenizers::split::SplitPattern,
    tokenizer::Tokenizer,
};

use serde_json::Value;

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
        openai::OA_GPT2_PATTERN,
    },
    spanners::TextSpanningConfig,
    support::{
        normalization::TextNormalizer,
        regex::RegexPattern,
        resources::ResourceLoader,
    },
    vocab::{
        ByteMapVocab,
        SpanMapVocab,
        SpanTokenMap,
    },
};

const HF_WHOLE_SEGMENT_PATTERN: &str = r"[\s\S]+";

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
                    ) => Ok(HF_WHOLE_SEGMENT_PATTERN.into()),
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
    replace: &tokenizers::normalizers::Replace,
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
    hf_vocab: &std::collections::HashMap<String, u32>,
    byte_fallback: bool,
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

    if byte_fallback {
        return Err(WCError::External(
            "BPE enables byte_fallback but vocab is missing one or more <0xXX> byte tokens"
                .into(),
        ));
    }

    Err(WCError::External(
        "unsupported huggingface BPE byte encoding".into(),
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
            WCError::External(crate::alloc::format!(
                "missing byte token for 0x{byte:02x}"
            ))
        })
}

/// Attempt to convert a `HuggingFace` tokenizer to a `WordChipper` vocabulary.
pub fn vocab_from_hf_tokenizer(tok: &Tokenizer) -> WCResult<Arc<UnifiedTokenVocab<u32>>> {
    type T = u32;

    let input_normalizer = extract_normalizer(tok.get_normalizer())?;
    let pattern = extract_pattern(tok.get_pre_tokenizer(), input_normalizer.as_ref())?;
    let mut span_config: TextSpanningConfig<T> = TextSpanningConfig::from_pattern(pattern);

    let BPE(bpe) = tok.get_model() else {
        return Err(WCError::External(
            "Tokenizer is not BPE compatible".to_string(),
        ));
    };

    let hf_vocab = bpe.get_vocab();
    let encoding = extract_bpe_encoding(&hf_vocab, bpe.byte_fallback)?;

    /*
    println!(
        "Debug: {:?}",
        hf_vocab.iter().find(|(_, id)| **id == 157513)
    );
     */

    let mut special_tokens: WCHashSet<T> = Default::default();

    let decoder = tok.get_added_tokens_decoder();
    /*
    println!("Debug: {:#?}", decoder);
     */

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

        span_config
            .specials_mut()
            .add_str_word(&special_content, *t);
        special_tokens.insert(*t);
    }

    // Span map: decode every non-special vocab string back to bytes.
    let mut span_map: SpanTokenMap<T> = SpanTokenMap::default();
    for (s, id) in &hf_vocab {
        if special_tokens.contains(id) {
            continue;
        }

        if let Some(bytes) = token_to_bytes(s, &encoding)? {
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

    let expected_len = span_vocab.len() + span_config.specials().len();

    let vocab = UnifiedTokenVocab::from_span_vocab(span_config, span_vocab)?;
    let vocab = if let Some(normalizer) = input_normalizer {
        vocab.with_input_normalizer(normalizer)
    } else {
        vocab
    };
    let vocab: Arc<UnifiedTokenVocab<T>> = Arc::new(vocab);

    // TODO: should `vocab.len()` include the special len()?
    if vocab.len() + vocab.special_vocab().len() != expected_len {
        return Err(WCError::External(format!(
            "Expected {} tokens, got {}",
            expected_len,
            vocab.len()
        )));
    }

    Ok(vocab)
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
    use super::*;
    use tokenizers::normalizers::Replace;
    use tokenizers::normalizers::{
        Lowercase,
        NFC,
        NormalizerWrapper,
        Sequence,
    };
    use tokenizers::tokenizer::SplitDelimiterBehavior;

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
}
