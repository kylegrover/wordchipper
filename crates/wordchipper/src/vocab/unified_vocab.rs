//! # Unified Token Vocabulary

use crate::{
    Pair,
    TokenType,
    WCError,
    WCHashSet,
    WCResult,
    alloc::{
        borrow::Cow,
        vec::Vec,
    },
    spanners::TextSpanningConfig,
    support::{
        normalization::TextNormalizer,
        strings::string_from_utf8_lossy,
    },
    vocab::{
        ByteMapVocab,
        PairMapVocab,
        SpanMapVocab,
        SpanTokenMap,
        SpecialVocab,
        TokenSpanMap,
        VocabIndex,
    },
};

#[derive(Clone, PartialEq)]
enum SeedStrategy<T: TokenType> {
    Bytes(core::marker::PhantomData<T>),
    #[cfg(any(feature = "huggingface", feature = "download", test))]
    UnicodeScalars {
        scalar_tokens: SpanTokenMap<T>,
        byte_fallback_tokens: Vec<T>,
        unk_token: Option<T>,
    },
}

/// A unified vocabulary structure for BPE tokenization that provides coherent
/// views of vocabulary components through multiple mapping interfaces.
///
/// # Overview
///
/// [`UnifiedTokenVocab<T>`] is the primary user-facing vocabulary type, generic
/// over `T: TokenType` (unsigned integer types like `u16`, `u32`, `u64`). It
/// provides several complementary views of the same vocabulary data:
///
/// * [`ByteMapVocab`] - Bijective `u8 ↔ T` byte-to-token mapping (256 entries)
/// * [`SpanMapVocab`] - Dictionary mapping `Vec<u8> → T` for multi-byte
///   sequences
/// * [`PairMapVocab`] - BPE merge rules mapping `(T, T) → T` for token pairs
/// * [`TextSpanningConfig`] - Text segmentation rules (regex patterns, special
///   tokens)
///
/// # Typical Usage
///
/// Pre-trained vocabulary loaders return `UnifiedTokenVocab<T>`, which can be
/// accessed via `Arc<UnifiedTokenVocab<T>>`:
/// - Converted to different token types via
///   [`to_token_type`](Self::to_token_type)
///
/// # Example
///
/// ```ignore
/// // Load a pre-trained vocabulary
/// let vocab: UnifiedTokenVocab<u32> = load_pretrained_vocab()?;
///
/// // Create a default encoder
/// let encoder = TokenEncoderBuilder::default(vocab.clone());
///
/// // Or configure with custom options
/// let encoder = TokenEncoderBuilder::new(vocab.clone())
///     .parallel(true)
///     .build();
/// ```
#[derive(Clone, PartialEq)]
pub struct UnifiedTokenVocab<T: TokenType> {
    /// Text Spanning Configuration
    spanning: TextSpanningConfig<T>,

    /// Optional text normalizer applied before encoding.
    input_normalizer: Option<TextNormalizer>,

    /// ``{ Vec<u8> -> T }`` vocabulary.
    span_vocab: SpanMapVocab<T>,

    /// ``{ (T, T) -> T }`` vocabulary.
    pair_vocab: PairMapVocab<T>,

    /// Initial tokenization strategy used before BPE merges.
    seed_strategy: SeedStrategy<T>,

    /// Whether word spans may be emitted directly from the span vocab.
    direct_word_lookup: bool,
}

impl<T: TokenType> UnifiedTokenVocab<T> {
    /// Build a new [`UnifiedTokenVocab`] from a [`SpanMapVocab`].
    ///
    /// ## Arguments
    /// * `span_config` - The spanners configuration.
    /// * `span_vocab` - The span map vocabulary.
    ///
    /// ## Returns
    /// A `Result<UnifiedTokenVocab>`, with errors on vocab conflict.
    pub fn from_span_vocab(
        span_config: TextSpanningConfig<T>,
        span_vocab: SpanMapVocab<T>,
    ) -> WCResult<Self> {
        let pair_vocab = span_vocab.to_pair_vocab();
        Self::new(span_config, span_vocab, pair_vocab)
    }

    /// Build a new [`UnifiedTokenVocab`] from a [`PairMapVocab`].
    ///
    /// ## Arguments
    /// * `span_config` - The spanners configuration.
    /// * `pair_vocab` - The pair map vocabulary.
    ///
    /// ## Returns
    /// A `Result<UnifiedTokenVocab>`, with errors on vocab conflict.
    pub fn from_pair_vocab(
        span_config: TextSpanningConfig<T>,
        pair_vocab: PairMapVocab<T>,
    ) -> WCResult<Self> {
        let word_vocab = pair_vocab.span_pairs().collect::<SpanTokenMap<T>>().into();
        Self::from_span_vocab(span_config, word_vocab)
    }

    /// Initialize a [`UnifiedTokenVocab`].
    ///
    /// ## Arguments
    /// * `span_config` - The spanners configuration.
    /// * `word_vocab` - The span map vocabulary.
    /// * `pair_vocab` - The pair map vocabulary.
    ///
    /// ## Returns
    /// A `Result<UnifiedTokenVocab>`, with errors on vocab conflict.
    pub fn new(
        span_config: TextSpanningConfig<T>,
        span_vocab: SpanMapVocab<T>,
        pair_vocab: PairMapVocab<T>,
    ) -> WCResult<Self> {
        if span_vocab.byte_vocab() != pair_vocab.byte_vocab() {
            return Err(WCError::VocabConflict(
                "span vocab and pair vocab have different byte vocabularies".into(),
            ));
        }

        let span_tokens = span_vocab.tokens();
        let primitive_tokens: WCHashSet<T> = pair_vocab.primitive_spans().keys().copied().collect();
        let available_tokens = span_tokens
            .iter()
            .copied()
            .chain(primitive_tokens.iter().copied())
            .collect::<WCHashSet<T>>();
        let pair_tokens = pair_vocab.tokens();
        if !pair_tokens.is_subset(&available_tokens) {
            let missing = pair_tokens
                .difference(&available_tokens)
                .copied()
                .collect::<Vec<_>>();
            return Err(WCError::VocabConflict(
                crate::alloc::format!(
                    "pair vocab contains tokens missing from span vocab: {missing:?}"
                ),
            ));
        }

        for t in span_config.specials().tokens() {
            if span_tokens.contains(&t) {
                let span = span_config.specials().lookup_span(&t).unwrap();
                let special = string_from_utf8_lossy(span.to_vec());
                return Err(WCError::VocabConflict(crate::alloc::format!(
                    "special token \"{special:?}\" -> ({t:?}) found in word vocab"
                )));
            }
        }

        Ok(Self {
            spanning: span_config,
            input_normalizer: None,
            span_vocab,
            pair_vocab,
            seed_strategy: SeedStrategy::Bytes(core::marker::PhantomData),
            direct_word_lookup: true,
        })
    }

    /// Create a copy of this [`UnifiedTokenVocab`] with a different
    /// [`TokenType`].
    ///
    /// This will fail if the maximum token index for the new token type is
    /// exceeded.
    pub fn to_token_type<G: TokenType>(&self) -> WCResult<UnifiedTokenVocab<G>> {
        Ok(UnifiedTokenVocab::<G> {
            spanning: self.spanning.to_token_type::<G>()?,
            input_normalizer: self.input_normalizer.clone(),
            span_vocab: self.span_vocab.to_token_type::<G>()?,
            pair_vocab: self.pair_vocab.to_token_type::<G>()?,
            seed_strategy: match &self.seed_strategy {
                SeedStrategy::Bytes(_) => SeedStrategy::Bytes(core::marker::PhantomData),
                #[cfg(any(feature = "huggingface", feature = "download", test))]
                SeedStrategy::UnicodeScalars {
                    scalar_tokens,
                    byte_fallback_tokens,
                    unk_token,
                } => SeedStrategy::UnicodeScalars {
                    scalar_tokens: scalar_tokens
                        .iter()
                        .map(|(span, &token)| (span.clone(), G::from(token).unwrap()))
                        .collect(),
                    byte_fallback_tokens: byte_fallback_tokens
                        .iter()
                        .map(|&token| G::from(token).unwrap())
                        .collect(),
                    unk_token: unk_token.map(|token| G::from(token).unwrap()),
                },
            },
            direct_word_lookup: self.direct_word_lookup,
        })
    }

    /// Attach an input normalizer used before encoding.
    pub fn with_input_normalizer(
        mut self,
        normalizer: TextNormalizer,
    ) -> Self {
        self.input_normalizer = Some(normalizer);
        self
    }

    #[cfg(any(feature = "huggingface", feature = "download", test))]
    /// Configure Unicode-scalar seeding before BPE merges.
    pub(crate) fn with_unicode_scalar_seeding(
        mut self,
        scalar_tokens: SpanTokenMap<T>,
        byte_fallback_tokens: Vec<T>,
        unk_token: Option<T>,
    ) -> Self {
        self.seed_strategy = SeedStrategy::UnicodeScalars {
            scalar_tokens,
            byte_fallback_tokens,
            unk_token,
        };
        self
    }

    #[cfg(any(feature = "huggingface", feature = "download", test))]
    /// Enable or disable direct whole-span lookup before merge encoding.
    pub(crate) fn with_direct_word_lookup(
        mut self,
        direct_word_lookup: bool,
    ) -> Self {
        self.direct_word_lookup = direct_word_lookup;
        self
    }

    /// Get the [`TextSpanningConfig`].
    pub fn spanning(&self) -> &TextSpanningConfig<T> {
        &self.spanning
    }

    /// Get the optional input normalizer.
    pub fn input_normalizer(&self) -> Option<&TextNormalizer> {
        self.input_normalizer.as_ref()
    }

    /// Normalize text prior to encoding.
    pub fn normalize_text<'a>(
        &self,
        text: &'a str,
    ) -> Cow<'a, str> {
        self.input_normalizer()
            .map(|normalizer| normalizer.normalize(text))
            .unwrap_or_else(|| Cow::Borrowed(text))
    }

    /// Get the `{ (T, T) -> T }` [`PairMapVocab`].
    pub fn pair_vocab(&self) -> &PairMapVocab<T> {
        &self.pair_vocab
    }

    /// Get the `{ Vec<u8> -> T }` [`SpanMapVocab`].
    pub fn span_vocab(&self) -> &SpanMapVocab<T> {
        &self.span_vocab
    }

    /// Get the `{ u8 -> T }` [`ByteMapVocab`].
    pub fn byte_vocab(&self) -> &ByteMapVocab<T> {
        self.span_vocab.byte_vocab()
    }

    /// Return whether word spans may be emitted directly from the span vocab.
    pub(crate) fn direct_word_lookup(&self) -> bool {
        self.direct_word_lookup
    }

    /// Return whether this vocab is compatible with the current backtrack encoder.
    pub(crate) fn supports_backtrack_encoder(&self) -> bool {
        matches!(self.seed_strategy, SeedStrategy::Bytes(_))
            && self.pair_vocab.pair_map().iter().all(|(pair, &token)| {
                self.lookup_pair_merge(pair)
                    .is_some_and(|(rank, merge_token)| {
                        merge_token == token && rank == token.to_u32().unwrap()
                    })
            })
    }

    /// Append the initial seed tokens for a compound span.
    pub(crate) fn append_seed_tokens(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        match &self.seed_strategy {
            SeedStrategy::Bytes(_) => self.byte_vocab().append_tokens(span, tokens),
            #[cfg(any(feature = "huggingface", feature = "download", test))]
            SeedStrategy::UnicodeScalars {
                scalar_tokens,
                byte_fallback_tokens,
                unk_token,
            } => {
                let text = core::str::from_utf8(span).expect("token spans must remain valid UTF-8");
                for ch in text.chars() {
                    let mut buf = [0u8; 4];
                    let bytes = ch.encode_utf8(&mut buf).as_bytes();
                    if let Some(&token) = scalar_tokens.get(bytes) {
                        tokens.push(token);
                    } else if !byte_fallback_tokens.is_empty() {
                        tokens.extend(bytes.iter().map(|&byte| byte_fallback_tokens[byte as usize]));
                    } else if let Some(token) = *unk_token {
                        tokens.push(token);
                    } else {
                        self.byte_vocab().append_tokens(bytes, tokens);
                    }
                }
            }
        }
    }

    /// Get the `{ Vec<u8> -> T }` special token [`SpanMapVocab`].
    pub fn special_vocab(&self) -> &SpecialVocab<T> {
        self.spanning.specials()
    }

    /// Get a mutable view of the [`SpecialVocab`]
    pub fn special_vocab_mut(&mut self) -> &mut SpecialVocab<T> {
        self.spanning.specials_mut()
    }

    /// Compile a unified `{ T -> Vec<u8> }` expansion dictionary.
    ///
    /// This will include the single-byte entries in the byte map,
    /// as well as the entries in the special tokens map.
    pub fn unified_dictionary(&self) -> TokenSpanMap<T> {
        let mut tmp = TokenSpanMap::default();

        self.span_vocab.iter().for_each(|(chunk, &token)| {
            tmp.insert(token, chunk.to_vec());
        });

        for (span, token) in self.pair_vocab.span_pairs() {
            if tmp.contains_key(&token) {
                continue;
            }
            tmp.insert(token, span);
        }

        for (span, t) in self.spanning.specials().span_pairs() {
            tmp.insert(t, span);
        }

        tmp
    }

    /// Looks up a token in the vocabulary using the provided byte slice.
    ///
    /// ## Arguments
    /// * `span` - A byte slice (`&[u8]`) representing the token to be looked up
    ///   in the vocabulary.
    ///
    /// ## Returns
    /// * `Option<T>` - Returns `Some(T)` if the token is found in the
    ///   vocabulary, where `T` is the type of the value associated with the
    ///   token. Returns `None` if the token is not found.
    pub fn lookup_token(
        &self,
        span: &[u8],
    ) -> Option<T> {
        self.span_vocab.lookup_token(span)
    }

    /// Looks up a given pair in the pair vocabulary and retrieves its
    /// associated data, if present.
    ///
    /// ## Arguments
    /// * `pair` - A reference to the `Pair<T>` to be looked up in the pair
    ///   vocabulary.
    ///
    /// ## Returns
    /// * `Option<&T>` - Returns `Some(&T)` if the pair is found in the pair
    ///   vocabulary, otherwise returns `None`.
    pub fn lookup_pair(
        &self,
        pair: &Pair<T>,
    ) -> Option<T> {
        self.pair_vocab.lookup_pair(pair)
    }

    /// Looks up a given pair together with its merge rank.
    pub fn lookup_pair_merge(
        &self,
        pair: &Pair<T>,
    ) -> Option<(u32, T)> {
        self.pair_vocab.lookup_pair_merge(pair)
    }
}

impl<T: TokenType> VocabIndex<T> for UnifiedTokenVocab<T> {
    type Token = T;

    fn len(&self) -> usize {
        self.tokens().len()
    }

    fn tokens(&self) -> WCHashSet<T> {
        self.span_vocab
            .tokens()
            .into_iter()
            .chain(self.pair_vocab.tokens())
            .collect::<WCHashSet<T>>()
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.unified_dictionary()
            .into_iter()
            .map(|(token, span)| (span, token))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use num_traits::FromPrimitive;

    use super::*;
    use crate::{
        spanners::TextSpanningConfig,
        support::normalization::TextNormalizer,
        vocab::{
            PairTokenMap,
            SpanMapVocab,
        },
    };

    #[test]
    fn test_init() {
        type T = u32;
        let mut span_vocab: SpanTokenMap<T> = Default::default();
        span_vocab.insert("at".as_bytes().to_vec(), 300);
        span_vocab.insert("ate".as_bytes().to_vec(), 301);

        let span_vocab: SpanMapVocab<T> = span_vocab.into();

        let seg_config = TextSpanningConfig::from_pattern(r"\w\+");

        let vocab = UnifiedTokenVocab::from_span_vocab(seg_config, span_vocab.clone()).unwrap();
        assert_eq!(vocab.len(), 256 + 2);

        let byte_vocab = vocab.byte_vocab();

        {
            let mut expected: PairTokenMap<T> = Default::default();
            expected.insert(
                (
                    T::from_u8('a' as u8).unwrap(),
                    T::from_u8('t' as u8).unwrap(),
                ),
                300,
            );
            expected.insert((300, T::from_u8('e' as u8).unwrap()), 301);
            let expected: PairMapVocab<T> =
                PairMapVocab::new(byte_vocab.clone(), expected).unwrap();

            assert_eq!(vocab.pair_vocab(), &expected);
        }

        {
            let mut expected: SpanTokenMap<T> = byte_vocab.span_pairs().collect();
            expected.extend(span_vocab.span_pairs());
            let expected: SpanMapVocab<T> = expected.into();

            assert_eq!(vocab.span_vocab(), &expected);
        }

        let mut actual = vocab.span_pairs().collect::<Vec<_>>();
        let mut expected = vocab
            .unified_dictionary()
            .into_iter()
            .map(|(token, span)| (span, token))
            .collect::<Vec<_>>();
        actual.sort();
        expected.sort();

        assert_eq!(actual, expected);

        assert_eq!(vocab.lookup_token("at".as_bytes()), Some(300));
        assert_eq!(vocab.lookup_token("ate".as_bytes()), Some(301));
        assert_eq!(
            vocab.lookup_token("a".as_bytes()),
            Some(byte_vocab.get_token(b'a'))
        );

        assert_eq!(
            vocab.lookup_pair(&(byte_vocab.get_token(b'a'), byte_vocab.get_token(b't'))),
            Some(300)
        );
    }

    #[test]
    fn test_convert() {
        type A = u32;
        let mut span_vocab: SpanTokenMap<A> = Default::default();
        span_vocab.insert("at".as_bytes().to_vec(), 300);
        span_vocab.insert("ate".as_bytes().to_vec(), 301);
        let span_vocab: SpanMapVocab<A> = span_vocab.into();

        let seg_config = TextSpanningConfig::from_pattern(r"\w\+");

        let vocab32 = UnifiedTokenVocab::from_span_vocab(seg_config, span_vocab.clone()).unwrap();

        type B = u64;

        let vocab64: UnifiedTokenVocab<B> = vocab32.to_token_type::<B>().unwrap();

        assert_eq!(vocab64.lookup_token("at".as_bytes()), Some(300 as u64));
        assert_eq!(vocab64.lookup_token("ate".as_bytes()), Some(301 as u64));
    }

    #[test]
    fn test_input_normalizer() {
        type T = u32;

        let mut span_vocab: SpanTokenMap<T> = Default::default();
        span_vocab.insert("abc".as_bytes().to_vec(), 300);
        let span_vocab: SpanMapVocab<T> = span_vocab.into();

        let vocab = UnifiedTokenVocab::from_span_vocab(
            TextSpanningConfig::from_pattern(r"\w+"),
            span_vocab,
        )
        .unwrap()
        .with_input_normalizer(TextNormalizer::NFC);

        assert_eq!(vocab.input_normalizer(), Some(&TextNormalizer::NFC));
        assert_eq!(vocab.normalize_text("e\u{301}clair").as_ref(), "éclair");
        assert_eq!(
            vocab.to_token_type::<u64>().unwrap().input_normalizer(),
            Some(&TextNormalizer::NFC)
        );
    }

    #[test]
    fn test_unicode_scalar_seed_strategy() {
        let vocab = UnifiedTokenVocab::from_span_vocab(
            TextSpanningConfig::from_pattern(r".+"),
            SpanMapVocab::default(),
        )
        .unwrap()
        .with_unicode_scalar_seeding(
            [("é".as_bytes().to_vec(), 500)].into_iter().collect(),
            (0u32..=255).collect(),
            None,
        )
        .with_direct_word_lookup(false);

        let mut tokens = Vec::new();
        vocab.append_seed_tokens("éz".as_bytes(), &mut tokens);

        assert_eq!(tokens, vec![500, b'z' as u32]);
        assert!(!vocab.direct_word_lookup());
    }

    #[test]
    fn test_init_allows_undecomposable_span_tokens() {
        type T = u32;

        let mut span_vocab: SpanTokenMap<T> = Default::default();
        span_vocab.insert("abc".as_bytes().to_vec(), 300);
        let span_vocab: SpanMapVocab<T> = span_vocab.into();

        let seg_config = TextSpanningConfig::from_pattern(r"\w+");

        let vocab = UnifiedTokenVocab::from_span_vocab(seg_config, span_vocab).unwrap();

        assert_eq!(vocab.lookup_token("abc".as_bytes()), Some(300));
        assert!(!vocab.pair_vocab().tokens().contains(&300));
        assert!(vocab.pair_vocab().pair_map().is_empty());
    }
}
