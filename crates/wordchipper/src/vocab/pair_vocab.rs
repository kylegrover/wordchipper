//! # Pair Map ``{ (T, T) -> T }`` Token Vocabulary

use crate::{
    WCResult,
    alloc::vec::Vec,
    decoders::{
        TokenDecoder,
        utility::PairExpansionDecoder,
    },
    types::{
        Pair,
        TokenType,
        WCHashSet,
    },
    vocab::{
        ByteMapVocab,
        PairRankMap,
        PairTokenMap,
        TokenSpanMap,
        VocabIndex,
        utility::validators::try_vocab_size,
    },
};

fn primitive_spans_from_byte_vocab<T: TokenType>(byte_vocab: &ByteMapVocab<T>) -> TokenSpanMap<T> {
    byte_vocab
        .span_pairs()
        .map(|(span, token)| (token, span))
        .collect()
}

fn pair_ranks_from_tokens<T: TokenType>(pairs: &PairTokenMap<T>) -> PairRankMap<T> {
    pairs.iter()
        .map(|(&pair, &token)| (pair, token.to_u32().unwrap()))
        .collect()
}

/// Validate that primitive spans and pair mappings are compatible.
///
/// - for every ``(a, b) -> t`` entry:
///   - the parents ``(a, b)``:
///     - are either in the primitive token map, or are targets in the map, not both.
///   - the target ``t`` is not itself a primitive token.
///
/// ## Arguments
/// * `primitive_spans` - The primitive token expansions to validate against.
/// * `pairs` - The pair token map to validate.
/// * `pair_ranks` - The merge-rank map to validate.
///
/// ## Returns
/// A `Result` indicating whether the maps are compatible.
pub fn try_validate_pair_map<T: TokenType>(
    primitive_spans: &TokenSpanMap<T>,
    pairs: &PairTokenMap<T>,
    pair_ranks: &PairRankMap<T>,
) -> WCResult<()> {
    if pairs.len() != pair_ranks.len() || pairs.keys().any(|pair| !pair_ranks.contains_key(pair)) {
        return Err(crate::WCError::VocabConflict(
            "pair map and merge-rank map have different keys".into(),
        ));
    }

    let primitive_tokens: WCHashSet<T> = primitive_spans.keys().copied().collect();
    let pair_targets: WCHashSet<T> = pairs.values().copied().collect();

    for t in &pair_targets {
        if primitive_tokens.contains(t) {
            return Err(crate::WCError::VocabConflict(crate::alloc::format!(
                "target token in pair map {t:?} is also a primitive token"
            )));
        }
    }

    const ORPHAN_TOKENS_ERROR: &str = indoc::indoc! {r#"
        This vocab has orphan tokens, which wordchipper does not yet support.
        See: https://github.com/zspacelabs/wordchipper/issues/386
    "#};

    for (&pair, &t) in pairs.iter() {
        for pt in [pair.0, pair.1] {
            let is_pair_target = pair_targets.contains(&pt);
            let is_primitive = primitive_tokens.contains(&pt);

            if is_pair_target && is_primitive {
                return Err(crate::WCError::NotImplemented(crate::alloc::format!(
                    "{PRE}Pair {pair:?} -> {t:?} parent {pt:?} is both a pair target and a primitive token",
                    PRE = ORPHAN_TOKENS_ERROR,
                )));
            }
            if !is_pair_target && !is_primitive {
                return Err(crate::WCError::NotImplemented(crate::alloc::format!(
                    "{PRE}Pair {pair:?} -> {t:?} parent {pt:?} is not defined",
                    PRE = ORPHAN_TOKENS_ERROR,
                )));
            }
        }
    }

    Ok(())
}

/// Pair ``(T, T) -> T`` Vocabulary.
///
/// - Grounded in a `ByteTable<T>` for byte-to-token mapping.
/// - Contains explicit primitive token expansions.
/// - Collection of ``(T, T) -> T`` pairs plus merge ranks.
#[derive(Debug, Clone, PartialEq)]
pub struct PairMapVocab<T: TokenType> {
    /// Byte/token mapping table.
    byte_vocab: ByteMapVocab<T>,

    /// Primitive token expansions used as encoder/decoder leaves.
    primitive_spans: TokenSpanMap<T>,

    /// Map of ``{ (T, T) -> T }``.
    pair_map: PairTokenMap<T>,

    /// Map of ``{ (T, T) -> rank }``.
    pair_ranks: PairRankMap<T>,
}

impl<T: TokenType> Default for PairMapVocab<T> {
    fn default() -> Self {
        Self::new(ByteMapVocab::default(), PairTokenMap::default()).unwrap()
    }
}

impl<T: TokenType> PairMapVocab<T> {
    /// Initialize a byte-grounded [`PairMapVocab`].
    ///
    /// ## Arguments
    /// * `byte_vocab` - The byte vocabulary mapping.
    /// * `pairs` - The pair token map.
    ///
    /// ## Returns
    /// A `Result` containing the new `PairMapVocab` instance or an error.
    pub fn new(
        byte_vocab: ByteMapVocab<T>,
        pairs: PairTokenMap<T>,
    ) -> WCResult<Self> {
        Self::new_with_parts(
            byte_vocab.clone(),
            primitive_spans_from_byte_vocab(&byte_vocab),
            pairs.clone(),
            pair_ranks_from_tokens(&pairs),
        )
    }

    /// Initialize a [`PairMapVocab`] with explicit primitive spans and merge ranks.
    pub(crate) fn new_with_parts(
        byte_vocab: ByteMapVocab<T>,
        mut primitive_spans: TokenSpanMap<T>,
        mut pairs: PairTokenMap<T>,
        mut pair_ranks: PairRankMap<T>,
    ) -> WCResult<Self> {
        try_validate_pair_map(&primitive_spans, &pairs, &pair_ranks)?;
        primitive_spans.shrink_to_fit();
        pairs.shrink_to_fit();
        pair_ranks.shrink_to_fit();

        Ok(Self {
            byte_vocab,
            primitive_spans,
            pair_map: pairs,
            pair_ranks,
        })
    }

    /// Convert to a different token type.
    pub fn to_token_type<G: TokenType>(&self) -> WCResult<PairMapVocab<G>> {
        try_vocab_size::<G>(self.max_token().unwrap().to_usize().unwrap() + 1)?;

        PairMapVocab::<G>::new_with_parts(
            self.byte_vocab.to_token_type::<G>()?,
            self.primitive_spans
                .iter()
                .map(|(&token, span)| (G::from(token).unwrap(), span.clone()))
                .collect(),
            self.pair_map
                .iter()
                .map(|(&(a, b), &token)| {
                    (
                        (G::from(a).unwrap(), G::from(b).unwrap()),
                        G::from(token).unwrap(),
                    )
                })
                .collect(),
            self.pair_ranks
                .iter()
                .map(|(&(a, b), &rank)| ((G::from(a).unwrap(), G::from(b).unwrap()), rank))
                .collect(),
        )
    }

    /// Get the byte vocabulary.
    pub fn byte_vocab(&self) -> &ByteMapVocab<T> {
        &self.byte_vocab
    }

    /// Get the primitive token expansions.
    pub(crate) fn primitive_spans(&self) -> &TokenSpanMap<T> {
        &self.primitive_spans
    }

    /// Get the map of pairs.
    pub fn pair_map(&self) -> &PairTokenMap<T> {
        &self.pair_map
    }

    /// Looks up a pair.
    ///
    /// ## Arguments
    /// * `pair` - The pair of tokens to look up.
    ///
    /// ## Returns
    /// An `Option` containing the token corresponding to the pair if it exists.
    pub fn lookup_pair(
        &self,
        pair: &Pair<T>,
    ) -> Option<T> {
        self.pair_map.get(pair).copied()
    }

    /// Looks up a pair together with its merge rank.
    pub fn lookup_pair_merge(
        &self,
        pair: &Pair<T>,
    ) -> Option<(u32, T)> {
        self.pair_map
            .get(pair)
            .zip(self.pair_ranks.get(pair))
            .map(|(&token, &rank)| (rank, token))
    }
}

impl<T: TokenType> VocabIndex<T> for PairMapVocab<T> {
    type Token = T;

    fn len(&self) -> usize {
        self.primitive_spans.len() + self.pair_map.len()
    }

    fn tokens(&self) -> WCHashSet<T> {
        self.primitive_spans
            .keys()
            .copied()
            .chain(self.pair_map.values().copied())
            .collect::<WCHashSet<T>>()
    }

    fn max_token(&self) -> Option<T> {
        let max_t = self.primitive_spans.keys().max().copied();
        let max_p = self.pair_map.values().max().copied();
        [max_t, max_p].into_iter().flatten().max()
    }

    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        let decoder = PairExpansionDecoder::from_pair_vocab(self);

        self.primitive_spans
            .iter()
            .map(|(&token, span)| (span.clone(), token))
            .chain(
                self.pair_map
                    .values()
                    .map(move |&t| (decoder.try_decode_to_bytes(&[t]).unwrap().unwrap(), t)),
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::ByteMapVocab;

    #[test]
    fn test_tokens_sorted() {
        type T = u32;
        let byte_vocab: ByteMapVocab<T> = Default::default();

        let mut vocab = PairMapVocab::<T>::default();

        assert_eq!(vocab.max_token().unwrap(), 255);

        assert_eq!(&vocab.tokens(), &byte_vocab.tokens());

        vocab.pair_map.insert((1, 2), 300);
        vocab.pair_map.insert((3, 4), 301);
        vocab.pair_map.insert((300, 301), 302);

        assert_eq!(vocab.max_token().unwrap(), 302);
        assert_eq!(vocab.len(), 256 + 3);

        assert_eq!(
            &vocab.tokens(),
            &byte_vocab
                .tokens()
                .into_iter()
                .chain([300_u32, 301, 302].into_iter())
                .collect()
        );
    }

    #[test]
    fn test_to_token_type_accepts_minimum_byte_vocab() {
        let vocab = PairMapVocab::<u32>::default();

        assert_eq!(vocab.max_token(), Some(255));

        let converted = vocab.to_token_type::<u8>().unwrap();
        assert_eq!(converted.max_token(), Some(255));
        assert_eq!(converted.len(), 256);
    }
}
