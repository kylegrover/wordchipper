//! # Span Encoder Selector

use crate::{
    TokenType,
    alloc::{
        boxed::Box,
        sync::Arc,
    },
    encoders::token_span_encoder::{
        SpanEncoder,
        span_encoders::{
            BufferSweepSpanEncoder,
            MergeHeapSpanEncoder,
            PriorityMergeSpanEncoder,
            RankBucketMergeSpanEncoder,
            TailSweepSpanEncoder,
            bpe_backtrack_encoder::{
                BpeBacktrackSpanEncoder,
                BpeVocab,
            },
        },
    },
    vocab::UnifiedTokenVocab,
};

/// Policy enum for selecting a [`SpanEncoder`] for
/// [`TokenSpanEncoder`](`crate::encoders::token_span_encoder::TokenSpanEncoder`).
#[derive(
    Default, Debug, Clone, Copy, PartialEq, strum::EnumString, strum::EnumIter, strum::Display,
)]
#[non_exhaustive]
pub enum SpanEncoderSelector {
    /// This is the canonical "best" concurrent encoder.
    ///
    /// This is currently an alias for: [`BpeBacktrack`](`Self::BpeBacktrack`)
    #[default]
    ConcurrentDefault,

    /// This the canonical "best" single-threaded encoder.
    ///
    /// This is currently an alias for: [`BpeBacktrack`](`Self::BpeBacktrack`)
    SingleThreadDefault,

    /// The canonical reference encoder, [`BufferSweepSpanEncoder`].
    ///
    /// This encoder is meant to be used as a reference implementation for
    /// testing and comparison. The code and behavior are as simple as
    /// possible, but it is not optimized for performance.
    ///
    /// This is currently an alias for: [`BufferSweep`](`Self::BufferSweep`)
    Reference,

    /// Use the [`TailSweepSpanEncoder`] encoder.
    TailSweep,

    /// Use the [`MergeHeapSpanEncoder`] encoder.
    MergeHeap,

    /// Use the [`PriorityMergeSpanEncoder`] encoder.
    PriorityMerge,

    /// Use the experimental flat-vector rank-bucket merge encoder.
    RankBucketMerge,

    /// Use the [`BufferSweepSpanEncoder`] encoder.
    BufferSweep,

    /// Use the [`BpeBacktrackSpanEncoder`] encoder.
    BpeBacktrack,
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use super::SpanEncoderSelector;
    use crate::{
        TokenEncoder,
        UnifiedTokenVocab,
        alloc::{
            string::ToString,
            sync::Arc,
        },
        encoders::token_span_encoder::TokenSpanEncoder,
        spanners::{
            TextSpannerBuilder,
            TextSpanningConfig,
        },
        vocab::utility::testing::{
            build_test_shift_byte_vocab,
            build_test_vocab,
        },
    };

    #[test]
    fn test_span_encoder_selector_strum_roundtrip() {
        for variant in <SpanEncoderSelector as strum::IntoEnumIterator>::iter() {
            let s = variant.to_string();
            assert_eq!(
                SpanEncoderSelector::from_str(&s).unwrap(),
                variant,
                "roundtrip failed for variant string: {s}"
            );
        }
    }

    #[test]
    fn test_default_exact_hf_fallback_matches_rank_bucket_merge_behavior() {
        let vocab: Arc<UnifiedTokenVocab<u32>> = build_test_vocab(
            build_test_shift_byte_vocab(10),
            TextSpanningConfig::from_pattern(r".+"),
        )
        .with_unicode_scalar_seeding(
            [(b"e".to_vec(), 500u32)].into_iter().collect(),
            (10u32..=265).collect(),
            None,
        )
        .into();

        assert!(!vocab.supports_backtrack_encoder());

        let default_encoder = TokenSpanEncoder::<u32>::new_with_selector(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            SpanEncoderSelector::SingleThreadDefault,
        );
        let rank_bucket_encoder = TokenSpanEncoder::<u32>::new_with_selector(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            SpanEncoderSelector::RankBucketMerge,
        );

        for text in ["hello", "éhello", "helloé", "éhelloé"] {
            assert_eq!(
                default_encoder.try_encode(text, None).unwrap(),
                rank_bucket_encoder.try_encode(text, None).unwrap(),
                "default fallback diverged from explicit rank-bucket encoding for {text:?}"
            );
        }
    }

    #[cfg(all(feature = "client", feature = "download"))]
    #[test]
    #[ignore]
    fn test_hf_exact_models_do_not_support_backtrack_encoder() {
        use crate::{
            disk_cache::WordchipperDiskCache,
            load_vocab,
        };

        let mut disk_cache = WordchipperDiskCache::default();

        for model in ["hf:google/gemma-4-26B-A4B-it", "hf:Qwen/Qwen3.5-0.8B"] {
            let vocab: Arc<UnifiedTokenVocab<u32>> = load_vocab(model, &mut disk_cache)
                .unwrap()
                .vocab()
                .clone();
            assert!(
                !vocab.supports_backtrack_encoder(),
                "{model} unexpectedly still supports the backtrack encoder"
            );
        }
    }
}

impl SpanEncoderSelector {
    /// Get a builder for the configured [`SpanEncoder`].
    ///
    /// The `vocab` parameter is needed by encoders that pre-build data
    /// structures from the vocabulary (e.g. BPE automaton).
    pub fn span_encoder_builder<T: TokenType>(
        &self,
        vocab: &UnifiedTokenVocab<T>,
    ) -> Arc<dyn Fn() -> Box<dyn SpanEncoder<T>> + Send + Sync> {
        use SpanEncoderSelector::*;
        match self {
            Reference | BufferSweep => {
                Arc::new(|| Box::new(BufferSweepSpanEncoder::<T>::default()))
            }
            TailSweep => Arc::new(|| Box::new(TailSweepSpanEncoder::<T>::default())),
            MergeHeap => Arc::new(|| Box::new(MergeHeapSpanEncoder::<T>::default())),
            PriorityMerge => Arc::new(|| Box::new(PriorityMergeSpanEncoder::<T>::default())),
            RankBucketMerge => {
                let encoder = RankBucketMergeSpanEncoder::<T>::from_vocab(vocab);
                Arc::new(move || Box::new(encoder.clone()))
            }
            ConcurrentDefault | SingleThreadDefault | BpeBacktrack => {
                if vocab.supports_backtrack_encoder() {
                    let bpe_vocab = Arc::new(BpeVocab::from_vocab(vocab));
                    Arc::new(move || Box::new(BpeBacktrackSpanEncoder::new(bpe_vocab.clone())))
                } else {
                    let encoder = RankBucketMergeSpanEncoder::<T>::from_vocab(vocab);
                    Arc::new(move || Box::new(encoder.clone()))
                }
            }
        }
    }
}
