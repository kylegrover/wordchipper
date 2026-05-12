use std::time::Instant;

use wordchipper::{
    encoders::token_span_encoder::span_encoders::{
        RankBucketMergeProfile,
        RankBucketMergeSpanEncoder,
    },
    spanners::{
        SpanRef,
        TextSpannerBuilder,
    },
};
use wordchipper_bench::{
    WC_GEMMA4_26B_A4B_IT,
    WC_QWEN35,
    load_cached_vocab,
};

static DIVERSE_CORPUS: &str = include_str!("../../benches/data/multilingual.txt");
static ENGLISH_CORPUS: &str = include_str!("../../benches/data/english.txt");

fn repeated(text: &str) -> String {
    text.repeat(10)
}

fn max_rank(vocab: &wordchipper::UnifiedTokenVocab<u32>) -> usize {
    vocab
        .pair_vocab()
        .pair_map()
        .keys()
        .filter_map(|pair| vocab.lookup_pair_merge(pair).map(|(rank, _)| rank as usize))
        .max()
        .unwrap_or(0)
}

fn profile_text(
    model: &str,
    label: &str,
    text: &str,
) {
    let vocab = load_cached_vocab::<u32>(model).unwrap();
    let spanner = TextSpannerBuilder::default(&vocab);
    let mut encoder = RankBucketMergeSpanEncoder::<u32>::new(max_rank(vocab.as_ref()));
    let mut tokens = Vec::new();
    let mut total_profile = RankBucketMergeProfile::default();

    let start = Instant::now();
    spanner.for_each_split_span(text, None, &mut |span_ref| {
        if let SpanRef::Word(range) = span_ref {
            let span = &text[range].as_bytes();
            let profile = encoder.profile_compound_span(vocab.as_ref(), span, &mut tokens);
            total_profile.accumulate(&profile);
        }
        true
    });
    let encode_time = start.elapsed();

    println!("model={model} corpus={label}");
    println!("  encode_ms={:.3}", encode_time.as_secs_f64() * 1000.0);
    println!(
        "  next_active_rank_ms={:.3}",
        total_profile.next_active_rank_time.as_secs_f64() * 1000.0
    );
    println!(
        "  unlink_pair_ms={:.3}",
        total_profile.unlink_pair_time.as_secs_f64() * 1000.0
    );
    println!(
        "  activate_pair_ms={:.3}",
        total_profile.activate_pair_time.as_secs_f64() * 1000.0
    );
    println!("  pairs_popped={}", total_profile.pairs_popped);
    println!("  pairs_merged={}", total_profile.pairs_merged);
    println!("  pairs_unlinked={}", total_profile.pairs_unlinked);
    println!("  pairs_activated={}", total_profile.pairs_activated);
    println!(
        "  pairs_rejected_inactive={}",
        total_profile.pairs_rejected_inactive
    );
    println!(
        "  pairs_rejected_missing_lookup={}",
        total_profile.pairs_rejected_missing_lookup
    );
    println!(
        "  pairs_rejected_rank_mismatch={}",
        total_profile.pairs_rejected_rank_mismatch
    );
    println!("  pop_reject_ratio={:.4}", total_profile.pop_reject_ratio());
    println!("  tokens_out={}", tokens.len());
}

fn main() {
    let english = repeated(ENGLISH_CORPUS);
    let diverse = repeated(DIVERSE_CORPUS);

    profile_text(WC_GEMMA4_26B_A4B_IT, "english", &english);
    profile_text(WC_GEMMA4_26B_A4B_IT, "diverse", &diverse);
    profile_text(WC_QWEN35, "english", &english);
    profile_text(WC_QWEN35, "diverse", &diverse);
}