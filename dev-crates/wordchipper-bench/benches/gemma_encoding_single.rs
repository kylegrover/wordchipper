#![allow(missing_docs)]

use divan::{
    Bencher,
    black_box,
    counter::BytesCount,
};
use wordchipper::{
    TokenEncoderOptions,
    encoders::token_span_encoder::SpanEncoderSelector,
};
use wordchipper_bench::{
    GemmaPrefixLatticeScanner,
    HF_GEMMA4_26B_A4B_IT,
    WC_GEMMA4_26B_A4B_IT,
    load_cached_encoder,
    load_cached_vocab,
};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

static DIVERSE_CORPUS: &str = include_str!("data/multilingual.txt");
static ENGLISH_CORPUS: &str = include_str!("data/english.txt");

fn diverse_text() -> String {
    DIVERSE_CORPUS.repeat(10)
}

fn english_text() -> String {
    ENGLISH_CORPUS.repeat(10)
}

fn bench_wc(
    bencher: Bencher,
    text: &str,
) {
    let encoder = load_cached_encoder::<u32>(WC_GEMMA4_26B_A4B_IT, TokenEncoderOptions::default());

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| encoder.try_encode(black_box(text), None).unwrap());
}

fn bench_rank_bucket(
    bencher: Bencher,
    text: &str,
) {
    let encoder = load_cached_encoder::<u32>(
        WC_GEMMA4_26B_A4B_IT,
        TokenEncoderOptions::default().with_span_encoder(SpanEncoderSelector::RankBucketMerge),
    );

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| encoder.try_encode(black_box(text), None).unwrap());
}

fn bench_hf(
    bencher: Bencher,
    text: &str,
) {
    let tok = tokenizers::Tokenizer::from_pretrained(HF_GEMMA4_26B_A4B_IT, None).unwrap();

    bencher
        .counter(BytesCount::new(text.len()))
        .bench(|| tok.encode(black_box(text), true).unwrap());
}

fn bench_lattice_scan(
    bencher: Bencher,
    text: &str,
) {
    let vocab = load_cached_vocab::<u32>(WC_GEMMA4_26B_A4B_IT).unwrap();
    let mut scanner = GemmaPrefixLatticeScanner::from_vocab(vocab.as_ref());
    scanner.warm_text(text);

    bencher
        .counter(BytesCount::new(text.len()))
        .bench_local(|| black_box(scanner.scan_text(black_box(text))));
}

mod english {
    use super::*;

    #[divan::bench]
    fn rank_bucket_merge(bencher: Bencher) {
        bench_rank_bucket(bencher, &english_text());
    }

    #[divan::bench]
    fn lattice_scan(bencher: Bencher) {
        bench_lattice_scan(bencher, &english_text());
    }

    #[divan::bench]
    fn wordchipper(bencher: Bencher) {
        bench_wc(bencher, &english_text());
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, &english_text());
    }
}

mod diverse {
    use super::*;

    #[divan::bench]
    fn rank_bucket_merge(bencher: Bencher) {
        bench_rank_bucket(bencher, &diverse_text());
    }

    #[divan::bench]
    fn lattice_scan(bencher: Bencher) {
        bench_lattice_scan(bencher, &diverse_text());
    }

    #[divan::bench]
    fn wordchipper(bencher: Bencher) {
        bench_wc(bencher, &diverse_text());
    }

    #[divan::bench]
    fn tokenizers(bencher: Bencher) {
        bench_hf(bencher, &diverse_text());
    }
}