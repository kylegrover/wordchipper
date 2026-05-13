//! # Rank-bucket merge [`SpanEncoder`].
//!
//! Uses flat token and pair vectors with intrusive occurrence lists keyed by
//! merge rank to avoid heap traffic during exact-HF BPE merging.

use core::time::Duration;

#[cfg(feature = "std")]
use std::time::Instant;

use crate::{
    alloc::sync::Arc,
    types::{
        Pair,
        TokenType,
        hash_map_with_capacity,
    },
    alloc::vec::Vec,
    encoders::token_span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

type PairLookupMap<T> = crate::types::WCHashMap<Pair<T>, (u32, T)>;

#[derive(Clone, Copy, Default)]
struct TokenNode<T> {
    token_id: T,
    is_active: bool,
    prev_text_idx: Option<usize>,
    next_text_idx: Option<usize>,
}

#[derive(Clone, Copy, Default)]
struct PairNode {
    left_text_idx: usize,
    right_text_idx: usize,
    rank: usize,
    prev_occurrence_idx: Option<usize>,
    next_occurrence_idx: Option<usize>,
}

/// Internal profiling counters for the flat-vector rank-bucket encoder.
#[derive(Debug, Default, Clone)]
pub struct RankBucketMergeProfile {
    /// Time spent finding the next active rank.
    pub next_active_rank_time: Duration,

    /// Time spent unlinking pair nodes from occurrence lists.
    pub unlink_pair_time: Duration,

    /// Time spent activating new pair nodes.
    pub activate_pair_time: Duration,

    /// Time spent looking up `(left_tok, right_tok)` in the vocab pair map.
    pub activate_lookup_time: Duration,

    /// Time spent inserting new pair nodes into the rank occurrence lists.
    pub activate_insert_time: Duration,

    /// Total pair nodes popped from the rank heads.
    pub pairs_popped: u64,

    /// Pair nodes rejected by the active-adjacency guard.
    pub pairs_rejected_inactive: u64,

    /// Pair nodes rejected because the current tokens no longer map to a merge.
    pub pairs_rejected_missing_lookup: u64,

    /// Pair nodes rejected because the pair rank changed before pop.
    pub pairs_rejected_rank_mismatch: u64,

    /// Successful merges executed.
    pub pairs_merged: u64,

    /// Pair unlink operations executed.
    pub pairs_unlinked: u64,

    /// Pair activation operations executed.
    pub pairs_activated: u64,
}

impl RankBucketMergeProfile {
    /// The total number of popped pairs rejected before merge.
    pub fn total_rejected(&self) -> u64 {
        self.pairs_rejected_inactive
            + self.pairs_rejected_missing_lookup
            + self.pairs_rejected_rank_mismatch
    }

    /// Rejected pops divided by total pops.
    pub fn pop_reject_ratio(&self) -> f64 {
        if self.pairs_popped == 0 {
            return 0.0;
        }
        self.total_rejected() as f64 / self.pairs_popped as f64
    }

    /// Merge counters and timers from another profile.
    pub fn accumulate(
        &mut self,
        other: &Self,
    ) {
        self.next_active_rank_time += other.next_active_rank_time;
        self.unlink_pair_time += other.unlink_pair_time;
        self.activate_pair_time += other.activate_pair_time;
        self.activate_lookup_time += other.activate_lookup_time;
        self.activate_insert_time += other.activate_insert_time;
        self.pairs_popped += other.pairs_popped;
        self.pairs_rejected_inactive += other.pairs_rejected_inactive;
        self.pairs_rejected_missing_lookup += other.pairs_rejected_missing_lookup;
        self.pairs_rejected_rank_mismatch += other.pairs_rejected_rank_mismatch;
        self.pairs_merged += other.pairs_merged;
        self.pairs_unlinked += other.pairs_unlinked;
        self.pairs_activated += other.pairs_activated;
    }
}

struct HierarchicalBitSet {
    levels: Vec<Vec<u64>>,
    touched_words: Vec<Vec<usize>>,
    max_bit: usize,
}

impl HierarchicalBitSet {
    fn new(bit_len: usize) -> Self {
        let bit_len = bit_len.max(1);
        let mut word_len = bit_len.div_ceil(64);
        let mut levels = vec![vec![0; word_len]];
        let mut touched_words = vec![Vec::new()];

        while word_len > 1 {
            word_len = word_len.div_ceil(64);
            levels.push(vec![0; word_len]);
            touched_words.push(Vec::new());
        }

        Self {
            levels,
            touched_words,
            max_bit: bit_len - 1,
        }
    }

    fn clear(&mut self) {
        for (level, touched) in self.touched_words.iter_mut().enumerate() {
            for &word_idx in touched.iter() {
                self.levels[level][word_idx] = 0;
            }
            touched.clear();
        }
    }

    fn set(&mut self, bit_idx: usize) {
        let mut idx = bit_idx;
        for level in 0..self.levels.len() {
            let word_idx = idx / 64;
            let mask = 1u64 << (idx % 64);
            let word = &mut self.levels[level][word_idx];
            let prev = *word;

            if prev & mask != 0 {
                break;
            }
            if prev == 0 {
                self.touched_words[level].push(word_idx);
            }

            *word |= mask;
            if prev != 0 {
                break;
            }

            idx = word_idx;
        }
    }

    fn clear_bit(&mut self, bit_idx: usize) {
        let mut idx = bit_idx;
        for level in 0..self.levels.len() {
            let word_idx = idx / 64;
            let mask = 1u64 << (idx % 64);
            let word = &mut self.levels[level][word_idx];

            if *word & mask == 0 {
                break;
            }

            *word &= !mask;
            if *word != 0 {
                break;
            }

            idx = word_idx;
        }
    }

    fn next_set_bit_from(
        &self,
        start_bit: usize,
    ) -> Option<usize> {
        if start_bit > self.max_bit {
            return None;
        }

        self.next_in_level(0, start_bit)
            .filter(|&bit_idx| bit_idx <= self.max_bit)
    }

    fn next_in_level(
        &self,
        level: usize,
        start_idx: usize,
    ) -> Option<usize> {
        if level + 1 == self.levels.len() {
            return self.scan_level_words(level, start_idx);
        }

        let start_word = start_idx / 64;
        let start_bit = start_idx % 64;
        let mut word_idx = self.next_in_level(level + 1, start_word)?;

        loop {
            if word_idx >= self.levels[level].len() {
                return None;
            }

            let word = self.levels[level][word_idx];
            let bit_start = if word_idx == start_word { start_bit } else { 0 };
            let masked = word & (u64::MAX << bit_start);
            if masked != 0 {
                return Some(word_idx * 64 + masked.trailing_zeros() as usize);
            }

            word_idx = self.next_in_level(level + 1, word_idx + 1)?;
        }
    }

    fn scan_level_words(
        &self,
        level: usize,
        start_idx: usize,
    ) -> Option<usize> {
        let words = &self.levels[level];
        let mut word_idx = start_idx / 64;
        if word_idx >= words.len() {
            return None;
        }

        let mut bits = words[word_idx] & (u64::MAX << (start_idx % 64));
        loop {
            if bits != 0 {
                return Some(word_idx * 64 + bits.trailing_zeros() as usize);
            }

            word_idx += 1;
            if word_idx >= words.len() {
                return None;
            }
            bits = words[word_idx];
        }
    }
}

/// A [`SpanEncoder`] using flat vectors and rank-indexed occurrence lists.
pub struct RankBucketMergeSpanEncoder<T: TokenType> {
    max_rank: usize,
    pair_lookup: Arc<PairLookupMap<T>>,
    seed_tokens: Vec<T>,
    tokens: Vec<TokenNode<T>>,
    pairs: Vec<PairNode>,
    rank_heads: Vec<Option<usize>>,
    active_ranks: HierarchicalBitSet,
    touched_ranks: Vec<usize>,
    token_to_pair: Vec<Option<usize>>,
    current_min_rank: usize,
}

impl<T: TokenType> RankBucketMergeSpanEncoder<T> {
    /// Create a new encoder with capacity for ranks up to `max_rank`.
    pub fn new(
        max_rank: usize,
        pair_lookup: Arc<PairLookupMap<T>>,
    ) -> Self {
        let rank_len = max_rank + 1;
        Self {
            max_rank,
            pair_lookup,
            seed_tokens: Vec::new(),
            tokens: Vec::new(),
            pairs: Vec::new(),
            rank_heads: vec![None; rank_len],
            active_ranks: HierarchicalBitSet::new(rank_len),
            touched_ranks: Vec::new(),
            token_to_pair: Vec::new(),
            current_min_rank: rank_len,
        }
    }

    /// Build a new encoder from a vocabulary, caching pair rank lookups once.
    pub fn from_vocab(vocab: &UnifiedTokenVocab<T>) -> Self {
        let mut max_rank = 0usize;
        let mut pair_lookup: PairLookupMap<T> =
            hash_map_with_capacity(vocab.pair_vocab().pair_map().len());

        for (&pair, &merge_token) in vocab.pair_vocab().pair_map() {
            if let Some((rank, _)) = vocab.lookup_pair_merge(&pair) {
                max_rank = max_rank.max(rank as usize);
                pair_lookup.insert(pair, (rank, merge_token));
            }
        }

        Self::new(max_rank, Arc::new(pair_lookup))
    }

    fn lookup_pair_merge(
        &self,
        pair: &Pair<T>,
    ) -> Option<(u32, T)> {
        self.pair_lookup.get(pair).copied()
    }

    fn reset_rank_heads(&mut self) {
        for &rank in &self.touched_ranks {
            self.rank_heads[rank] = None;
        }
        self.touched_ranks.clear();
        self.active_ranks.clear();
        self.current_min_rank = self.rank_heads.len();
    }

    fn set_active_rank(
        &mut self,
        rank: usize,
    ) {
        self.active_ranks.set(rank);
    }

    fn clear_active_rank(
        &mut self,
        rank: usize,
    ) {
        self.active_ranks.clear_bit(rank);
    }

    fn next_active_rank(
        &self,
        start_rank: usize,
    ) -> Option<usize> {
        self.active_ranks.next_set_bit_from(start_rank)
    }

    fn insert_pair(
        &mut self,
        pair_idx: usize,
    ) {
        let rank = self.pairs[pair_idx].rank;
        if self.rank_heads[rank].is_none() {
            self.touched_ranks.push(rank);
            self.set_active_rank(rank);
        }

        let head = self.rank_heads[rank];
        self.pairs[pair_idx].prev_occurrence_idx = None;
        self.pairs[pair_idx].next_occurrence_idx = head;
        if let Some(head_idx) = head {
            self.pairs[head_idx].prev_occurrence_idx = Some(pair_idx);
        }
        self.rank_heads[rank] = Some(pair_idx);
        if rank < self.current_min_rank {
            self.current_min_rank = rank;
        }
    }

    fn unlink_pair(
        &mut self,
        pair_idx: usize,
        mut profile: Option<&mut RankBucketMergeProfile>,
    ) {
        #[cfg(feature = "std")]
        let start = profile.as_ref().map(|_| Instant::now());

        let pair = self.pairs[pair_idx];
        if let Some(prev_idx) = pair.prev_occurrence_idx {
            self.pairs[prev_idx].next_occurrence_idx = pair.next_occurrence_idx;
        } else {
            self.rank_heads[pair.rank] = pair.next_occurrence_idx;
            if pair.next_occurrence_idx.is_none() {
                self.clear_active_rank(pair.rank);
            }
        }

        if let Some(next_idx) = pair.next_occurrence_idx {
            self.pairs[next_idx].prev_occurrence_idx = pair.prev_occurrence_idx;
        }

        self.pairs[pair_idx].prev_occurrence_idx = None;
        self.pairs[pair_idx].next_occurrence_idx = None;

        if let Some(profile) = profile.as_deref_mut() {
            profile.pairs_unlinked += 1;
            #[cfg(feature = "std")]
            if let Some(start) = start {
                profile.unlink_pair_time += start.elapsed();
            }
        }
    }

    fn pop_min_pair_profiled(
        &mut self,
        mut profile: Option<&mut RankBucketMergeProfile>,
    ) -> Option<usize> {
        loop {
            #[cfg(feature = "std")]
            let start = profile.as_ref().map(|_| Instant::now());
            let rank = self.next_active_rank(self.current_min_rank)?;
            if let Some(profile) = profile.as_deref_mut() {
                #[cfg(feature = "std")]
                if let Some(start) = start {
                    profile.next_active_rank_time += start.elapsed();
                }
            }
            self.current_min_rank = rank;

            let Some(pair_idx) = self.rank_heads[rank] else {
                self.current_min_rank = rank.saturating_add(1);
                continue;
            };

            self.unlink_pair(pair_idx, profile.as_deref_mut());
            if let Some(profile) = profile.as_deref_mut() {
                profile.pairs_popped += 1;
            }
            return Some(pair_idx);
        }
    }

    fn deactivate_pair_starting_at(
        &mut self,
        left_idx: Option<usize>,
        mut profile: Option<&mut RankBucketMergeProfile>,
    ) {
        let Some(left_idx) = left_idx else {
            return;
        };

        let Some(pair_idx) = self.token_to_pair[left_idx].take() else {
            return;
        };

        self.unlink_pair(pair_idx, profile.as_deref_mut());
    }

    fn activate_pair_starting_at(
        &mut self,
        left_idx: usize,
        mut profile: Option<&mut RankBucketMergeProfile>,
    ) {
        if left_idx >= self.pairs.len() || !self.tokens[left_idx].is_active {
            if left_idx < self.token_to_pair.len() {
                self.token_to_pair[left_idx] = None;
            }
            return;
        }

        let Some(right_idx) = self.tokens[left_idx].next_text_idx else {
            self.token_to_pair[left_idx] = None;
            return;
        };

        #[cfg(feature = "std")]
        let start = profile.as_ref().map(|_| Instant::now());
        #[cfg(feature = "std")]
        let lookup_start = profile.as_ref().map(|_| Instant::now());

        let left_tok = self.tokens[left_idx].token_id;
        let right_tok = self.tokens[right_idx].token_id;
        let Some((rank, _)) = self.lookup_pair_merge(&(left_tok, right_tok)) else {
            if let Some(profile) = profile.as_deref_mut() {
                #[cfg(feature = "std")]
                if let Some(lookup_start) = lookup_start {
                    profile.activate_lookup_time += lookup_start.elapsed();
                }
            }
            self.token_to_pair[left_idx] = None;
            return;
        };

        if let Some(profile) = profile.as_deref_mut() {
            #[cfg(feature = "std")]
            if let Some(lookup_start) = lookup_start {
                profile.activate_lookup_time += lookup_start.elapsed();
            }
        }

        #[cfg(feature = "std")]
        let insert_start = profile.as_ref().map(|_| Instant::now());
        let pair_idx = left_idx;
        self.pairs[pair_idx] = PairNode {
            left_text_idx: left_idx,
            right_text_idx: right_idx,
            rank: rank as usize,
            prev_occurrence_idx: None,
            next_occurrence_idx: None,
        };
        self.token_to_pair[left_idx] = Some(pair_idx);
        self.insert_pair(pair_idx);

        if let Some(profile) = profile.as_deref_mut() {
            profile.pairs_activated += 1;
            #[cfg(feature = "std")]
            if let Some(insert_start) = insert_start {
                profile.activate_insert_time += insert_start.elapsed();
            }
            #[cfg(feature = "std")]
            if let Some(start) = start {
                profile.activate_pair_time += start.elapsed();
            }
        }
    }

    fn encode_append_compound_span_impl(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
        mut profile: Option<&mut RankBucketMergeProfile>,
    ) {
        self.seed_tokens.clear();
        vocab.append_seed_tokens(span, &mut self.seed_tokens);
        let n = self.seed_tokens.len();

        if n < 2 {
            tokens.extend_from_slice(&self.seed_tokens);
            return;
        }

        self.tokens.clear();
        self.tokens.resize(n, TokenNode::default());
        self.pairs.clear();
        self.pairs.resize(n - 1, PairNode::default());
        self.token_to_pair.clear();
        self.token_to_pair.resize(n, None);
        self.reset_rank_heads();

        for (idx, &token_id) in self.seed_tokens.iter().enumerate() {
            self.tokens[idx] = TokenNode {
                token_id,
                is_active: true,
                prev_text_idx: idx.checked_sub(1),
                next_text_idx: (idx + 1 < n).then_some(idx + 1),
            };
        }

        for left_idx in (0..(n - 1)).rev() {
            self.activate_pair_starting_at(left_idx, profile.as_deref_mut());
        }

        while let Some(pair_idx) = self.pop_min_pair_profiled(profile.as_deref_mut()) {
            let pair = self.pairs[pair_idx];
            let left_idx = pair.left_text_idx;
            let right_idx = pair.right_text_idx;

            if !self.tokens[left_idx].is_active
                || self.token_to_pair[left_idx] != Some(pair_idx)
                || self.tokens[left_idx].next_text_idx != Some(right_idx)
            {
                if let Some(profile) = profile.as_deref_mut() {
                    profile.pairs_rejected_inactive += 1;
                }
                continue;
            }

            let left_tok = self.tokens[left_idx].token_id;
            let right_tok = self.tokens[right_idx].token_id;
            let Some((rank, merge_token)) = self.lookup_pair_merge(&(left_tok, right_tok)) else {
                self.token_to_pair[left_idx] = None;
                if let Some(profile) = profile.as_deref_mut() {
                    profile.pairs_rejected_missing_lookup += 1;
                }
                continue;
            };

            if rank as usize != pair.rank {
                self.token_to_pair[left_idx] = None;
                if let Some(profile) = profile.as_deref_mut() {
                    profile.pairs_rejected_rank_mismatch += 1;
                }
                continue;
            }

            if let Some(profile) = profile.as_deref_mut() {
                profile.pairs_merged += 1;
            }

            self.token_to_pair[left_idx] = None;
            let left_prev = self.tokens[left_idx].prev_text_idx;
            let right_next = self.tokens[right_idx].next_text_idx;

            self.deactivate_pair_starting_at(left_prev, profile.as_deref_mut());
            self.deactivate_pair_starting_at(Some(right_idx), profile.as_deref_mut());

            self.tokens[left_idx].token_id = merge_token;
            self.tokens[left_idx].next_text_idx = right_next;
            self.tokens[right_idx].is_active = false;
            self.tokens[right_idx].prev_text_idx = None;
            self.tokens[right_idx].next_text_idx = None;

            if let Some(next_idx) = right_next {
                self.tokens[next_idx].prev_text_idx = Some(left_idx);
            }

            if let Some(prev_idx) = left_prev {
                self.activate_pair_starting_at(prev_idx, profile.as_deref_mut());
            }
            if right_next.is_some() {
                self.activate_pair_starting_at(left_idx, profile.as_deref_mut());
            }
        }

        let mut idx = Some(0usize);
        while let Some(token_idx) = idx {
            tokens.push(self.tokens[token_idx].token_id);
            idx = self.tokens[token_idx].next_text_idx;
        }
    }

    /// Profile a single compound span with the real merge loop.
    #[cfg(feature = "std")]
    pub fn profile_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) -> RankBucketMergeProfile {
        let mut profile = RankBucketMergeProfile::default();
        self.encode_append_compound_span_impl(vocab, span, tokens, Some(&mut profile));
        profile
    }
}

impl<T: TokenType> core::fmt::Debug for RankBucketMergeSpanEncoder<T> {
    fn fmt(
        &self,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        f.debug_struct("RankBucketMergeSpanEncoder").finish()
    }
}

impl<T: TokenType> Clone for RankBucketMergeSpanEncoder<T> {
    fn clone(&self) -> Self {
        Self::new(self.max_rank, self.pair_lookup.clone())
    }
}

impl<T: TokenType> SpanEncoder<T> for RankBucketMergeSpanEncoder<T> {
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        self.encode_append_compound_span_impl(vocab, span, tokens, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TokenEncoder,
        TokenType,
        alloc::sync::Arc,
        encoders::{
            testing::{
                common_encoder_test_vocab,
                common_encoder_tests,
            },
            token_span_encoder::{
                SpanEncoderSelector,
                TokenSpanEncoder,
            },
        },
        spanners::TextSpannerBuilder,
        vocab::UnifiedTokenVocab,
    };

    fn test_encoder<T: TokenType>() {
        let vocab: Arc<UnifiedTokenVocab<T>> = common_encoder_test_vocab().into();
        let encoder = TokenSpanEncoder::<T>::new_with_selector(
            TextSpannerBuilder::default(&vocab),
            vocab.clone(),
            SpanEncoderSelector::RankBucketMerge,
        );
        let encoder: Arc<dyn TokenEncoder<T>> = Arc::new(encoder);
        common_encoder_tests(vocab, encoder)
    }

    #[test]
    fn test_rank_bucket_merge_encoder_u16() {
        test_encoder::<u16>();
    }

    #[test]
    fn test_rank_bucket_merge_encoder_u32() {
        test_encoder::<u32>();
    }
}