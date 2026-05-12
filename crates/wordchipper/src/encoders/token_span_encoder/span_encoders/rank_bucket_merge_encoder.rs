//! # Rank-bucket merge [`SpanEncoder`].
//!
//! Uses flat token and pair vectors with intrusive occurrence lists keyed by
//! merge rank to avoid heap traffic during exact-HF BPE merging.

use crate::{
    TokenType,
    alloc::vec::Vec,
    encoders::token_span_encoder::SpanEncoder,
    vocab::UnifiedTokenVocab,
};

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

/// A [`SpanEncoder`] using flat vectors and rank-indexed occurrence lists.
pub struct RankBucketMergeSpanEncoder<T: TokenType> {
    max_rank: usize,
    seed_tokens: Vec<T>,
    tokens: Vec<TokenNode<T>>,
    pairs: Vec<PairNode>,
    rank_heads: Vec<Option<usize>>,
    active_words: Vec<u64>,
    touched_ranks: Vec<usize>,
    touched_words: Vec<usize>,
    token_to_pair: Vec<Option<usize>>,
    current_min_rank: usize,
}

impl<T: TokenType> RankBucketMergeSpanEncoder<T> {
    /// Create a new encoder with capacity for ranks up to `max_rank`.
    pub fn new(max_rank: usize) -> Self {
        let rank_len = max_rank + 1;
        Self {
            max_rank,
            seed_tokens: Vec::new(),
            tokens: Vec::new(),
            pairs: Vec::new(),
            rank_heads: vec![None; rank_len],
            active_words: vec![0; rank_len.div_ceil(64)],
            touched_ranks: Vec::new(),
            touched_words: Vec::new(),
            token_to_pair: Vec::new(),
            current_min_rank: rank_len,
        }
    }

    fn reset_rank_heads(&mut self) {
        for &rank in &self.touched_ranks {
            self.rank_heads[rank] = None;
        }
        for &word_idx in &self.touched_words {
            self.active_words[word_idx] = 0;
        }
        self.touched_ranks.clear();
        self.touched_words.clear();
        self.current_min_rank = self.rank_heads.len();
    }

    fn set_active_rank(
        &mut self,
        rank: usize,
    ) {
        let word_idx = rank / 64;
        let mask = 1u64 << (rank % 64);
        if self.active_words[word_idx] == 0 {
            self.touched_words.push(word_idx);
        }
        self.active_words[word_idx] |= mask;
    }

    fn clear_active_rank(
        &mut self,
        rank: usize,
    ) {
        let word_idx = rank / 64;
        let mask = 1u64 << (rank % 64);
        self.active_words[word_idx] &= !mask;
    }

    fn next_active_rank(
        &self,
        start_rank: usize,
    ) -> Option<usize> {
        if start_rank > self.max_rank {
            return None;
        }

        let mut word_idx = start_rank / 64;
        let bit_offset = start_rank % 64;
        let mut bits = self.active_words[word_idx] & (u64::MAX << bit_offset);

        loop {
            if bits != 0 {
                return Some(word_idx * 64 + bits.trailing_zeros() as usize);
            }

            word_idx += 1;
            if word_idx >= self.active_words.len() {
                return None;
            }
            bits = self.active_words[word_idx];
        }
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
    ) {
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
    }

    fn pop_min_pair(&mut self) -> Option<usize> {
        loop {
            let rank = self.next_active_rank(self.current_min_rank)?;
            self.current_min_rank = rank;

            let Some(pair_idx) = self.rank_heads[rank] else {
                self.current_min_rank = rank.saturating_add(1);
                continue;
            };

            self.unlink_pair(pair_idx);
            return Some(pair_idx);
        }
    }

    fn deactivate_pair_starting_at(
        &mut self,
        left_idx: Option<usize>,
    ) {
        let Some(left_idx) = left_idx else {
            return;
        };

        let Some(pair_idx) = self.token_to_pair[left_idx].take() else {
            return;
        };

        self.unlink_pair(pair_idx);
    }

    fn activate_pair_starting_at(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        left_idx: usize,
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

        let left_tok = self.tokens[left_idx].token_id;
        let right_tok = self.tokens[right_idx].token_id;
        let Some((rank, _)) = vocab.lookup_pair_merge(&(left_tok, right_tok)) else {
            self.token_to_pair[left_idx] = None;
            return;
        };

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
        Self::new(self.max_rank)
    }
}

impl<T: TokenType> SpanEncoder<T> for RankBucketMergeSpanEncoder<T> {
    fn encode_append_compound_span(
        &mut self,
        vocab: &UnifiedTokenVocab<T>,
        span: &[u8],
        tokens: &mut Vec<T>,
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
            self.activate_pair_starting_at(vocab, left_idx);
        }

        while let Some(pair_idx) = self.pop_min_pair() {
            let pair = self.pairs[pair_idx];
            let left_idx = pair.left_text_idx;
            let right_idx = pair.right_text_idx;

            if !self.tokens[left_idx].is_active
                || self.token_to_pair[left_idx] != Some(pair_idx)
                || self.tokens[left_idx].next_text_idx != Some(right_idx)
            {
                continue;
            }

            let left_tok = self.tokens[left_idx].token_id;
            let right_tok = self.tokens[right_idx].token_id;
            let Some((rank, merge_token)) = vocab.lookup_pair_merge(&(left_tok, right_tok)) else {
                self.token_to_pair[left_idx] = None;
                continue;
            };

            if rank as usize != pair.rank {
                self.token_to_pair[left_idx] = None;
                continue;
            }

            self.token_to_pair[left_idx] = None;
            let left_prev = self.tokens[left_idx].prev_text_idx;
            let right_next = self.tokens[right_idx].next_text_idx;

            self.deactivate_pair_starting_at(left_prev);
            self.deactivate_pair_starting_at(Some(right_idx));

            self.tokens[left_idx].token_id = merge_token;
            self.tokens[left_idx].next_text_idx = right_next;
            self.tokens[right_idx].is_active = false;
            self.tokens[right_idx].prev_text_idx = None;
            self.tokens[right_idx].next_text_idx = None;

            if let Some(next_idx) = right_next {
                self.tokens[next_idx].prev_text_idx = Some(left_idx);
            }

            if let Some(prev_idx) = left_prev {
                self.activate_pair_starting_at(vocab, prev_idx);
            }
            if right_next.is_some() {
                self.activate_pair_starting_at(vocab, left_idx);
            }
        }

        let mut idx = Some(0usize);
        while let Some(token_idx) = idx {
            tokens.push(self.tokens[token_idx].token_id);
            idx = self.tokens[token_idx].next_text_idx;
        }
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