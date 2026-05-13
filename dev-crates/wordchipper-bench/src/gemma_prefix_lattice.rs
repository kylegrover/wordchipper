use std::{
    collections::HashMap,
    sync::Arc,
};

use wordchipper::{
    UnifiedTokenVocab,
    VocabIndex,
    spanners::{
        SpanRef,
        TextSpanner,
        TextSpannerBuilder,
    },
};

#[derive(Clone, Copy)]
struct BuildEdge {
    byte: u8,
    next: u32,
}

#[derive(Default)]
struct BuildNode {
    edges: Vec<BuildEdge>,
    outputs: Vec<LatticeOutput>,
}

#[derive(Clone, Copy)]
struct TrieNode {
    edge_start: u32,
    edge_len: u32,
    output_start: u32,
    output_len: u32,
}

#[derive(Clone, Copy)]
struct TrieEdge {
    byte: u8,
    next: u32,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct LatticeOutput {
    token: u32,
    rank: u32,
}

#[derive(Clone, Copy)]
struct LatticeEdge {
    end: u32,
    token: u32,
    rank: u32,
}

fn find_or_insert_child(
    nodes: &mut Vec<BuildNode>,
    node: u32,
    byte: u8,
) -> u32 {
    if let Some(next) = nodes[node as usize]
        .edges
        .iter()
        .find(|edge| edge.byte == byte)
        .map(|edge| edge.next)
    {
        return next;
    }

    let next = nodes.len() as u32;
    nodes.push(BuildNode::default());
    nodes[node as usize].edges.push(BuildEdge { byte, next });
    next
}

/// Bench-only flat prefix lattice scanner for Gemma-like explicit-rank vocabs.
pub struct GemmaPrefixLatticeScanner {
    spanner: Arc<dyn TextSpanner>,
    nodes: Vec<TrieNode>,
    trie_edges: Vec<TrieEdge>,
    outputs: Vec<LatticeOutput>,
    row_starts: Vec<u32>,
    edges: Vec<LatticeEdge>,
}

impl GemmaPrefixLatticeScanner {
    /// Build a scanner from a loaded vocabulary.
    pub fn from_vocab(vocab: &UnifiedTokenVocab<u32>) -> Self {
        let mut merge_ranks = HashMap::with_capacity(vocab.pair_vocab().pair_map().len());
        for (&pair, &token) in vocab.pair_vocab().pair_map() {
            let (rank, merge_token) = vocab.lookup_pair_merge(&pair).unwrap();
            debug_assert_eq!(merge_token, token);
            merge_ranks.insert(token, rank);
        }

        let mut span_pairs: Vec<(Vec<u8>, u32, u32)> = vocab
            .pair_vocab()
            .span_pairs()
            .map(|(span, token)| (span, token, *merge_ranks.get(&token).unwrap_or(&u32::MAX)))
            .collect();

        span_pairs.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.2.cmp(&right.2))
                .then(left.1.cmp(&right.1))
        });

        let mut build_nodes = vec![BuildNode::default()];
        for (span, token, rank) in span_pairs {
            let mut node = 0u32;
            for &byte in &span {
                node = find_or_insert_child(&mut build_nodes, node, byte);
            }
            build_nodes[node as usize]
                .outputs
                .push(LatticeOutput { token, rank });
        }

        let mut nodes = Vec::with_capacity(build_nodes.len());
        let mut trie_edges = Vec::new();
        let mut outputs = Vec::new();

        for build_node in &mut build_nodes {
            build_node.edges.sort_by_key(|edge| edge.byte);
            build_node.outputs.sort();

            nodes.push(TrieNode {
                edge_start: trie_edges.len() as u32,
                edge_len: build_node.edges.len() as u32,
                output_start: outputs.len() as u32,
                output_len: build_node.outputs.len() as u32,
            });

            trie_edges.extend(build_node.edges.iter().map(|edge| TrieEdge {
                byte: edge.byte,
                next: edge.next,
            }));
            outputs.extend(build_node.outputs.iter().copied());
        }

        Self {
            spanner: TextSpannerBuilder::default(vocab),
            nodes,
            trie_edges,
            outputs,
            row_starts: Vec::new(),
            edges: Vec::new(),
        }
    }

    fn find_child(
        &self,
        node: u32,
        byte: u8,
    ) -> Option<u32> {
        let node = self.nodes[node as usize];
        let start = node.edge_start as usize;
        let end = start + node.edge_len as usize;
        let edges = &self.trie_edges[start..end];
        let index = edges.binary_search_by_key(&byte, |edge| edge.byte).ok()?;
        Some(edges[index].next)
    }

    /// Pre-grow the working buffers so benchmark iterations avoid one-time allocations.
    pub fn warm_text(
        &mut self,
        text: &str,
    ) {
        let _ = self.scan_text(text);
    }

    /// Scan a single compound span and materialize a flat DAG.
    pub fn scan_span(
        &mut self,
        span: &[u8],
    ) -> usize {
        self.row_starts.clear();
        self.edges.clear();

        self.row_starts.reserve(span.len().saturating_add(1));

        for start in 0..span.len() {
            self.row_starts.push(self.edges.len() as u32);

            let mut node = 0u32;
            let mut end = start;

            while end < span.len() {
                let Some(next) = self.find_child(node, span[end]) else {
                    break;
                };

                node = next;
                end += 1;

                let node_info = self.nodes[node as usize];
                let out_start = node_info.output_start as usize;
                let out_end = out_start + node_info.output_len as usize;
                for output in &self.outputs[out_start..out_end] {
                    self.edges.push(LatticeEdge {
                        end: end as u32,
                        token: output.token,
                        rank: output.rank,
                    });
                }
            }
        }

        self.row_starts.push(self.edges.len() as u32);
        self.edges.len()
    }

    /// Scan all normal word spans in a text using the same splitter as the encoder.
    pub fn scan_text(
        &mut self,
        text: &str,
    ) -> usize {
        let mut total_edges = 0usize;
        let mut checksum = 0u64;
        let spanner = self.spanner.clone();

        spanner.for_each_split_span(text, None, &mut |span_ref| {
            if let SpanRef::Word(range) = span_ref {
                total_edges += self.scan_span(text[range].as_bytes());
                checksum = self.edges.iter().fold(checksum, |acc, edge| {
                    acc ^ ((edge.end as u64) << 32) ^ edge.rank as u64 ^ edge.token as u64
                });
            }
            true
        });

        total_edges ^ checksum as usize
    }
}