use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Bm25Config {
    pub k1: f32,
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                current.push(lc);
            }
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

pub struct InvertedIndex {
    pub(crate) config: Bm25Config,
    pub(crate) doc_lengths: HashMap<u64, usize>,
    pub(crate) total_tokens: usize,
    pub(crate) postings: HashMap<String, HashMap<u64, usize>>,
}

impl InvertedIndex {
    pub fn new(config: Bm25Config) -> Self {
        Self {
            config,
            doc_lengths: HashMap::new(),
            total_tokens: 0,
            postings: HashMap::new(),
        }
    }

    pub(crate) fn from_raw(
        config: Bm25Config,
        doc_lengths: HashMap<u64, usize>,
        total_tokens: usize,
        postings: HashMap<String, HashMap<u64, usize>>,
    ) -> Self {
        Self {
            config,
            doc_lengths,
            total_tokens,
            postings,
        }
    }

    pub fn config(&self) -> &Bm25Config {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.doc_lengths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_lengths.is_empty()
    }

    pub fn index_document(&mut self, doc_id: u64, text: &str) {
        if self.doc_lengths.contains_key(&doc_id) {
            self.remove_document(doc_id);
        }

        let tokens = tokenize(text);
        let doc_len = tokens.len();
        self.doc_lengths.insert(doc_id, doc_len);
        self.total_tokens += doc_len;

        let mut tf_map: HashMap<String, usize> = HashMap::new();
        for tok in tokens {
            *tf_map.entry(tok).or_insert(0) += 1;
        }

        for (term, tf) in tf_map {
            self.postings
                .entry(term)
                .or_default()
                .insert(doc_id, tf);
        }
    }

    pub fn remove_document(&mut self, doc_id: u64) {
        if let Some(old_len) = self.doc_lengths.remove(&doc_id) {
            self.total_tokens = self.total_tokens.saturating_sub(old_len);
            let mut empty_terms = Vec::new();
            for (term, doc_tfs) in self.postings.iter_mut() {
                doc_tfs.remove(&doc_id);
                if doc_tfs.is_empty() {
                    empty_terms.push(term.clone());
                }
            }
            for term in empty_terms {
                self.postings.remove(&term);
            }
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Vec<(u64, f32)> {
        if top_k == 0 || self.doc_lengths.is_empty() {
            return Vec::new();
        }

        let query_terms: HashSet<String> = tokenize(query).into_iter().collect();
        if query_terms.is_empty() {
            return Vec::new();
        }

        let n_docs = self.doc_lengths.len() as f32;
        let avgdl = (self.total_tokens as f32 / n_docs).max(1.0);
        let k1 = self.config.k1;
        let b = self.config.b;

        let mut scores: HashMap<u64, f32> = HashMap::new();

        for term in query_terms {
            let Some(doc_tfs) = self.postings.get(&term) else {
                continue;
            };
            let df = doc_tfs.len() as f32;
            let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();

            for (&doc_id, &tf_usize) in doc_tfs {
                let tf = tf_usize as f32;
                let dl = *self.doc_lengths.get(&doc_id).unwrap_or(&1) as f32;
                let denom = tf + k1 * (1.0 - b + b * (dl / avgdl));
                let score = idf * ((tf * (k1 + 1.0)) / denom.max(1e-6));
                *scores.entry(doc_id).or_insert(0.0) += score;
            }
        }

        let mut ranked: Vec<(u64, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(top_k);
        ranked
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_ranking() {
        let mut idx = InvertedIndex::new(Bm25Config::default());
        idx.index_document(1, "Rust systems programming with low latency and memory safety");
        idx.index_document(2, "Cooking pasta carbonara with pecorino romano and guanciale");
        idx.index_document(3, "Distributed consensus raft protocol and rust networking");

        let hits = idx.search("rust latency", 2);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 1);
    }
}
