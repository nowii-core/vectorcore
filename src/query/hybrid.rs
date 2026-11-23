use std::collections::HashMap;

use crate::index::hnsw::HnswIndex;
use crate::index::inverted::InvertedIndex;
use crate::SearchHit;

#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub rrf_k: f32,
    pub vector_weight: f32,
    pub bm25_weight: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            rrf_k: 60.0,
            vector_weight: 1.0,
            bm25_weight: 1.0,
        }
    }
}

pub struct HybridSearchEngine {
    pub hnsw: HnswIndex,
    pub inverted: InvertedIndex,
    pub documents: HashMap<u64, String>,
    pub config: HybridConfig,
}

impl HybridSearchEngine {
    pub fn new(hnsw: HnswIndex, inverted: InvertedIndex, config: HybridConfig) -> Self {
        Self {
            hnsw,
            inverted,
            documents: HashMap::new(),
            config,
        }
    }

    pub fn from_parts(
        hnsw: HnswIndex,
        inverted: InvertedIndex,
        documents: HashMap<u64, String>,
        config: HybridConfig,
    ) -> Self {
        Self {
            hnsw,
            inverted,
            documents,
            config,
        }
    }

    pub fn upsert(&mut self, id: u64, text: String, vector: Vec<f32>) {
        self.documents.insert(id, text.clone());
        self.inverted.index_document(id, &text);
        self.hnsw.insert(id, vector);
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn search(&self, query_text: &str, query_vector: &[f32], top_k: usize) -> Vec<SearchHit> {
        if top_k == 0 || self.documents.is_empty() {
            return Vec::new();
        }

        let candidate_k = (top_k * 4).max(20);

        let vec_results = if !query_vector.is_empty() && !self.hnsw.is_empty() {
            self.hnsw.search(query_vector, candidate_k)
        } else {
            Vec::new()
        };

        let bm25_results = if !query_text.trim().is_empty() && !self.inverted.is_empty() {
            self.inverted.search(query_text, candidate_k)
        } else {
            Vec::new()
        };

        struct Acc {
            rrf_score: f32,
            vec_score: Option<f32>,
            bm25_score: Option<f32>,
        }

        let mut merged: HashMap<u64, Acc> = HashMap::new();

        for (rank_idx, (doc_id, sim)) in vec_results.into_iter().enumerate() {
            let rank = (rank_idx + 1) as f32;
            let contrib = self.config.vector_weight * (1.0 / (self.config.rrf_k + rank));
            let entry = merged.entry(doc_id).or_insert(Acc {
                rrf_score: 0.0,
                vec_score: None,
                bm25_score: None,
            });
            entry.rrf_score += contrib;
            entry.vec_score = Some(sim);
        }

        for (rank_idx, (doc_id, bm25)) in bm25_results.into_iter().enumerate() {
            let rank = (rank_idx + 1) as f32;
            let contrib = self.config.bm25_weight * (1.0 / (self.config.rrf_k + rank));
            let entry = merged.entry(doc_id).or_insert(Acc {
                rrf_score: 0.0,
                vec_score: None,
                bm25_score: None,
            });
            entry.rrf_score += contrib;
            entry.bm25_score = Some(bm25);
        }

        let mut hits: Vec<SearchHit> = merged
            .into_iter()
            .filter_map(|(id, acc)| {
                let text = self.documents.get(&id)?.clone();
                Some(SearchHit {
                    id,
                    text,
                    score: acc.rrf_score,
                    vector_score: acc.vec_score,
                    bm25_score: acc.bm25_score,
                })
            })
            .collect();

        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        hits.truncate(top_k);
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::hnsw::HnswConfig;
    use crate::index::inverted::Bm25Config;

    #[test]
    fn test_hybrid_rrf_search() {
        let hnsw = HnswIndex::new(HnswConfig::default());
        let inverted = InvertedIndex::new(Bm25Config::default());
        let mut engine = HybridSearchEngine::new(hnsw, inverted, HybridConfig::default());

        engine.upsert(1, "Vector search with HNSW graph in Rust".to_string(), vec![1.0, 0.0, 0.0]);
        engine.upsert(2, "Write-ahead log durability and CRC32 checksums".to_string(), vec![0.0, 1.0, 0.0]);
        engine.upsert(3, "Hybrid search combining BM25 and dense embeddings".to_string(), vec![0.9, 0.1, 0.0]);

        let hits = engine.search("HNSW Rust", &[1.0, 0.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 1);
    }
}
