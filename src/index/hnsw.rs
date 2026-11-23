use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::index::distance::{similarity, DistanceMetric};

#[derive(Clone, Copy)]
struct Candidate {
    id: u64,
    sim: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.sim.to_bits() == other.sim.to_bits()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim.total_cmp(&other.sim).then_with(|| self.id.cmp(&other.id))
    }
}

#[derive(Clone, Copy)]
struct MinCandidate(Candidate);

impl PartialEq for MinCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for MinCandidate {}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HnswNode {
    pub(crate) id: u64,
    pub(crate) vector: Vec<f32>,
    #[allow(dead_code)]
    pub(crate) level: usize,
    pub(crate) neighbors: Vec<Vec<u64>>,
}

#[derive(Debug, Clone)]
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 128,
            ef_search: 64,
            metric: DistanceMetric::Cosine,
        }
    }
}

pub struct HnswIndex {
    pub(crate) config: HnswConfig,
    pub(crate) nodes: HashMap<u64, HnswNode>,
    pub(crate) entry_point: Option<u64>,
    pub(crate) max_level: usize,
    pub(crate) rng_state: u64,
}

impl HnswIndex {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            entry_point: None,
            max_level: 0,
            rng_state: 0x9E3779B97F4A7C15,
        }
    }

    pub(crate) fn from_raw(
        config: HnswConfig,
        nodes: HashMap<u64, HnswNode>,
        entry_point: Option<u64>,
        max_level: usize,
        rng_state: u64,
    ) -> Self {
        Self {
            config,
            nodes,
            entry_point,
            max_level,
            rng_state,
        }
    }

    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn has_vector(&self, id: u64) -> bool {
        self.nodes.contains_key(&id)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    fn random_level(&mut self) -> usize {
        let m = self.config.m.max(2) as f64;
        let mult = 1.0 / m.ln();
        let u = ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
        let u_clamped = u.clamp(1e-12, 1.0 - 1e-12);
        ((-u_clamped.ln()) * mult).floor() as usize
    }

    fn sim_between(&self, a: &[f32], b: &[f32]) -> f32 {
        similarity(self.config.metric, a, b)
    }

    fn search_layer(&self, query: &[f32], entry_id: u64, ef: usize, layer: usize) -> Vec<(u64, f32)> {
        let Some(entry_node) = self.nodes.get(&entry_id) else {
            return Vec::new();
        };

        let mut visited: HashSet<u64> = HashSet::new();
        visited.insert(entry_id);

        let entry_sim = self.sim_between(query, &entry_node.vector);
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<MinCandidate> = BinaryHeap::new();

        let c = Candidate {
            id: entry_id,
            sim: entry_sim,
        };
        candidates.push(c);
        results.push(MinCandidate(c));

        while let Some(curr) = candidates.pop() {
            let worst_result_sim = results.peek().map(|r| r.0.sim).unwrap_or(f32::NEG_INFINITY);
            if results.len() >= ef && curr.sim < worst_result_sim {
                break;
            }

            if let Some(curr_node) = self.nodes.get(&curr.id) {
                if layer < curr_node.neighbors.len() {
                    for &nbr_id in &curr_node.neighbors[layer] {
                        if !visited.insert(nbr_id) {
                            continue;
                        }
                        if let Some(nbr_node) = self.nodes.get(&nbr_id) {
                            let nbr_sim = self.sim_between(query, &nbr_node.vector);
                            let worst_sim = results.peek().map(|r| r.0.sim).unwrap_or(f32::NEG_INFINITY);
                            if results.len() < ef || nbr_sim > worst_sim {
                                let cand = Candidate {
                                    id: nbr_id,
                                    sim: nbr_sim,
                                };
                                candidates.push(cand);
                                results.push(MinCandidate(cand));
                                if results.len() > ef {
                                    results.pop();
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u64, f32)> = results.into_iter().map(|m| (m.0.id, m.0.sim)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        if let Some(existing) = self.nodes.get_mut(&id) {
            existing.vector = vector;
            return;
        }

        let level = self.random_level();
        let max_conn_upper = self.config.m.max(1);
        let max_conn_zero = (self.config.m * 2).max(1);

        let mut neighbors = Vec::with_capacity(level + 1);
        for l in 0..=level {
            let cap = if l == 0 { max_conn_zero } else { max_conn_upper };
            neighbors.push(Vec::with_capacity(cap));
        }

        let new_node = HnswNode {
            id,
            vector: vector.clone(),
            level,
            neighbors,
        };

        let Some(mut curr_ep) = self.entry_point else {
            self.nodes.insert(id, new_node);
            self.entry_point = Some(id);
            self.max_level = level;
            return;
        };

        self.nodes.insert(id, new_node);

        if self.max_level > level {
            for l in (level + 1..=self.max_level).rev() {
                let top = self.search_layer(&vector, curr_ep, 1, l);
                if let Some(&(best_id, _)) = top.first() {
                    curr_ep = best_id;
                }
            }
        }

        let top_layer_to_connect = level.min(self.max_level);
        for l in (0..=top_layer_to_connect).rev() {
            let max_conn = if l == 0 { max_conn_zero } else { max_conn_upper };
            let candidates = self.search_layer(&vector, curr_ep, self.config.ef_construction, l);
            let selected: Vec<u64> = candidates.iter().take(max_conn).map(|&(nid, _)| nid).collect();

            if let Some(node_mut) = self.nodes.get_mut(&id) {
                node_mut.neighbors[l] = selected.clone();
            }

            for &nbr_id in &selected {
                if nbr_id == id {
                    continue;
                }
                let nbr_vec = match self.nodes.get(&nbr_id) {
                    Some(n) => n.vector.clone(),
                    None => continue,
                };
                let current_nbrs = self
                    .nodes
                    .get(&nbr_id)
                    .and_then(|n| n.neighbors.get(l).cloned());

                if let Some(mut nbr_list) = current_nbrs {
                    if !nbr_list.contains(&id) {
                        nbr_list.push(id);
                    }
                    if nbr_list.len() > max_conn {
                        let mut scored: Vec<(u64, f32)> = nbr_list
                            .iter()
                            .filter_map(|&other_id| {
                                self.nodes
                                    .get(&other_id)
                                    .map(|on| (other_id, self.sim_between(&nbr_vec, &on.vector)))
                            })
                            .collect();
                        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                        nbr_list = scored.into_iter().take(max_conn).map(|(oid, _)| oid).collect();
                    }
                    if let Some(nbr_mut) = self.nodes.get_mut(&nbr_id) {
                        if l < nbr_mut.neighbors.len() {
                            nbr_mut.neighbors[l] = nbr_list;
                        }
                    }
                }
            }

            if let Some(&(best_id, _)) = candidates.first() {
                curr_ep = best_id;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id);
        }
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let Some(mut curr_ep) = self.entry_point else {
            return Vec::new();
        };
        if top_k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }

        for l in (1..=self.max_level).rev() {
            let top = self.search_layer(query, curr_ep, 1, l);
            if let Some(&(best_id, _)) = top.first() {
                curr_ep = best_id;
            }
        }

        let ef = self.config.ef_search.max(top_k);
        let mut results = self.search_layer(query, curr_ep, ef, 0);
        results.truncate(top_k);
        results
    }

    pub fn brute_force_search(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        if top_k == 0 || self.nodes.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(u64, f32)> = self
            .nodes
            .values()
            .map(|n| (n.id, self.sim_between(query, &n.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(top_k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_recall_against_brute_force() {
        let cfg = HnswConfig {
            m: 16,
            ef_construction: 120,
            ef_search: 80,
            metric: DistanceMetric::Cosine,
        };
        let mut index = HnswIndex::new(cfg);

        let dim = 32;
        let num_items = 600;
        let mut seed: u64 = 0x123456789ABCDEF0;

        let mut next_f32 = || -> f32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (((seed >> 40) as f32) / 8388608.0) - 1.0
        };

        for id in 1..=num_items {
            let vec: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
            index.insert(id, vec);
        }

        let query: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        let top_k = 10;
        let ann_hits = index.search(&query, top_k);
        let exact_hits = index.brute_force_search(&query, top_k);

        let exact_set: HashSet<u64> = exact_hits.iter().map(|(id, _)| *id).collect();
        let matched = ann_hits.iter().filter(|(id, _)| exact_set.contains(id)).count();
        let recall = matched as f32 / top_k as f32;

        assert_eq!(ann_hits.len(), top_k);
        assert!(recall >= 0.70, "Expected decent recall, got {}", recall);
    }
}
