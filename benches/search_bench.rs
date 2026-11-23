use std::collections::HashSet;
use std::time::Instant;

use vectorcore::index::distance::DistanceMetric;
use vectorcore::index::hnsw::{HnswConfig, HnswIndex};

fn main() {
    let num_vectors = 5_000usize;
    let dim = 64usize;
    let top_k = 10usize;
    let num_queries = 50usize;

    let mut seed: u64 = 0xCAFEBABE12345678;
    let mut next_f32 = || -> f32 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (((seed >> 40) as f32) / 8388608.0) - 1.0
    };

    let mut index = HnswIndex::new(HnswConfig {
        m: 16,
        ef_construction: 120,
        ef_search: 64,
        metric: DistanceMetric::Cosine,
    });

    let t_insert_start = Instant::now();
    for id in 1..=(num_vectors as u64) {
        let v: Vec<f32> = (0..dim).map(|_| next_f32()).collect();
        index.insert(id, v);
    }
    let insert_elapsed = t_insert_start.elapsed();
    println!(
        "Inserted {} vectors (dim={}) in {:.2} ms",
        num_vectors,
        dim,
        insert_elapsed.as_secs_f64() * 1000.0
    );

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dim).map(|_| next_f32()).collect())
        .collect();

    let mut latencies_us: Vec<f64> = Vec::with_capacity(num_queries);
    let mut total_recall = 0.0f64;

    for q in &queries {
        let t0 = Instant::now();
        let ann = index.search(q, top_k);
        let dt_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        latencies_us.push(dt_us);

        let exact = index.brute_force_search(q, top_k);
        let exact_set: HashSet<u64> = exact.into_iter().map(|(id, _)| id).collect();
        let hits = ann.into_iter().filter(|(id, _)| exact_set.contains(id)).count();
        total_recall += (hits as f64) / (top_k as f64);
    }

    latencies_us.sort_by(|a, b| a.total_cmp(b));
    let p50 = latencies_us[num_queries / 2];
    let p95 = latencies_us[((num_queries as f64) * 0.95).floor() as usize];
    let p99 = latencies_us[((num_queries as f64) * 0.99).min((num_queries - 1) as f64).floor() as usize];
    let avg_recall = (total_recall / (num_queries as f64)) * 100.0;

    println!("HNSW Search (top_k={}, ef_search=64):", top_k);
    println!("  Latency p50: {:.1} us | p95: {:.1} us | p99: {:.1} us", p50, p95, p99);
    println!("  Mean Recall@{}: {:.2}%", top_k, avg_recall);
}
