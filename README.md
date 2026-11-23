# VectorCore

A clean, modular, zero-heavy-dependency **Vector + Hybrid Search Engine** written from scratch in pure Rust:

- **Distance & Vector Math (`src/index/distance.rs`)**: Cosine similarity, L2 squared distance, dot product, and L2 norm normalization.
- **HNSW Graph (`src/index/hnsw.rs`)**: Multi-layer Hierarchical Navigable Small World approximate nearest neighbor (ANN) index with configurable `m`, `ef_construction`, and `ef_search`, plus exact brute-force baseline for recall evaluation.
- **BM25 Inverted Index (`src/index/inverted.rs`)**: Tokenization, term frequencies, document lengths, average document length, and Okapi BM25 scoring.
- **Hybrid Fusion (`src/query/hybrid.rs`)**: Reciprocal Rank Fusion (RRF) combining dense vector search and sparse BM25 lexical search.
- **Crash-Safe Write-Ahead Log (`src/storage/wal.rs`)**: Append-only binary WAL (`VCW1`) with magic header, payload length, CRC32 IEEE checksum, `fsync`, and automatic crash recovery.
- **Binary Snapshot Persistence (`src/storage/snapshot.rs`)**: Full engine state serialization (`VCS1`) with CRC32 IEEE verification to save and restore documents, HNSW nodes/neighbors/entry-point/max-level/RNG state, and BM25 index without replaying from scratch.
- **Zero-Dependency HTTP/1.1 Server (`src/server/http.rs`)**: Lightweight `std::net::TcpListener` JSON API for `/health`, `/ingest`, `/search`, and `/snapshot` (save/load).

## Apple Silicon Benchmark (`5,000` vectors, `dim=64`, `top_k=10`, `ef_search=64`)

- **Ingestion**: `5,000` vectors (`dim=64`) in `1456.87 ms`
- **Latency p50**: `181.5 us` (`0.18 ms`)
- **Latency p95**: `243.5 us` (`0.24 ms`)
- **Latency p99**: `332.7 us` (`0.33 ms`)
- **Mean Recall@10**: `94.80%`

## Quick Start

```bash
cargo test
cargo run --release --example semantic_search
cargo bench --bench search_bench
cargo run --release --bin vectorcore_server
```
