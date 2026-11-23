use vectorcore::index::distance::DistanceMetric;
use vectorcore::index::hnsw::{HnswConfig, HnswIndex};
use vectorcore::index::inverted::{Bm25Config, InvertedIndex};
use vectorcore::query::hybrid::{HybridConfig, HybridSearchEngine};
use vectorcore::storage::snapshot::{load_snapshot, save_snapshot};
use vectorcore::storage::wal::{WalEntry, WriteAheadLog};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut wal_path = std::env::temp_dir();
    wal_path.push("vectorcore_demo.wal");
    let _ = std::fs::remove_file(&wal_path);

    let mut snap_path = std::env::temp_dir();
    snap_path.push("vectorcore_demo.vcsnap");
    let _ = std::fs::remove_file(&snap_path);

    let hnsw = HnswIndex::new(HnswConfig {
        m: 16,
        ef_construction: 100,
        ef_search: 50,
        metric: DistanceMetric::Cosine,
    });
    let inverted = InvertedIndex::new(Bm25Config::default());
    let mut engine = HybridSearchEngine::new(hnsw, inverted, HybridConfig::default());

    let mut wal = WriteAheadLog::open(&wal_path)?;

    let sample_docs = vec![
        (1, "HNSW graph enables sub-millisecond approximate nearest neighbor search", vec![0.95, 0.10, 0.05, 0.0]),
        (2, "Raft consensus replicates logs deterministically across distributed nodes", vec![0.10, 0.90, 0.20, 0.0]),
        (3, "Write-ahead logging with CRC32 prevents data loss after sudden crashes", vec![0.20, 0.85, 0.30, 0.0]),
        (4, "Okapi BM25 scores lexical term matches while RRF merges ranking signals", vec![0.80, 0.25, 0.15, 0.1]),
    ];

    for (id, text, vec) in &sample_docs {
        wal.append(&WalEntry {
            id: *id,
            text: text.to_string(),
            vector: vec.clone(),
        })?;
        engine.upsert(*id, text.to_string(), vec.clone());
    }
    wal.sync()?;

    let replayed = WriteAheadLog::replay(&wal_path)?;
    println!("Replayed {} entries from WAL at {:?}", replayed.len(), wal_path);

    save_snapshot(&engine, &snap_path)?;
    let loaded_engine = load_snapshot(&snap_path)?;
    println!("Saved and restored snapshot with {} docs at {:?}", loaded_engine.len(), snap_path);

    let hits = loaded_engine.search("HNSW search", &[0.92, 0.12, 0.08, 0.0], 3);
    println!("Top hybrid search hits:");
    for hit in hits {
        println!(
            "  [id={}] score={:.5} (vec={:?}, bm25={:?}) -> {}",
            hit.id, hit.score, hit.vector_score, hit.bm25_score, hit.text
        );
    }

    let _ = std::fs::remove_file(&wal_path);
    let _ = std::fs::remove_file(&snap_path);
    Ok(())
}
