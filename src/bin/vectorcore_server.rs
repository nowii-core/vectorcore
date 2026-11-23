use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use vectorcore::index::distance::DistanceMetric;
use vectorcore::index::hnsw::{HnswConfig, HnswIndex};
use vectorcore::index::inverted::{Bm25Config, InvertedIndex};
use vectorcore::query::hybrid::{HybridConfig, HybridSearchEngine};
use vectorcore::server::http::{serve, ServerState};
use vectorcore::storage::snapshot::load_snapshot;
use vectorcore::storage::wal::{WalEntry, WriteAheadLog};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snap_path = PathBuf::from("vectorcore.vcsnap");
    let wal_path = PathBuf::from("vectorcore.wal");

    let mut engine = if snap_path.exists() {
        load_snapshot(&snap_path)?
    } else {
        HybridSearchEngine::new(
            HnswIndex::new(HnswConfig {
                m: 16,
                ef_construction: 120,
                ef_search: 64,
                metric: DistanceMetric::Cosine,
            }),
            InvertedIndex::new(Bm25Config::default()),
            HybridConfig::default(),
        )
    };

    let wal_entries = WriteAheadLog::replay(&wal_path)?;
    for entry in wal_entries {
        if !engine.documents.contains_key(&entry.id) {
            let doc = entry.into_document();
            engine.upsert(doc.id, doc.text, doc.vector);
        }
    }

    let wal = WriteAheadLog::open(&wal_path)?;
    let state = Arc::new(Mutex::new(ServerState {
        engine,
        wal: Some(wal),
        snapshot_path: Some(snap_path),
    }));

    serve("127.0.0.1:8080", state)?;
    Ok(())
}
