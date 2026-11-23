use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::index::distance::DistanceMetric;
use crate::index::hnsw::{HnswConfig, HnswIndex, HnswNode};
use crate::index::inverted::{Bm25Config, InvertedIndex};
use crate::query::hybrid::{HybridConfig, HybridSearchEngine};
use crate::storage::wal::crc32_ieee;

const SNAP_MAGIC: &[u8; 4] = b"VCS1";

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let b = s.as_bytes();
    write_u32(w, b.len() as u32)?;
    w.write_all(b)
}

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    Ok(f32::from_bits(read_u32(r)?))
}

fn read_str<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn save_snapshot<P: AsRef<Path>>(engine: &HybridSearchEngine, path: P) -> io::Result<()> {
    let mut payload: Vec<u8> = Vec::new();

    // 1. HybridConfig
    write_f32(&mut payload, engine.config.rrf_k)?;
    write_f32(&mut payload, engine.config.vector_weight)?;
    write_f32(&mut payload, engine.config.bm25_weight)?;

    // 2. Documents
    write_u64(&mut payload, engine.documents.len() as u64)?;
    for (&doc_id, text) in &engine.documents {
        write_u64(&mut payload, doc_id)?;
        write_str(&mut payload, text)?;
    }

    // 3. HNSW
    write_u32(&mut payload, engine.hnsw.config.m as u32)?;
    write_u32(&mut payload, engine.hnsw.config.ef_construction as u32)?;
    write_u32(&mut payload, engine.hnsw.config.ef_search as u32)?;
    write_u8(&mut payload, engine.hnsw.config.metric.to_u8())?;

    match engine.hnsw.entry_point {
        Some(ep) => {
            write_u8(&mut payload, 1)?;
            write_u64(&mut payload, ep)?;
        }
        None => write_u8(&mut payload, 0)?,
    }
    write_u32(&mut payload, engine.hnsw.max_level as u32)?;
    write_u64(&mut payload, engine.hnsw.rng_state)?;

    write_u64(&mut payload, engine.hnsw.nodes.len() as u64)?;
    for (&nid, node) in &engine.hnsw.nodes {
        write_u64(&mut payload, nid)?;
        write_u32(&mut payload, node.vector.len() as u32)?;
        for &v in &node.vector {
            write_f32(&mut payload, v)?;
        }
        write_u32(&mut payload, node.level as u32)?;
        write_u32(&mut payload, node.neighbors.len() as u32)?;
        for layer_nbrs in &node.neighbors {
            write_u32(&mut payload, layer_nbrs.len() as u32)?;
            for &nbr_id in layer_nbrs {
                write_u64(&mut payload, nbr_id)?;
            }
        }
    }

    // 4. InvertedIndex (BM25)
    write_f32(&mut payload, engine.inverted.config.k1)?;
    write_f32(&mut payload, engine.inverted.config.b)?;
    write_u64(&mut payload, engine.inverted.total_tokens as u64)?;

    write_u64(&mut payload, engine.inverted.doc_lengths.len() as u64)?;
    for (&doc_id, &len) in &engine.inverted.doc_lengths {
        write_u64(&mut payload, doc_id)?;
        write_u64(&mut payload, len as u64)?;
    }

    write_u64(&mut payload, engine.inverted.postings.len() as u64)?;
    for (term, doc_tfs) in &engine.inverted.postings {
        write_str(&mut payload, term)?;
        write_u64(&mut payload, doc_tfs.len() as u64)?;
        for (&doc_id, &tf) in doc_tfs {
            write_u64(&mut payload, doc_id)?;
            write_u64(&mut payload, tf as u64)?;
        }
    }

    let path_ref = path.as_ref();
    if let Some(parent) = path_ref.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path_ref)?;
    let mut writer = BufWriter::new(file);
    let checksum = crc32_ieee(&payload);

    writer.write_all(SNAP_MAGIC)?;
    write_u64(&mut writer, payload.len() as u64)?;
    write_u32(&mut writer, checksum)?;
    writer.write_all(&payload)?;
    writer.flush()?;
    writer.get_ref().sync_data()?;
    Ok(())
}

pub fn load_snapshot<P: AsRef<Path>>(path: P) -> io::Result<HybridSearchEngine> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != SNAP_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid snapshot magic header"));
    }

    let payload_len = read_u64(&mut reader)? as usize;
    let expected_crc = read_u32(&mut reader)?;

    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;

    let actual_crc = crc32_ieee(&payload);
    if actual_crc != expected_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Snapshot CRC32 mismatch: expected {:#x}, got {:#x}", expected_crc, actual_crc),
        ));
    }

    let mut cursor: &[u8] = &payload;
    let cur = &mut cursor;

    let rrf_k = read_f32(cur)?;
    let vector_weight = read_f32(cur)?;
    let bm25_weight = read_f32(cur)?;
    let hybrid_cfg = HybridConfig {
        rrf_k,
        vector_weight,
        bm25_weight,
    };

    let doc_count = read_u64(cur)? as usize;
    let mut documents = HashMap::with_capacity(doc_count);
    for _ in 0..doc_count {
        let doc_id = read_u64(cur)?;
        let text = read_str(cur)?;
        documents.insert(doc_id, text);
    }

    let m = read_u32(cur)? as usize;
    let ef_construction = read_u32(cur)? as usize;
    let ef_search = read_u32(cur)? as usize;
    let metric_byte = read_u8(cur)?;
    let metric = DistanceMetric::from_u8(metric_byte).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("Invalid metric byte: {}", metric_byte))
    })?;
    let hnsw_cfg = HnswConfig {
        m,
        ef_construction,
        ef_search,
        metric,
    };

    let has_ep = read_u8(cur)?;
    let entry_point = if has_ep != 0 {
        Some(read_u64(cur)?)
    } else {
        None
    };
    let max_level = read_u32(cur)? as usize;
    let rng_state = read_u64(cur)?;

    let node_count = read_u64(cur)? as usize;
    let mut nodes = HashMap::with_capacity(node_count);
    for _ in 0..node_count {
        let nid = read_u64(cur)?;
        let vlen = read_u32(cur)? as usize;
        let mut vector = Vec::with_capacity(vlen);
        for _ in 0..vlen {
            vector.push(read_f32(cur)?);
        }
        let level = read_u32(cur)? as usize;
        let layers_len = read_u32(cur)? as usize;
        let mut neighbors = Vec::with_capacity(layers_len);
        for _ in 0..layers_len {
            let nbr_len = read_u32(cur)? as usize;
            let mut layer_nbrs = Vec::with_capacity(nbr_len);
            for _ in 0..nbr_len {
                layer_nbrs.push(read_u64(cur)?);
            }
            neighbors.push(layer_nbrs);
        }
        nodes.insert(
            nid,
            HnswNode {
                id: nid,
                vector,
                level,
                neighbors,
            },
        );
    }

    let hnsw = HnswIndex::from_raw(hnsw_cfg, nodes, entry_point, max_level, rng_state);

    let k1 = read_f32(cur)?;
    let b = read_f32(cur)?;
    let bm25_cfg = Bm25Config { k1, b };
    let total_tokens = read_u64(cur)? as usize;

    let dl_count = read_u64(cur)? as usize;
    let mut doc_lengths = HashMap::with_capacity(dl_count);
    for _ in 0..dl_count {
        let doc_id = read_u64(cur)?;
        let dlen = read_u64(cur)? as usize;
        doc_lengths.insert(doc_id, dlen);
    }

    let post_count = read_u64(cur)? as usize;
    let mut postings = HashMap::with_capacity(post_count);
    for _ in 0..post_count {
        let term = read_str(cur)?;
        let map_len = read_u64(cur)? as usize;
        let mut doc_tfs = HashMap::with_capacity(map_len);
        for _ in 0..map_len {
            let doc_id = read_u64(cur)?;
            let tf = read_u64(cur)? as usize;
            doc_tfs.insert(doc_id, tf);
        }
        postings.insert(term, doc_tfs);
    }

    let inverted = InvertedIndex::from_raw(bm25_cfg, doc_lengths, total_tokens, postings);
    Ok(HybridSearchEngine::from_parts(hnsw, inverted, documents, hybrid_cfg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::wal::crc32_ieee;
    use std::io::Seek;

    #[test]
    fn test_snapshot_save_load_and_crc_tamper_detection() {
        let mut snap_path = std::env::temp_dir();
        snap_path.push(format!(
            "vectorcore_snap_{}_{}.vcsnap",
            std::process::id(),
            crc32_ieee(b"snap_test")
        ));
        let _ = fs::remove_file(&snap_path);

        let hnsw = HnswIndex::new(HnswConfig::default());
        let inverted = InvertedIndex::new(Bm25Config::default());
        let mut engine = HybridSearchEngine::new(hnsw, inverted, HybridConfig::default());

        engine.upsert(1, "HNSW graph search in Rust".to_string(), vec![1.0, 0.0, 0.0]);
        engine.upsert(2, "BM25 lexical scoring with RRF".to_string(), vec![0.0, 1.0, 0.0]);

        save_snapshot(&engine, &snap_path).unwrap();
        let restored = load_snapshot(&snap_path).unwrap();

        assert_eq!(restored.len(), 2);
        let hits = restored.search("HNSW Rust", &[1.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);

        // Flip a byte inside payload to verify CRC32 detects corruption
        {
            let mut f = fs::OpenOptions::new().read(true).write(true).open(&snap_path).unwrap();
            let file_len = f.metadata().unwrap().len();
            f.seek(std::io::SeekFrom::Start(file_len - 1)).unwrap();
            let mut last = [0u8; 1];
            f.read_exact(&mut last).unwrap();
            last[0] ^= 0xFF;
            f.seek(std::io::SeekFrom::Start(file_len - 1)).unwrap();
            f.write_all(&last).unwrap();
        }

        assert!(load_snapshot(&snap_path).is_err());
        let _ = fs::remove_file(&snap_path);
    }
}
