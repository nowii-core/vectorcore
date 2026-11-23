use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::Document;

const WAL_MAGIC: &[u8; 4] = b"VCW1";

pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[derive(Debug, Clone)]
pub struct WalEntry {
    pub id: u64,
    pub text: String,
    pub vector: Vec<f32>,
}

impl WalEntry {
    pub fn from_document(doc: &Document) -> Self {
        Self {
            id: doc.id,
            text: doc.text.clone(),
            vector: doc.vector.clone(),
        }
    }

    pub fn into_document(self) -> Document {
        Document {
            id: self.id,
            text: self.text,
            vector: self.vector,
        }
    }

    fn encode_payload(&self) -> Vec<u8> {
        let text_bytes = self.text.as_bytes();
        let dim = self.vector.len() as u32;
        let text_len = text_bytes.len() as u32;

        let mut payload = Vec::with_capacity(8 + 4 + text_bytes.len() + 4 + self.vector.len() * 4);
        payload.extend_from_slice(&self.id.to_le_bytes());
        payload.extend_from_slice(&text_len.to_le_bytes());
        payload.extend_from_slice(text_bytes);
        payload.extend_from_slice(&dim.to_le_bytes());
        for &val in &self.vector {
            payload.extend_from_slice(&val.to_le_bytes());
        }
        payload
    }

    fn decode_payload(payload: &[u8]) -> io::Result<Self> {
        if payload.len() < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "WAL payload too short"));
        }
        let mut cursor = 0usize;

        let id = u64::from_le_bytes(payload[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;

        let text_len = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        if cursor + text_len + 4 > payload.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "WAL text length out of bounds"));
        }

        let text = String::from_utf8(payload[cursor..cursor + text_len].to_vec())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        cursor += text_len;

        let dim = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        let expected_vec_bytes = dim.checked_mul(4).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "WAL vector dimension overflow")
        })?;
        if cursor + expected_vec_bytes != payload.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "WAL vector byte length mismatch"));
        }

        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            let bits = u32::from_le_bytes(payload[cursor..cursor + 4].try_into().unwrap());
            vector.push(f32::from_bits(bits));
            cursor += 4;
        }

        Ok(Self { id, text, vector })
    }
}

pub struct WriteAheadLog {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl WriteAheadLog {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        if let Some(parent) = path_buf.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path_buf)?;
        Ok(Self {
            path: path_buf,
            writer: BufWriter::new(file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&mut self, entry: &WalEntry) -> io::Result<()> {
        let payload = entry.encode_payload();
        let payload_len = payload.len() as u32;
        let checksum = crc32_ieee(&payload);

        self.writer.write_all(WAL_MAGIC)?;
        self.writer.write_all(&payload_len.to_le_bytes())?;
        self.writer.write_all(&checksum.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        Ok(())
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    pub fn replay<P: AsRef<Path>>(path: P) -> io::Result<Vec<WalEntry>> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e),
        };
        let mut reader = BufReader::new(file);
        let mut entries = Vec::new();

        loop {
            let mut magic = [0u8; 4];
            match reader.read_exact(&mut magic) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            if &magic != WAL_MAGIC {
                break;
            }

            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() {
                break;
            }
            let payload_len = u32::from_le_bytes(len_buf) as usize;

            let mut crc_buf = [0u8; 4];
            if reader.read_exact(&mut crc_buf).is_err() {
                break;
            }
            let expected_crc = u32::from_le_bytes(crc_buf);

            let mut payload = vec![0u8; payload_len];
            if reader.read_exact(&mut payload).is_err() {
                break;
            }

            if crc32_ieee(&payload) != expected_crc {
                break;
            }

            if let Ok(entry) = WalEntry::decode_payload(&payload) {
                entries.push(entry);
            } else {
                break;
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_append_and_replay_with_partial_tail() {
        let mut temp_path = std::env::temp_dir();
        temp_path.push(format!("vectorcore_test_{}_{}.wal", std::process::id(), crc32_ieee(b"test")));
        let _ = std::fs::remove_file(&temp_path);

        {
            let mut wal = WriteAheadLog::open(&temp_path).unwrap();
            wal.append(&WalEntry {
                id: 10,
                text: "First document".to_string(),
                vector: vec![0.1, 0.2, 0.3],
            })
            .unwrap();
            wal.append(&WalEntry {
                id: 20,
                text: "Second document".to_string(),
                vector: vec![0.4, 0.5, 0.6],
            })
            .unwrap();
            wal.sync().unwrap();
        }

        {
            let mut raw = OpenOptions::new().append(true).open(&temp_path).unwrap();
            raw.write_all(b"VCW1\x10\x00\x00\x00CORRUPT_TAIL").unwrap();
        }

        let replayed = WriteAheadLog::replay(&temp_path).unwrap();
        assert_eq!(replayed.len(), 2);
        assert_eq!(replayed[0].id, 10);
        assert_eq!(replayed[1].id, 20);

        let _ = std::fs::remove_file(&temp_path);
    }
}
