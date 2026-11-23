pub mod index;
pub mod query;
pub mod server;
pub mod storage;

use serde_json_lite::escape_json;

#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub id: u64,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub id: u64,
    pub text: String,
    pub score: f32,
    pub vector_score: Option<f32>,
    pub bm25_score: Option<f32>,
}

impl SearchHit {
    pub fn to_json(&self) -> String {
        let vec_str = match self.vector_score {
            Some(v) => format!("{:.6}", v),
            None => "null".to_string(),
        };
        let bm25_str = match self.bm25_score {
            Some(b) => format!("{:.6}", b),
            None => "null".to_string(),
        };
        format!(
            r#"{{"id":{},"text":"{}","score":{:.6},"vector_score":{},"bm25_score":{}}}"#,
            self.id,
            escape_json(&self.text),
            self.score,
            vec_str,
            bm25_str
        )
    }
}

pub mod serde_json_lite {
    pub fn escape_json(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    pub fn extract_json_string(body: &str, key: &str) -> Option<String> {
        let pattern = format!(r#""{}""#, key);
        let start_key = body.find(&pattern)?;
        let after_key = &body[start_key + pattern.len()..];
        let colon = after_key.find(':')?;
        let after_colon = after_key[colon + 1..].trim_start();
        if !after_colon.starts_with('"') {
            return None;
        }
        let rest = &after_colon[1..];
        let mut out = String::new();
        let mut chars = rest.chars();
        while let Some(c) = chars.next() {
            match c {
                '"' => return Some(out),
                '\\' => {
                    if let Some(esc) = chars.next() {
                        match esc {
                            '"' => out.push('"'),
                            '\\' => out.push('\\'),
                            '/' => out.push('/'),
                            'n' => out.push('\n'),
                            'r' => out.push('\r'),
                            't' => out.push('\t'),
                            other => out.push(other),
                        }
                    }
                }
                other => out.push(other),
            }
        }
        None
    }

    pub fn extract_usize(body: &str, key: &str) -> Option<usize> {
        let pattern = format!(r#""{}""#, key);
        let start_key = body.find(&pattern)?;
        let after_key = &body[start_key + pattern.len()..];
        let colon = after_key.find(':')?;
        let after_colon = after_key[colon + 1..].trim_start();
        let num_str: String = after_colon
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        num_str.parse::<usize>().ok()
    }

    pub fn extract_f32_array(body: &str, key: &str) -> Option<Vec<f32>> {
        let pattern = format!(r#""{}""#, key);
        let start_key = body.find(&pattern)?;
        let after_key = &body[start_key + pattern.len()..];
        let bracket_start = after_key.find('[')?;
        let bracket_end = after_key[bracket_start..].find(']')? + bracket_start;
        let inner = &after_key[bracket_start + 1..bracket_end];
        if inner.trim().is_empty() {
            return Some(Vec::new());
        }
        let mut vec = Vec::new();
        for part in inner.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            let val = trimmed.parse::<f32>().ok()?;
            if !val.is_finite() {
                return None;
            }
            vec.push(val);
        }
        Some(vec)
    }

    pub fn parse_ingest_request(body: &str) -> Result<(u64, String, Vec<f32>), String> {
        let id = extract_usize(body, "id").ok_or_else(|| "Missing or invalid 'id'".to_string())? as u64;
        let text = extract_json_string(body, "text").ok_or_else(|| "Missing or invalid 'text'".to_string())?;
        let vector = extract_f32_array(body, "vector").ok_or_else(|| "Missing or invalid 'vector'".to_string())?;
        if vector.is_empty() {
            return Err("'vector' must not be empty".to_string());
        }
        Ok((id, text, vector))
    }

    pub fn parse_search_request(body: &str) -> Result<(String, Vec<f32>, usize), String> {
        let query_text = extract_json_string(body, "query").unwrap_or_default();
        let query_vector = extract_f32_array(body, "vector").unwrap_or_default();
        let top_k = extract_usize(body, "top_k").unwrap_or(5).max(1);
        Ok((query_text, query_vector, top_k))
    }
}
