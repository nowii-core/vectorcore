use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::query::hybrid::HybridSearchEngine;
use crate::serde_json_lite::{escape_json, parse_ingest_request, parse_search_request};
use crate::storage::snapshot::{load_snapshot, save_snapshot};
use crate::storage::wal::{WalEntry, WriteAheadLog};

pub struct ServerState {
    pub engine: HybridSearchEngine,
    pub wal: Option<WriteAheadLog>,
    pub snapshot_path: Option<PathBuf>,
}

pub fn serve(addr: &str, state: Arc<Mutex<ServerState>>) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    println!("VectorCore HTTP server listening on http://{}", addr);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let st = Arc::clone(&state);
                if let Err(e) = handle_connection(stream, st) {
                    eprintln!("Connection error: {}", e);
                }
            }
            Err(e) => eprintln!("Accept error: {}", e),
        }
    }
    Ok(())
}

fn handle_connection(mut stream: TcpStream, state: Arc<Mutex<ServerState>>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }

    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        let lower = line.to_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse::<usize>().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    let body_str = String::from_utf8_lossy(&body).to_string();

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");

    let (status, content_type, resp_body) = match (method, path) {
        ("GET", "/health") => {
            let guard = state.lock().unwrap();
            let count = guard.engine.len();
            (
                "200 OK",
                "application/json",
                format!(r#"{{"status":"ok","documents":{}}}"#, count),
            )
        }
        ("POST", "/ingest") => match parse_ingest_request(&body_str) {
            Ok((id, text, vector)) => {
                let mut guard = state.lock().unwrap();
                if let Some(wal) = guard.wal.as_mut() {
                    let entry = WalEntry {
                        id,
                        text: text.clone(),
                        vector: vector.clone(),
                    };
                    if let Err(e) = wal.append(&entry).and_then(|_| wal.sync()) {
                        return write_response(
                            &mut stream,
                            "500 Internal Server Error",
                            "application/json",
                            &format!(r#"{{"error":"WAL write failed: {}"}}"#, escape_json(&e.to_string())),
                        );
                    }
                }
                guard.engine.upsert(id, text, vector);
                (
                    "200 OK",
                    "application/json",
                    format!(r#"{{"ingested":true,"id":{}}}"#, id),
                )
            }
            Err(err) => (
                "400 Bad Request",
                "application/json",
                format!(r#"{{"error":"{}"}}"#, escape_json(&err)),
            ),
        },
        ("POST", "/search") => match parse_search_request(&body_str) {
            Ok((query_text, query_vector, top_k)) => {
                let guard = state.lock().unwrap();
                let hits = guard.engine.search(&query_text, &query_vector, top_k);
                let items: Vec<String> = hits.iter().map(|h| h.to_json()).collect();
                (
                    "200 OK",
                    "application/json",
                    format!(r#"{{"results":[{}]}}"#, items.join(",")),
                )
            }
            Err(err) => (
                "400 Bad Request",
                "application/json",
                format!(r#"{{"error":"{}"}}"#, escape_json(&err)),
            ),
        },
        ("POST", "/snapshot/save") => {
            let mut guard = state.lock().unwrap();
            let snap_path = guard
                .snapshot_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("vectorcore.vcsnap"));
            match save_snapshot(&guard.engine, &snap_path) {
                Ok(()) => (
                    "200 OK",
                    "application/json",
                    format!(
                        r#"{{"saved":true,"path":"{}","documents":{}}}"#,
                        escape_json(&snap_path.display().to_string()),
                        guard.engine.len()
                    ),
                ),
                Err(e) => (
                    "500 Internal Server Error",
                    "application/json",
                    format!(r#"{{"error":"Snapshot save failed: {}"}}"#, escape_json(&e.to_string())),
                ),
            }
        },
        ("POST", "/snapshot/load") => {
            let mut guard = state.lock().unwrap();
            let snap_path = guard
                .snapshot_path
                .clone()
                .unwrap_or_else(|| PathBuf::from("vectorcore.vcsnap"));
            match load_snapshot(&snap_path) {
                Ok(restored) => {
                    let count = restored.len();
                    guard.engine = restored;
                    (
                        "200 OK",
                        "application/json",
                        format!(
                            r#"{{"loaded":true,"path":"{}","documents":{}}}"#,
                            escape_json(&snap_path.display().to_string()),
                            count
                        ),
                    )
                }
                Err(e) => (
                    "500 Internal Server Error",
                    "application/json",
                    format!(r#"{{"error":"Snapshot load failed: {}"}}"#, escape_json(&e.to_string())),
                ),
            }
        },
        _ => (
            "404 Not Found",
            "application/json",
            r#"{"error":"Not found"}"#.to_string(),
        ),
    };

    write_response(&mut stream, status, content_type, &resp_body)
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        content_type,
        body.len(),
        body
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()
}
