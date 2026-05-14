mod dataset;
mod vectorize;

use dataset::{Dataset, knn_search, preprocess};
use vectorize::{MccRiskTable, NormConstants, NormConstantsRaw, vectorize_manual};

use std::env;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

/// 6 possible pre-computed responses — zero allocation, zero formatting
static RESP_OK: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 33\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 34\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1}",
];

static RESP_READY: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\nOK";
static RESP_404: &[u8] = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

struct State {
    dataset: Dataset,
    norm: NormConstants,
    mcc_risk: MccRiskTable,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "--preprocess" {
        let mut input = String::new();
        let mut output = String::new();
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--input" => { i += 1; input = args[i].clone(); }
                "--output" => { i += 1; output = args[i].clone(); }
                _ => {}
            }
            i += 1;
        }
        preprocess(&input, &output);
        return;
    }

    let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "/data".into());
    let port: u16 = env::var("PORT").unwrap_or_else(|_| "8080".into()).parse().unwrap();

    let bin = format!("{}/references.bin", data_dir);
    let gz = format!("{}/references.json.gz", data_dir);
    if !std::path::Path::new(&bin).exists() {
        if std::path::Path::new(&gz).exists() { preprocess(&gz, &bin); }
        else { eprintln!("ERROR: no data"); std::process::exit(1); }
    }

    let dataset = Dataset::from_mmap(&bin);
    let norm_raw: NormConstantsRaw = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/normalization.json", data_dir)).unwrap()
    ).unwrap();
    let norm = NormConstants::from_raw(&norm_raw);
    let mcc_map: std::collections::HashMap<String, f64> = serde_json::from_str(
        &std::fs::read_to_string(format!("{}/mcc_risk.json", data_dir)).unwrap()
    ).unwrap();
    let mcc_risk = MccRiskTable::from_map(&mcc_map);

    let state = Arc::new(State { dataset, norm, mcc_risk });

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).unwrap();
    eprintln!("[srv] :{} ready, {} vectors", port, state.dataset.count);

    // Concurrency: Thread per connection model. Zero thread contention on single CPU,
    // perfectly robust keep-alive pipelining.
    for stream in listener.incoming() {
        if let Ok(mut stream) = stream {
            let _ = stream.set_nodelay(true); // Disable Nagle's algorithm
            let state = state.clone();
            std::thread::spawn(move || {
                handle_connection(&mut stream, &state);
            });
        }
    }
}

fn handle_connection(stream: &mut std::net::TcpStream, state: &State) {
    // Pre-allocated stack buffer — zero heap allocation
    let mut buf = [0u8; 4096];
    let mut filled = 0usize;

    loop {
        let n = match stream.read(&mut buf[filled..]) {
            Ok(0) => return,
            Ok(n) => n,
            Err(_) => return,
        };
        filled += n;

        let header_end = match find_header_end(&buf[..filled]) {
            Some(pos) => pos,
            None => {
                if filled >= buf.len() { return; }
                continue;
            }
        };

        if buf[0] == b'G' {
            let _ = stream.write_all(RESP_READY);
        } else if buf[0] == b'P' {
            let content_len = parse_content_length(&buf[..header_end]);
            let body_start = header_end;
            let body_end = body_start + content_len;

            while filled < body_end {
                let n = match stream.read(&mut buf[filled..]) {
                    Ok(0) => return,
                    Ok(n) => n,
                    Err(_) => return,
                };
                filled += n;
            }

            if buf[5] == b'f' {
                let body = &buf[body_start..body_end];
                let resp = match vectorize_manual(body, &state.norm, &state.mcc_risk) {
                    Some(query) => {
                        let fraud_count = knn_search(&state.dataset, &query);
                        RESP_OK[fraud_count as usize]
                    }
                    None => RESP_404,
                };
                let _ = stream.write_all(resp);
            } else {
                let _ = stream.write_all(RESP_404);
            }
        } else {
            let _ = stream.write_all(RESP_404);
        }

        let consumed = if buf[0] == b'G' {
            header_end
        } else {
            let cl = parse_content_length(&buf[..header_end]);
            header_end + cl
        };

        if consumed < filled {
            buf.copy_within(consumed..filled, 0);
            filled -= consumed;
        } else {
            filled = 0;
        }
    }
}

#[inline]
fn find_header_end(buf: &[u8]) -> Option<usize> {
    let len = buf.len();
    if len < 4 { return None; }
    let mut i = 0;
    while i + 3 < len {
        if buf[i] == b'\r' && buf[i+1] == b'\n' && buf[i+2] == b'\r' && buf[i+3] == b'\n' {
            return Some(i + 4);
        }
        i += 1;
    }
    None
}

#[inline]
fn parse_content_length(headers: &[u8]) -> usize {
    let mut i = 0;
    while i + 16 < headers.len() {
        if (headers[i] == b'C' || headers[i] == b'c')
            && (headers[i+8] == b'L' || headers[i+8] == b'l')
            && headers[i+15] == b' '
        {
            let mut n = 0usize;
            let mut j = i + 16;
            while j < headers.len() && headers[j] >= b'0' && headers[j] <= b'9' {
                n = n * 10 + (headers[j] - b'0') as usize;
                j += 1;
            }
            return n;
        }
        i += 1;
    }
    0
}
