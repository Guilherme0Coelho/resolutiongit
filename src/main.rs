mod dataset;
mod vectorize;

use dataset::{Dataset, knn_search, preprocess};
use vectorize::{NormConstants, TransactionPayload};

use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::collections::HashMap;
use std::convert::Infallible;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

struct AppState {
    dataset: Dataset,
    norm: NormConstants,
    mcc_risk: HashMap<String, f64>,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // CLI mode: preprocess dataset
    if args.len() > 1 && args[1] == "--preprocess" {
        let mut input = String::new();
        let mut output = String::new();

        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--input" => {
                    i += 1;
                    input = args[i].clone();
                }
                "--output" => {
                    i += 1;
                    output = args[i].clone();
                }
                _ => {}
            }
            i += 1;
        }

        if input.is_empty() || output.is_empty() {
            eprintln!("Usage: fraud-detector --preprocess --input <gz> --output <bin>");
            std::process::exit(1);
        }

        preprocess(&input, &output);
        return;
    }

    // Server mode
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
        .block_on(run_server());
}

async fn run_server() {
    let data_dir = env::var("DATA_DIR").unwrap_or_else(|_| "/data".to_string());
    let port: u16 = env::var("PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse()
        .unwrap();

    let bin_path = format!("{}/references.bin", data_dir);
    let gz_path = format!("{}/references.json.gz", data_dir);

    // Preprocess if binary doesn't exist yet (fallback for dev)
    if !std::path::Path::new(&bin_path).exists() {
        if std::path::Path::new(&gz_path).exists() {
            eprintln!("[main] Binary not found, preprocessing...");
            preprocess(&gz_path, &bin_path);
        } else {
            eprintln!("[main] ERROR: Neither {} nor {} found", bin_path, gz_path);
            std::process::exit(1);
        }
    }

    // Load dataset via mmap — pages shared between instances by kernel
    let dataset = Dataset::from_mmap(&bin_path);

    // Load normalization constants
    let norm_path = format!("{}/normalization.json", data_dir);
    let norm_json = std::fs::read_to_string(&norm_path).expect("cannot read normalization.json");
    let norm: NormConstants = serde_json::from_str(&norm_json).expect("parse normalization.json");

    // Load MCC risk
    let mcc_path = format!("{}/mcc_risk.json", data_dir);
    let mcc_json = std::fs::read_to_string(&mcc_path).expect("cannot read mcc_risk.json");
    let mcc_risk: HashMap<String, f64> =
        serde_json::from_str(&mcc_json).expect("parse mcc_risk.json");

    let state = Arc::new(AppState {
        dataset,
        norm,
        mcc_risk,
    });

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await.unwrap();
    eprintln!("[main] Listening on :{} with {} vectors loaded", port, state.dataset.count());

    loop {
        let (stream, _) = listener.accept().await.unwrap();
        let io = TokioIo::new(stream);
        let state = state.clone();

        tokio::task::spawn(async move {
            let service = service_fn(move |req| {
                let state = state.clone();
                async move { handle(req, state).await }
            });

            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, service)
                .await
            {
                eprintln!("[http] error: {}", e);
            }
        });
    }
}

async fn handle(
    req: Request<Incoming>,
    state: Arc<AppState>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    match (req.method().clone(), req.uri().path()) {
        (hyper::Method::GET, "/ready") => {
            Ok(Response::builder()
                .status(StatusCode::OK)
                .body(Full::new(Bytes::from("OK")))
                .unwrap())
        }
        (hyper::Method::POST, "/fraud-score") => {
            let body = req.collect().await.unwrap().to_bytes();

            let payload: TransactionPayload = match serde_json::from_slice(&body) {
                Ok(p) => p,
                Err(e) => {
                    return Ok(Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from(format!("{{\"error\":\"{}\"}}", e))))
                        .unwrap());
                }
            };

            // Vectorize the transaction
            let query_vec = vectorize::vectorize(&payload, &state.norm, &state.mcc_risk);

            // KNN brute-force search (auto-vectorized with SIMD)
            let fraud_score = knn_search(&state.dataset, &query_vec);

            let approved = fraud_score < 0.6;

            // Format with controlled precision
            let response_body = format!(
                "{{\"approved\":{},\"fraud_score\":{}}}",
                approved, fraud_score
            );

            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from(response_body)))
                .unwrap())
        }
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("Not Found")))
            .unwrap()),
    }
}
