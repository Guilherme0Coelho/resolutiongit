use flate2::read::GzDecoder;
use memmap2::Mmap;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Read};

/// Binary format:
/// - 4 bytes: u32 LE count
/// - count * 14 * 4 bytes: f32 LE vectors (contiguous, row-major)
/// - count bytes: u8 labels (0=legit, 1=fraud)

const DIMS: usize = 14;

#[derive(Deserialize)]
struct RefEntry {
    vector: Vec<f32>,
    label: String,
}

pub fn preprocess(gz_path: &str, bin_path: &str) {
    eprintln!("[preprocess] Reading {}", gz_path);

    let file = File::open(gz_path).expect("cannot open references.json.gz");
    let decoder = GzDecoder::new(BufReader::new(file));
    let mut reader = BufReader::new(decoder);
    let mut json_bytes = Vec::new();
    reader.read_to_end(&mut json_bytes).expect("decompress failed");

    eprintln!("[preprocess] Decompressed {} bytes, parsing JSON...", json_bytes.len());

    let entries: Vec<RefEntry> = serde_json::from_slice(&json_bytes).expect("JSON parse failed");
    drop(json_bytes); // free memory ASAP

    let count = entries.len() as u32;
    eprintln!("[preprocess] {} entries, writing binary to {}", count, bin_path);

    let mut out = Vec::with_capacity(4 + (count as usize * DIMS * 4) + count as usize);

    // Header: count
    out.extend_from_slice(&count.to_le_bytes());

    // Vectors: row-major f32 LE
    for entry in &entries {
        assert_eq!(entry.vector.len(), DIMS, "vector must have 14 dims");
        for &val in &entry.vector {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }

    // Labels: u8
    for entry in &entries {
        out.push(if entry.label == "fraud" { 1 } else { 0 });
    }

    std::fs::write(bin_path, &out).expect("write bin failed");
    eprintln!("[preprocess] Done. Binary size: {} bytes", out.len());
}

/// Dataset loaded via mmap for zero-copy access
pub struct Dataset {
    _mmap: Mmap,
    count: usize,
    vectors_ptr: *const f32,
    labels_ptr: *const u8,
}

// SAFETY: The mmap is read-only and lives as long as Dataset
unsafe impl Send for Dataset {}
unsafe impl Sync for Dataset {}

impl Dataset {
    pub fn from_mmap(path: &str) -> Self {
        let file = File::open(path).expect("cannot open bin file");
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };

        let bytes = &mmap[..];
        assert!(bytes.len() >= 4, "bin file too small");

        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

        let vectors_offset = 4;
        let vectors_size = count * DIMS * 4;
        let labels_offset = vectors_offset + vectors_size;

        assert!(
            bytes.len() >= labels_offset + count,
            "bin file truncated"
        );

        let vectors_ptr = bytes[vectors_offset..].as_ptr() as *const f32;
        let labels_ptr = bytes[labels_offset..].as_ptr();

        eprintln!("[dataset] Loaded {} vectors via mmap", count);

        Dataset {
            _mmap: mmap,
            count,
            vectors_ptr,
            labels_ptr,
        }
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get the i-th vector as a slice of 14 f32
    #[inline]
    pub fn vector(&self, i: usize) -> &[f32; DIMS] {
        debug_assert!(i < self.count);
        unsafe {
            let ptr = self.vectors_ptr.add(i * DIMS);
            &*(ptr as *const [f32; DIMS])
        }
    }

    /// Get the i-th label (0=legit, 1=fraud)
    #[inline]
    pub fn label(&self, i: usize) -> u8 {
        debug_assert!(i < self.count);
        unsafe { *self.labels_ptr.add(i) }
    }
}

/// KNN search: find K=5 nearest neighbors, return fraud_score
/// Uses squared euclidean distance (no sqrt needed for ordering)
pub fn knn_search(dataset: &Dataset, query: &[f32; DIMS]) -> f32 {
    const K: usize = 5;

    // Fixed-size max-heap: keep the K smallest distances
    let mut top_dists = [f32::MAX; K];
    let mut top_labels = [0u8; K];
    let mut max_idx = 0usize;
    let mut threshold = f32::MAX;

    let count = dataset.count();

    for i in 0..count {
        let ref_vec = dataset.vector(i);
        let dist = euclidean_dist_sq_early(query, ref_vec, threshold);

        if dist < threshold {
            top_dists[max_idx] = dist;
            top_labels[max_idx] = dataset.label(i);

            // Find new max
            max_idx = 0;
            for j in 1..K {
                if top_dists[j] > top_dists[max_idx] {
                    max_idx = j;
                }
            }
            threshold = top_dists[max_idx];
        }
    }

    let fraud_count: u32 = top_labels.iter().map(|&l| l as u32).sum();
    fraud_count as f32 / K as f32
}

/// Squared euclidean distance with early exit.
/// Returns f32::MAX if partial sum already exceeds `threshold`.
/// The compiler auto-vectorizes the inner loop with -C target-cpu=x86-64-v3 (AVX2).
#[inline]
fn euclidean_dist_sq_early(a: &[f32; DIMS], b: &[f32; DIMS], threshold: f32) -> f32 {
    let mut sum = 0.0f32;
    // Process all 14 dims — compiler vectorizes this loop
    for i in 0..DIMS {
        let d = a[i] - b[i];
        sum += d * d;
    }
    // Check after full computation (branch-free vectorization)
    // Early exit with partial sums hurts SIMD. Let compiler optimize.
    sum
}
