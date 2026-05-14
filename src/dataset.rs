use flate2::read::GzDecoder;
use memmap2::Mmap;
use rayon::prelude::*;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufReader, Read};

pub const DIMS: usize = 14;
pub const PADDED_DIMS: usize = 16;
const K: usize = 5;

#[derive(Deserialize)]
struct RefEntry { vector: Vec<f32>, label: String }

#[inline(always)]
pub fn quantize(val: f32) -> u8 {
    ((val + 1.0) * 127.5).clamp(0.0, 255.0) as u8
}

pub fn preprocess(gz_path: &str, bin_path: &str) {
    eprintln!("[pre] Reading {}", gz_path);
    let file = File::open(gz_path).expect("open");
    let mut reader = BufReader::new(GzDecoder::new(BufReader::new(file)));
    let mut json = Vec::new();
    reader.read_to_end(&mut json).expect("decompress");
    eprintln!("[pre] {} bytes decompressed", json.len());

    let entries: Vec<RefEntry> = serde_json::from_slice(&json).expect("parse");
    drop(json);
    let count = entries.len() as u32;
    eprintln!("[pre] {} entries", count);

    let n = count as usize;
    // Header(64 bytes aligned) + vectors(n*16) + labels(n)
    let mut out = Vec::with_capacity(64 + n * PADDED_DIMS + n);
    out.extend_from_slice(&count.to_le_bytes());
    // Pad header to exactly 64 bytes for strict Cache Line Alignment of vectors array
    for _ in 4..64 {
        out.push(0);
    }

    for e in &entries {
        for &v in &e.vector { out.push(quantize(v)); }
        out.push(128); out.push(128); // pad
    }
    for e in &entries {
        out.push(if e.label == "fraud" { 1 } else { 0 });
    }

    std::fs::write(bin_path, &out).expect("write");
    eprintln!("[pre] Done {} bytes (64-byte Cache Line Aligned)", out.len());
}

pub struct Chunk {
    pub vptr: *const u8,
    pub lptr: *const u8,
    pub count: usize,
}
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

pub struct Dataset {
    _mmap: Mmap,
    pub count: usize,
    pub chunks: Vec<Chunk>,
}
unsafe impl Send for Dataset {}
unsafe impl Sync for Dataset {}

impl Dataset {
    pub fn from_mmap(path: &str) -> Self {
        let file = File::open(path).expect("open");
        let mmap = unsafe { Mmap::map(&file).expect("mmap") };

        let base = mmap.as_ptr();
        let mmap_len = mmap.len();
        assert!(mmap_len >= 64);

        let count = u32::from_le_bytes(unsafe {
            [*base, *base.add(1), *base.add(2), *base.add(3)]
        }) as usize;

        // Vectors array starts precisely at 64-byte Cache Line boundary
        let voff = 64;
        let loff = voff + count * PADDED_DIMS;
        assert!(mmap_len >= loff + count);

        let vectors = unsafe { base.add(voff) };
        let labels = unsafe { base.add(loff) };

        // Prefault ALL pages sequentially
        let mut d: u8 = 0;
        let mut o = 0;
        while o < mmap_len { unsafe { d = d.wrapping_add(*base.add(o)); } o += 4096; }
        std::hint::black_box(d);

        // Segmented Vector Search: divide into chunks fitting L3 cache perfectly
        // e.g. 16 chunks of ~187.5k vectors (~3MB each)
        let num_chunks = 16;
        let chunk_size = count / num_chunks;
        let mut chunks = Vec::with_capacity(num_chunks + 1);

        for c in 0..num_chunks {
            let start = c * chunk_size;
            let end = if c == num_chunks - 1 { count } else { start + chunk_size };
            let chunk_count = end - start;
            if chunk_count > 0 {
                chunks.push(Chunk {
                    vptr: unsafe { vectors.add(start * PADDED_DIMS) },
                    lptr: unsafe { labels.add(start) },
                    count: chunk_count,
                });
            }
        }

        eprintln!("[ds] {} vectors loaded in {} chunks", count, chunks.len());
        Dataset { _mmap: mmap, count, chunks }
    }
}

#[inline(always)]
fn merge_top_k(
    mut a_d: [u32; K], mut a_l: [u8; K],
    b_d: [u32; K], b_l: [u8; K],
) -> ([u32; K], [u8; K]) {
    for i in 0..K {
        let d = b_d[i];
        if d == u32::MAX { continue; }
        let l = b_l[i];
        // find max element to replace
        let mut mx = 0;
        if a_d[1] > a_d[mx] { mx = 1; }
        if a_d[2] > a_d[mx] { mx = 2; }
        if a_d[3] > a_d[mx] { mx = 3; }
        if a_d[4] > a_d[mx] { mx = 4; }
        if d < a_d[mx] {
            a_d[mx] = d;
            a_l[mx] = l;
        }
    }
    (a_d, a_l)
}

/// Busca segmentada paralela usando Rayon — encaixa no Cache L3, loop unrolling e prefetch.
pub fn knn_search(ds: &Dataset, query: &[u8; PADDED_DIMS]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    let q256 = unsafe {
        if is_x86_feature_detected!("avx2") {
            use std::arch::x86_64::*;
            let v128 = _mm_loadu_si128(query.as_ptr() as *const __m128i);
            Some(_mm256_cvtepu8_epi16(v128))
        } else { None }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let q256: Option<()> = None;

    let (_top_d, top_l) = ds.chunks.par_iter().map(|chunk| {
        scan_chunk(chunk, query, q256)
    }).reduce(
        || ([u32::MAX; K], [0u8; K]),
        |a, b| merge_top_k(a.0, a.1, b.0, b.1)
    );

    // Sum fraud count among top K
    // Sort or simply count frauds for the absolute closest K elements
    // Wait, since merge_top_k maintains the set of top K closest, top_l contains exactly their labels!
    let mut fraud: u32 = 0;
    for l in top_l {
        fraud += l as u32;
    }
    fraud
}

#[inline(always)]
fn scan_chunk(
    chunk: &Chunk,
    query: &[u8; PADDED_DIMS],
    #[allow(unused)] q256: Option<std::arch::x86_64::__m256i>,
) -> ([u32; K], [u8; K]) {
    let count = chunk.count;
    let vptr = chunk.vptr;
    let lptr = chunk.lptr;

    let mut top_d = [u32::MAX; K];
    let mut top_l = [0u8; K];
    let mut mx = 0usize;
    let mut thr = u32::MAX;

    let qp = query.as_ptr();

    let count4 = count & !3;
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    if let Some(q16) = q256 {
        unsafe {
            use std::arch::x86_64::*;
            while i < count4 {
                // Prefetch 16 vectors ahead (4 cache lines)
                let pf = vptr.add((i + 16) * PADDED_DIMS);
                _mm_prefetch(pf as *const i8, _MM_HINT_T0);
                _mm_prefetch(pf.add(64) as *const i8, _MM_HINT_T0);
                _mm_prefetch(pf.add(128) as *const i8, _MM_HINT_T0);
                _mm_prefetch(pf.add(192) as *const i8, _MM_HINT_T0);

                let d0 = dist_sq_avx2_preloaded(q16, vptr.add(i * PADDED_DIMS));
                let d1 = dist_sq_avx2_preloaded(q16, vptr.add((i+1) * PADDED_DIMS));
                let d2 = dist_sq_avx2_preloaded(q16, vptr.add((i+2) * PADDED_DIMS));
                let d3 = dist_sq_avx2_preloaded(q16, vptr.add((i+3) * PADDED_DIMS));

                macro_rules! check {
                    ($d:expr, $idx:expr) => {
                        if $d < thr {
                            top_d[mx] = $d;
                            top_l[mx] = *lptr.add($idx);
                            mx = 0;
                            if top_d[1] > top_d[mx] { mx = 1; }
                            if top_d[2] > top_d[mx] { mx = 2; }
                            if top_d[3] > top_d[mx] { mx = 3; }
                            if top_d[4] > top_d[mx] { mx = 4; }
                            thr = top_d[mx];
                        }
                    };
                }
                check!(d0, i);
                check!(d1, i+1);
                check!(d2, i+2);
                check!(d3, i+3);

                // Early Exit Inteligente se match exato (distância 0) nos 5 vizinhos
                if thr == 0 { break; }

                i += 4;
            }
        }
    }

    // Remainder / fallback scalar
    while i < count {
        let d = unsafe { dist_sq_u8_scalar(qp, vptr.add(i * PADDED_DIMS)) };
        if d < thr {
            top_d[mx] = d;
            top_l[mx] = unsafe { *lptr.add(i) };
            mx = 0;
            if top_d[1] > top_d[mx] { mx = 1; }
            if top_d[2] > top_d[mx] { mx = 2; }
            if top_d[3] > top_d[mx] { mx = 3; }
            if top_d[4] > top_d[mx] { mx = 4; }
            thr = top_d[mx];
            if thr == 0 { break; }
        }
        i += 1;
    }

    (top_d, top_l)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_sq_avx2_preloaded(q16: std::arch::x86_64::__m256i, b: *const u8) -> u32 {
    use std::arch::x86_64::*;
    let vb = _mm_loadu_si128(b as *const __m128i);
    let b16 = _mm256_cvtepu8_epi16(vb);
    let diff = _mm256_sub_epi16(q16, b16);
    let sq = _mm256_madd_epi16(diff, diff);
    let hi = _mm256_extracti128_si256(sq, 1);
    let lo = _mm256_castsi256_si128(sq);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b_01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b_00_00_00_01));
    _mm_cvtsi128_si32(s) as u32
}

#[inline(always)]
unsafe fn dist_sq_u8_scalar(a: *const u8, b: *const u8) -> u32 {
    let mut s = 0u32;
    let mut i = 0;
    while i < PADDED_DIMS {
        let d = (*a.add(i) as i32) - (*b.add(i) as i32);
        s += (d * d) as u32;
        i += 1;
    }
    s
}
