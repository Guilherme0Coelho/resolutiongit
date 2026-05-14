use serde::Deserialize;
use crate::dataset::{PADDED_DIMS, quantize};

/// Pre-computed reciprocals (f32) — multiplications instead of divisions
#[derive(Clone, Copy)]
pub struct NormConstants {
    pub inv_max_amount: f32,
    pub inv_max_installments: f32,
    pub inv_amount_vs_avg_ratio: f32,
    pub inv_max_minutes: f32,
    pub inv_max_km: f32,
    pub inv_max_tx_count_24h: f32,
    pub inv_max_merchant_avg_amount: f32,
}

#[derive(Deserialize)]
pub struct NormConstantsRaw {
    pub max_amount: f64,
    pub max_installments: f64,
    pub amount_vs_avg_ratio: f64,
    pub max_minutes: f64,
    pub max_km: f64,
    pub max_tx_count_24h: f64,
    pub max_merchant_avg_amount: f64,
}

impl NormConstants {
    pub fn from_raw(raw: &NormConstantsRaw) -> Self {
        NormConstants {
            inv_max_amount: (1.0 / raw.max_amount) as f32,
            inv_max_installments: (1.0 / raw.max_installments) as f32,
            inv_amount_vs_avg_ratio: (1.0 / raw.amount_vs_avg_ratio) as f32,
            inv_max_minutes: (1.0 / raw.max_minutes) as f32,
            inv_max_km: (1.0 / raw.max_km) as f32,
            inv_max_tx_count_24h: (1.0 / raw.max_tx_count_24h) as f32,
            inv_max_merchant_avg_amount: (1.0 / raw.max_merchant_avg_amount) as f32,
        }
    }
}

/// Fixed-size MCC risk table — O(1) lookup, no HashMap
pub struct MccRiskTable {
    slots: [(u16, f32); 32],
}

impl MccRiskTable {
    pub fn from_map(map: &std::collections::HashMap<String, f64>) -> Self {
        let mut slots = [(0u16, 0.5f32); 32];
        for (key, &val) in map {
            let mcc: u16 = key.parse().unwrap_or(0);
            let idx = (mcc as usize) & 31;
            for offset in 0..32 {
                let slot = (idx + offset) & 31;
                if slots[slot].0 == 0 || slots[slot].0 == mcc {
                    slots[slot] = (mcc, val as f32);
                    break;
                }
            }
        }
        MccRiskTable { slots }
    }

    #[inline(always)]
    pub fn get(&self, mcc_bytes: &[u8]) -> f32 {
        let mcc = parse_u16_fast(mcc_bytes);
        let idx = (mcc as usize) & 31;
        let mut offset = 0;
        while offset < 32 {
            let slot = (idx + offset) & 31;
            if self.slots[slot].0 == mcc { return self.slots[slot].1; }
            if self.slots[slot].0 == 0 { return 0.5; }
            offset += 1;
        }
        0.5
    }
}

#[inline(always)]
fn parse_u16_fast(bytes: &[u8]) -> u16 {
    let mut n = 0u16;
    let mut i = 0;
    while i < bytes.len() && i < 4 {
        n = n.wrapping_mul(10).wrapping_add((bytes[i].wrapping_sub(b'0')) as u16);
        i += 1;
    }
    n
}

#[inline(always)]
fn clamp01(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x }
}

#[inline(always)]
fn fast_u32_4(a: u8, b: u8, c: u8, d: u8) -> u32 {
    ((a.wrapping_sub(b'0')) as u32) * 1000 + ((b.wrapping_sub(b'0')) as u32) * 100
        + ((c.wrapping_sub(b'0')) as u32) * 10 + ((d.wrapping_sub(b'0')) as u32)
}

#[inline(always)]
fn fast_u32_2(a: u8, b: u8) -> u32 {
    ((a.wrapping_sub(b'0')) as u32) * 10 + ((b.wrapping_sub(b'0')) as u32)
}

#[inline(always)]
fn day_of_week(year: u32, month: u32, day: u32) -> u32 {
    const T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year };
    let dow = (y + y / 4 - y / 100 + y / 400 + T[(month - 1) as usize] + day) % 7;
    if dow == 0 { 6 } else { dow - 1 }
}

#[inline(always)]
fn find_subsequence(haystack: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    if start + needle.len() > haystack.len() { return None; }
    haystack[start..].windows(needle.len()).position(|w| w == needle).map(|p| start + p)
}

#[inline(always)]
fn parse_float(bytes: &[u8], start: usize) -> f32 {
    let mut i = start;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':' || bytes[i] == b'"') {
        i += 1;
    }
    let mut sign = 1.0f32;
    if i < bytes.len() && bytes[i] == b'-' {
        sign = -1.0;
        i += 1;
    }
    let mut val = 0.0f32;
    while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
        val = val * 10.0 + (bytes[i] - b'0') as f32;
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let mut div = 10.0f32;
        while i < bytes.len() && bytes[i] >= b'0' && bytes[i] <= b'9' {
            val += (bytes[i] - b'0') as f32 / div;
            div *= 10.0;
            i += 1;
        }
    }
    val * sign
}

#[inline(always)]
fn parse_timestamp_str(bytes: &[u8], start: usize) -> (f32, f32, i64) {
    let mut i = start;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':' || bytes[i] == b'"') {
        i += 1;
    }
    let b = &bytes[i..];
    if b.len() < 19 { return (0.0, 0.0, 0); }
    let year = fast_u32_4(b[0], b[1], b[2], b[3]);
    let month = fast_u32_2(b[5], b[6]);
    let day = fast_u32_2(b[8], b[9]);
    let hour = fast_u32_2(b[11], b[12]);
    let min = fast_u32_2(b[14], b[15]);
    let sec = fast_u32_2(b[17], b[18]);
    let dow = day_of_week(year, month, day);

    let mut td = 365 * year as i64 + year as i64 / 4 - year as i64 / 100 + year as i64 / 400;
    const DIM: [i64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    td += DIM[month as usize];
    if month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) { td += 1; }
    td += day as i64;
    let ts = td * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64;

    (hour as f32, dow as f32, ts)
}

/// Manual Byte Slicing Vectorizer — Zero-copy, Zero-allocation, No-Parsing Path.
#[inline]
pub fn vectorize_manual(
    bytes: &[u8],
    norm: &NormConstants,
    mcc_risk: &MccRiskTable,
) -> Option<[u8; PADDED_DIMS]> {
    // Locate "transaction"
    let tx_pos = find_subsequence(bytes, b"\"transaction\"", 0)?;
    let amount_pos = find_subsequence(bytes, b"\"amount\"", tx_pos)?;
    let amount = parse_float(bytes, amount_pos + 8);

    let inst_pos = find_subsequence(bytes, b"\"installments\"", tx_pos)?;
    let installments = parse_float(bytes, inst_pos + 14);

    let req_at_pos = find_subsequence(bytes, b"\"requested_at\"", tx_pos)?;
    let (hour, dow, ts_req) = parse_timestamp_str(bytes, req_at_pos + 14);

    // Locate "customer"
    let cust_pos = find_subsequence(bytes, b"\"customer\"", 0)?;
    let cust_avg_pos = find_subsequence(bytes, b"\"avg_amount\"", cust_pos)?;
    let cust_avg_amount = parse_float(bytes, cust_avg_pos + 12);

    let tx_count_pos = find_subsequence(bytes, b"\"tx_count_24h\"", cust_pos)?;
    let tx_count_24h = parse_float(bytes, tx_count_pos + 14);

    // Locate "merchant"
    let merch_pos = find_subsequence(bytes, b"\"merchant\"", 0)?;
    let merch_id_pos = find_subsequence(bytes, b"\"id\"", merch_pos)?;
    // extract merchant id bytes
    let mut i = merch_id_pos + 4;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':' || bytes[i] == b'"') { i += 1; }
    let merch_id_start = i;
    while i < bytes.len() && bytes[i] != b'"' { i += 1; }
    let merch_id = &bytes[merch_id_start..i];

    let mcc_pos = find_subsequence(bytes, b"\"mcc\"", merch_pos)?;
    let mut i = mcc_pos + 5;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':' || bytes[i] == b'"') { i += 1; }
    let mcc_bytes = if i + 4 <= bytes.len() { &bytes[i..i+4] } else { b"0000" };

    let merch_avg_pos = find_subsequence(bytes, b"\"avg_amount\"", merch_pos)?;
    let merch_avg_amount = parse_float(bytes, merch_avg_pos + 12);

    // Check if merch_id is in known_merchants
    let known_pos = find_subsequence(bytes, b"\"known_merchants\"", cust_pos)?;
    let mut i = known_pos + 17;
    while i < bytes.len() && bytes[i] != b'[' { i += 1; }
    let arr_start = i;
    while i < bytes.len() && bytes[i] != b']' { i += 1; }
    let known_slice = &bytes[arr_start..i];
    let is_known = find_subsequence(known_slice, merch_id, 0).is_some();

    // Locate "terminal"
    let term_pos = find_subsequence(bytes, b"\"terminal\"", 0)?;
    let online_pos = find_subsequence(bytes, b"\"is_online\"", term_pos)?;
    let mut i = online_pos + 11;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':') { i += 1; }
    let is_online = i < bytes.len() && bytes[i] == b't';

    let card_pos = find_subsequence(bytes, b"\"card_present\"", term_pos)?;
    let mut i = card_pos + 14;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':') { i += 1; }
    let card_present = i < bytes.len() && bytes[i] == b't';

    let km_home_pos = find_subsequence(bytes, b"\"km_from_home\"", term_pos)?;
    let km_from_home = parse_float(bytes, km_home_pos + 14);

    // Locate optional "last_transaction"
    let (d5, d6) = match find_subsequence(bytes, b"\"last_transaction\"", 0) {
        Some(last_pos) => {
            let mut i = last_pos + 18;
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':') { i += 1; }
            if i < bytes.len() && bytes[i] == b'{' {
                let lt_ts_pos = find_subsequence(bytes, b"\"timestamp\"", i).unwrap_or(i);
                let (_, _, ts_last) = parse_timestamp_str(bytes, lt_ts_pos + 11);

                let lt_km_pos = find_subsequence(bytes, b"\"km_from_current\"", i).unwrap_or(i);
                let km_from_current = parse_float(bytes, lt_km_pos + 17);

                let diff = if ts_req > ts_last { ts_req - ts_last } else { ts_last - ts_req };
                let minutes = diff as f32 / 60.0;
                (
                    clamp01(minutes * norm.inv_max_minutes),
                    clamp01(km_from_current * norm.inv_max_km),
                )
            } else {
                (-1.0, -1.0)
            }
        }
        None => (-1.0, -1.0),
    };

    // Calculate dimensions branchless / optimized
    let d0 = clamp01(amount * norm.inv_max_amount);
    let d1 = clamp01(installments * norm.inv_max_installments);
    let d2 = clamp01((amount / if cust_avg_amount > 0.0 { cust_avg_amount } else { 1.0 }) * norm.inv_amount_vs_avg_ratio);
    let d3 = hour * (1.0 / 23.0);
    let d4 = dow * (1.0 / 6.0);
    let d7 = clamp01(km_from_home * norm.inv_max_km);
    let d8 = clamp01(tx_count_24h * norm.inv_max_tx_count_24h);
    let d9 = if is_online { 1.0 } else { 0.0 };
    let d10 = if card_present { 1.0 } else { 0.0 };
    let d11 = if is_known { 0.0 } else { 1.0 };
    let d12 = mcc_risk.get(mcc_bytes);
    let d13 = clamp01(merch_avg_amount * norm.inv_max_merchant_avg_amount);

    Some([
        quantize(d0), quantize(d1), quantize(d2), quantize(d3),
        quantize(d4), quantize(d5), quantize(d6), quantize(d7),
        quantize(d8), quantize(d9), quantize(d10), quantize(d11),
        quantize(d12), quantize(d13),
        128, 128, // padding
    ])
}
