use serde::Deserialize;
use std::collections::HashMap;

const DIMS: usize = 14;

#[derive(Deserialize)]
pub struct NormConstants {
    pub max_amount: f64,
    pub max_installments: f64,
    pub amount_vs_avg_ratio: f64,
    pub max_minutes: f64,
    pub max_km: f64,
    pub max_tx_count_24h: f64,
    pub max_merchant_avg_amount: f64,
}

#[derive(Deserialize)]
pub struct TransactionPayload {
    // id is ignored for scoring
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Deserialize)]
pub struct Transaction {
    pub amount: f64,
    pub installments: f64,
    pub requested_at: String,
}

#[derive(Deserialize)]
pub struct Customer {
    pub avg_amount: f64,
    pub tx_count_24h: f64,
    pub known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f64,
}

#[derive(Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f64,
}

#[inline]
fn clamp01(x: f64) -> f32 {
    x.clamp(0.0, 1.0) as f32
}

/// Parse ISO 8601 timestamp and extract hour (0-23) and day_of_week (mon=0, sun=6)
fn parse_datetime(s: &str) -> (u32, u32) {
    // Format: "2026-03-11T20:23:35Z" or with timezone offset
    // We only need hour and day_of_week in UTC

    let bytes = s.as_bytes();

    // Parse year-month-day
    let year = parse_u32(&bytes[0..4]);
    let month = parse_u32(&bytes[5..7]);
    let day = parse_u32(&bytes[8..10]);
    let hour = parse_u32(&bytes[11..13]);

    // Zeller-like day of week calculation (Monday=0, Sunday=6)
    let dow = day_of_week(year, month, day);

    (hour, dow)
}

#[inline]
fn parse_u32(bytes: &[u8]) -> u32 {
    let mut n = 0u32;
    for &b in bytes {
        n = n * 10 + (b - b'0') as u32;
    }
    n
}

/// Returns day of week: Monday=0, Sunday=6
fn day_of_week(year: u32, month: u32, day: u32) -> u32 {
    // Tomohiko Sakamoto's algorithm
    let t = [0u32, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if month < 3 { year - 1 } else { year };
    let dow = (y + y / 4 - y / 100 + y / 400 + t[(month - 1) as usize] + day) % 7;
    // Sakamoto returns Sunday=0, we want Monday=0
    if dow == 0 { 6 } else { dow - 1 }
}

/// Calculate minutes difference between two ISO timestamps
fn minutes_between(earlier: &str, later: &str) -> f64 {
    let ts1 = parse_timestamp_seconds(earlier);
    let ts2 = parse_timestamp_seconds(later);
    let diff = if ts2 > ts1 { ts2 - ts1 } else { ts1 - ts2 };
    diff as f64 / 60.0
}

/// Parse ISO 8601 to unix-like seconds (simplified, UTC only)
fn parse_timestamp_seconds(s: &str) -> i64 {
    let bytes = s.as_bytes();
    let year = parse_u32(&bytes[0..4]) as i64;
    let month = parse_u32(&bytes[5..7]) as i64;
    let day = parse_u32(&bytes[8..10]) as i64;
    let hour = parse_u32(&bytes[11..13]) as i64;
    let min = parse_u32(&bytes[14..16]) as i64;
    let sec = parse_u32(&bytes[17..19]) as i64;

    // Days from year 0 to year (simplified, good enough for 2026)
    let mut total_days = 365 * year + year / 4 - year / 100 + year / 400;

    let days_in_months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30];
    for m in 0..month {
        total_days += days_in_months[m as usize] as i64;
    }
    // Leap year adjustment
    if month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
        total_days += 1;
    }
    total_days += day;

    total_days * 86400 + hour * 3600 + min * 60 + sec
}

pub fn vectorize(
    payload: &TransactionPayload,
    norm: &NormConstants,
    mcc_risk: &HashMap<String, f64>,
) -> [f32; DIMS] {
    let tx = &payload.transaction;
    let cust = &payload.customer;
    let merch = &payload.merchant;
    let term = &payload.terminal;

    let (hour, dow) = parse_datetime(&tx.requested_at);

    // dim 0: amount
    let d0 = clamp01(tx.amount / norm.max_amount);

    // dim 1: installments
    let d1 = clamp01(tx.installments / norm.max_installments);

    // dim 2: amount_vs_avg
    let d2 = clamp01((tx.amount / cust.avg_amount) / norm.amount_vs_avg_ratio);

    // dim 3: hour_of_day
    let d3 = (hour as f32) / 23.0;

    // dim 4: day_of_week
    let d4 = (dow as f32) / 6.0;

    // dim 5: minutes_since_last_tx
    // dim 6: km_from_last_tx
    let (d5, d6) = match &payload.last_transaction {
        Some(lt) => {
            let minutes = minutes_between(&lt.timestamp, &tx.requested_at);
            (
                clamp01(minutes / norm.max_minutes),
                clamp01(lt.km_from_current / norm.max_km),
            )
        }
        None => (-1.0f32, -1.0f32),
    };

    // dim 7: km_from_home
    let d7 = clamp01(term.km_from_home / norm.max_km);

    // dim 8: tx_count_24h
    let d8 = clamp01(cust.tx_count_24h / norm.max_tx_count_24h);

    // dim 9: is_online
    let d9 = if term.is_online { 1.0f32 } else { 0.0f32 };

    // dim 10: card_present
    let d10 = if term.card_present { 1.0f32 } else { 0.0f32 };

    // dim 11: unknown_merchant (1 = unknown)
    let d11 = if cust.known_merchants.contains(&merch.id) {
        0.0f32
    } else {
        1.0f32
    };

    // dim 12: mcc_risk (default 0.5)
    let d12 = *mcc_risk.get(&merch.mcc).unwrap_or(&0.5) as f32;

    // dim 13: merchant_avg_amount
    let d13 = clamp01(merch.avg_amount / norm.max_merchant_avg_amount);

    [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorize_legit_example() {
        let norm = NormConstants {
            max_amount: 10000.0,
            max_installments: 12.0,
            amount_vs_avg_ratio: 10.0,
            max_minutes: 1440.0,
            max_km: 1000.0,
            max_tx_count_24h: 20.0,
            max_merchant_avg_amount: 10000.0,
        };

        let mut mcc_risk = HashMap::new();
        mcc_risk.insert("5411".to_string(), 0.15);

        let payload = TransactionPayload {
            transaction: Transaction {
                amount: 41.12,
                installments: 2.0,
                requested_at: "2026-03-11T18:45:53Z".to_string(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3.0,
                known_merchants: vec!["MERC-003".to_string(), "MERC-016".to_string()],
            },
            merchant: Merchant {
                id: "MERC-016".to_string(),
                mcc: "5411".to_string(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        };

        let vec = vectorize(&payload, &norm, &mcc_risk);

        // Expected: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
        let expected = [0.0041f32, 0.1667, 0.05, 0.7826, 0.3333, -1.0, -1.0, 0.0292, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006];

        for i in 0..14 {
            assert!(
                (vec[i] - expected[i]).abs() < 0.01,
                "dim {} mismatch: got {} expected {}",
                i, vec[i], expected[i]
            );
        }
    }

    #[test]
    fn test_day_of_week() {
        // 2026-03-11 is a Wednesday = 2
        assert_eq!(day_of_week(2026, 3, 11), 2);
        // 2026-03-14 is a Saturday = 5
        assert_eq!(day_of_week(2026, 3, 14), 5);
    }
}
