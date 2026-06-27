//! `progress.bin` wire format: v1 (weights only) and v2 (weights + quantile trailer)。
//!
//! v1: f64 LE × `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS` = `PROGRESS_BIN_WEIGHT_BYTES` 固定。
//! v2: v1 の直後に trailer (`PRGQ` magic + version + `num_buckets` + thresholds)。

use crate::progress_kpabs::{MAX_NUM_BUCKETS, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS};

/// v1 weights block の byte 長 (`125_388` × 8)。
pub const PROGRESS_BIN_WEIGHT_BYTES: usize =
    SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();

const TRAILER_MAGIC: &[u8; 4] = b"PRGQ";
const TRAILER_VERSION: u32 = 1;
const TRAILER_HEADER_BYTES: usize = 12;

/// progress.bin 読込後の bucket 割当方式。
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ProgressBinning {
    /// `floor(p × N)` による `[0, 1]` 等幅分割 (v1 既定)。
    #[default]
    EqualWidth,
    /// キャリブレーション集合の分位点閾値による等頻度分割 (v2 trailer)。
    Quantile {
        num_buckets: usize,
        thresholds: Box<[f32]>,
    },
}

impl ProgressBinning {
    /// 等頻度 trailer が付いているか。
    pub fn is_quantile(&self) -> bool {
        matches!(self, Self::Quantile { .. })
    }

    /// 等頻度 trailer の `num_buckets`。等幅時は `None`。
    pub fn quantile_num_buckets(&self) -> Option<usize> {
        match self {
            Self::EqualWidth => None,
            Self::Quantile { num_buckets, .. } => Some(*num_buckets),
        }
    }
}

/// `p ∈ [0, 1]` を等幅 N-bucket に割当 (`floor(p × N)` clamp)。
#[inline]
pub fn equal_width_bucket(p: f32, num_buckets: usize) -> u8 {
    let n_i32 = num_buckets as i32;
    let raw = (p * num_buckets as f32).floor() as i32;
    raw.clamp(0, n_i32 - 1) as u8
}

/// 等頻度閾値で `p` を bucket へ割当。`thresholds.len() == num_buckets - 1`。
#[inline]
pub fn quantile_bucket(p: f32, num_buckets: usize, thresholds: &[f32]) -> u8 {
    debug_assert_eq!(thresholds.len() + 1, num_buckets);
    let bucket = thresholds.partition_point(|&t| p >= t);
    bucket.min(num_buckets - 1) as u8
}

/// ソート済み progress 値から等頻度閾値 `N-1` 個を導出する。
pub fn compute_quantile_thresholds(
    sorted_ps: &[f32],
    num_buckets: usize,
) -> Result<Vec<f32>, String> {
    if !(2..=MAX_NUM_BUCKETS).contains(&num_buckets) {
        return Err(format!(
            "num_buckets must be in [2, {MAX_NUM_BUCKETS}] (got {num_buckets})"
        ));
    }
    if sorted_ps.is_empty() {
        return Err("cannot compute quantile thresholds from empty sample".to_string());
    }
    let len = sorted_ps.len();
    let mut thresholds = Vec::with_capacity(num_buckets - 1);
    for k in 0..(num_buckets - 1) {
        let idx = (k + 1) * len / num_buckets;
        let idx = idx.min(len - 1);
        thresholds.push(sorted_ps[idx]);
    }
    validate_thresholds(&thresholds, num_buckets)?;
    Ok(thresholds)
}

fn validate_thresholds(thresholds: &[f32], num_buckets: usize) -> Result<(), String> {
    if thresholds.len() + 1 != num_buckets {
        return Err(format!(
            "threshold count {} does not match num_buckets {num_buckets}",
            thresholds.len()
        ));
    }
    for (i, &t) in thresholds.iter().enumerate() {
        if !t.is_finite() || !(0.0..1.0).contains(&t) {
            return Err(format!("threshold[{i}] = {t} is not in (0, 1)"));
        }
        if i > 0 && t <= thresholds[i - 1] {
            return Err(format!(
                "thresholds must be strictly increasing (threshold[{i}] = {t} <= threshold[{}] = {})",
                i - 1,
                thresholds[i - 1]
            ));
        }
    }
    Ok(())
}

fn trailer_byte_len(num_buckets: usize) -> usize {
    TRAILER_HEADER_BYTES + (num_buckets - 1) * std::mem::size_of::<f32>()
}

fn decode_weights(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() < PROGRESS_BIN_WEIGHT_BYTES {
        return Err(format!(
            "progress.bin too short: got {} bytes, need at least {PROGRESS_BIN_WEIGHT_BYTES}",
            bytes.len()
        ));
    }
    let weight_bytes = &bytes[..PROGRESS_BIN_WEIGHT_BYTES];
    let weights: Vec<f32> = weight_bytes
        .chunks_exact(std::mem::size_of::<f64>())
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size is checked")) as f32)
        .collect();
    if weights.len() != SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
        return Err(format!(
            "weight count {} != expected {}",
            weights.len(),
            SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS
        ));
    }
    Ok(weights)
}

fn parse_trailer(trailer: &[u8]) -> Result<ProgressBinning, String> {
    if trailer.len() < TRAILER_HEADER_BYTES {
        return Err(format!(
            "progress.bin v2 trailer too short: got {} bytes, need at least {TRAILER_HEADER_BYTES}",
            trailer.len()
        ));
    }
    if trailer[..4] != *TRAILER_MAGIC {
        return Err(format!(
            "progress.bin v2 trailer magic mismatch: got {:?}, expected {TRAILER_MAGIC:?}",
            &trailer[..4]
        ));
    }
    let version = u32::from_le_bytes(trailer[4..8].try_into().expect("slice len checked"));
    if version != TRAILER_VERSION {
        return Err(format!(
            "progress.bin v2 trailer version {version} is unsupported (expected {TRAILER_VERSION})"
        ));
    }
    let num_buckets = u32::from_le_bytes(trailer[8..12].try_into().expect("slice len checked"));
    let num_buckets = usize::try_from(num_buckets)
        .map_err(|_| format!("progress.bin v2 num_buckets {num_buckets} does not fit in usize"))?;
    if !(2..=MAX_NUM_BUCKETS).contains(&num_buckets) {
        return Err(format!(
            "progress.bin v2 num_buckets must be in [2, {MAX_NUM_BUCKETS}] (got {num_buckets})"
        ));
    }
    let expected_trailer_len = trailer_byte_len(num_buckets);
    if trailer.len() != expected_trailer_len {
        return Err(format!(
            "progress.bin v2 trailer length mismatch: got {} bytes, expected {expected_trailer_len} for num_buckets={num_buckets}",
            trailer.len()
        ));
    }
    let threshold_bytes = &trailer[TRAILER_HEADER_BYTES..];
    let thresholds: Vec<f32> = threshold_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size is checked")))
        .collect();
    validate_thresholds(&thresholds, num_buckets)?;
    Ok(ProgressBinning::Quantile {
        num_buckets,
        thresholds: thresholds.into_boxed_slice(),
    })
}

/// `progress.bin` bytes を weights + binning に分解する。
pub fn parse_progress_bin(bytes: &[u8]) -> Result<(Vec<f32>, ProgressBinning), String> {
    let weights = decode_weights(bytes)?;
    let binning = if bytes.len() == PROGRESS_BIN_WEIGHT_BYTES {
        ProgressBinning::EqualWidth
    } else if bytes.len() > PROGRESS_BIN_WEIGHT_BYTES {
        parse_trailer(&bytes[PROGRESS_BIN_WEIGHT_BYTES..])?
    } else {
        return Err(format!(
            "progress.bin size mismatch: got {} bytes, expected {PROGRESS_BIN_WEIGHT_BYTES} (v1) or {PROGRESS_BIN_WEIGHT_BYTES} + trailer (v2)",
            bytes.len()
        ));
    };
    Ok((weights, binning))
}

/// weights + binning から `progress.bin` bytes を構築する。
pub fn write_progress_bin_bytes(
    weights: &[f32],
    binning: &ProgressBinning,
) -> Result<Vec<u8>, String> {
    if weights.len() != SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
        return Err(format!(
            "weight slice length {} != expected {}",
            weights.len(),
            SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS
        ));
    }
    let mut out = Vec::with_capacity(PROGRESS_BIN_WEIGHT_BYTES + trailer_byte_len(MAX_NUM_BUCKETS));
    for &w in weights {
        out.extend_from_slice(&(w as f64).to_le_bytes());
    }
    match binning {
        ProgressBinning::EqualWidth => {}
        ProgressBinning::Quantile {
            num_buckets,
            thresholds,
        } => {
            validate_thresholds(thresholds, *num_buckets)?;
            out.extend_from_slice(TRAILER_MAGIC);
            out.extend_from_slice(&TRAILER_VERSION.to_le_bytes());
            let num_buckets_u32 = u32::try_from(*num_buckets)
                .map_err(|_| format!("num_buckets {num_buckets} does not fit in u32"))?;
            out.extend_from_slice(&num_buckets_u32.to_le_bytes());
            for &t in thresholds.iter() {
                out.extend_from_slice(&t.to_le_bytes());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_weights() -> Vec<f32> {
        vec![0.0_f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS]
    }

    #[test]
    fn v1_round_trip_is_equal_width() {
        let weights = dummy_weights();
        let bytes = write_progress_bin_bytes(&weights, &ProgressBinning::EqualWidth).unwrap();
        assert_eq!(bytes.len(), PROGRESS_BIN_WEIGHT_BYTES);
        let (read_weights, binning) = parse_progress_bin(&bytes).unwrap();
        assert_eq!(read_weights, weights);
        assert_eq!(binning, ProgressBinning::EqualWidth);
    }

    #[test]
    fn v2_round_trip_preserves_thresholds() {
        let weights = dummy_weights();
        let binning = ProgressBinning::Quantile {
            num_buckets: 4,
            thresholds: Box::from([0.25_f32, 0.5, 0.75]),
        };
        let bytes = write_progress_bin_bytes(&weights, &binning).unwrap();
        assert_eq!(bytes.len(), PROGRESS_BIN_WEIGHT_BYTES + trailer_byte_len(4));
        let (read_weights, read_binning) = parse_progress_bin(&bytes).unwrap();
        assert_eq!(read_weights, weights);
        assert_eq!(read_binning, binning);
    }

    #[test]
    fn quantile_bucket_assigns_by_threshold_count() {
        let thresholds = [0.25_f32, 0.5, 0.75];
        assert_eq!(quantile_bucket(0.1, 4, &thresholds), 0);
        assert_eq!(quantile_bucket(0.25, 4, &thresholds), 1);
        assert_eq!(quantile_bucket(0.5, 4, &thresholds), 2);
        assert_eq!(quantile_bucket(0.49, 4, &thresholds), 1);
        assert_eq!(quantile_bucket(0.75, 4, &thresholds), 3);
        assert_eq!(quantile_bucket(1.0, 4, &thresholds), 3);
    }

    #[test]
    fn compute_quantile_thresholds_even_split() {
        let ps: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let thresholds = compute_quantile_thresholds(&ps, 4).unwrap();
        assert_eq!(thresholds.len(), 3);
        assert!((thresholds[0] - ps[25]).abs() < 1e-5);
        assert!((thresholds[1] - ps[50]).abs() < 1e-5);
        assert!((thresholds[2] - ps[75]).abs() < 1e-5);
    }

    #[test]
    fn rejects_non_monotonic_trailer() {
        let weights = dummy_weights();
        let binning = ProgressBinning::Quantile {
            num_buckets: 3,
            thresholds: Box::from([0.5_f32, 0.4]),
        };
        assert!(write_progress_bin_bytes(&weights, &binning).is_err());
    }
}
