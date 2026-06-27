//! `progress.bin` I/O (f64 little-endian × N_WEIGHTS、任意 quantile trailer)。
//!
//! Rust kernel 側は重みを `f32` で持つが、`progress.bin` の wire format は
//! `f64` LE であることに注意。書き込み時 f32→f64 cast、読み込み時 f64→f32 cast。
//! v2 trailer 付きファイルからも先頭 weights block のみ読む。

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use shogi_features::{PROGRESS_BIN_WEIGHT_BYTES, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS};

/// `weights` (長さ `SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS`、f32) を
/// v1 `progress.bin` (f64 LE × N、trailer 無し) として書き出す。
pub fn write_progress_bin(path: &Path, weights: &[f32]) -> io::Result<()> {
    if weights.len() != SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "weight slice length {} != expected {}",
                weights.len(),
                SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS
            ),
        ));
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for &w in weights {
        let bytes = (w as f64).to_le_bytes();
        writer.write_all(&bytes)?;
    }
    writer.flush()
}

/// progress.bin を読み込んで `Vec<f32>` (N_WEIGHTS 要素) として返す。
/// v2 (quantile trailer 付き) でも先頭 weights block のみ読む。
pub fn read_progress_bin(path: &Path) -> io::Result<Vec<f32>> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if (metadata.len() as usize) < PROGRESS_BIN_WEIGHT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "progress.bin size {} < expected weight block {} bytes",
                metadata.len(),
                PROGRESS_BIN_WEIGHT_BYTES
            ),
        ));
    }
    let mut reader = BufReader::new(file);
    let mut buf = vec![0_u8; PROGRESS_BIN_WEIGHT_BYTES];
    reader.read_exact(&mut buf)?;
    let weights = buf
        .chunks_exact(std::mem::size_of::<f64>())
        .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size guaranteed")) as f32)
        .collect();
    Ok(weights)
}
