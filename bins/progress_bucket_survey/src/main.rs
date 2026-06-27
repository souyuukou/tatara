//! `progress-bucket-survey`: survey the progress-bucket distribution of a
//! `progress.bin` over PSV data.
//!
//! `progress-kpabs-train` produces `progress.bin` (KP-absolute progress
//! coefficients) that the LayerStack architecture uses to route each position
//! to an output bucket. This tool loads a `progress.bin`, assigns sampled PSV
//! positions to their progress8kpabs buckets, and prints the resulting
//! histogram — a quick way to check the buckets are not badly skewed.
//!
//! With `--write-calibrated`, sampled progress values are turned into quantile
//! thresholds and written as a v2 `progress.bin` (weights + embedded trailer).
//!
//! ```bash
//! cargo run --release -p progress-bucket-survey -- \
//!     --data <path/to/psv.bin> \
//!     --progress output/progress/<run-name>.e5.bin \
//!     --num-buckets 32 --samples 200000 \
//!     --write-calibrated output/progress/<run-name>.e5.q32.bin
//! ```

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use shogi_features::{
    ProgressBinning, ShogiProgressKPAbs, compute_quantile_thresholds, parse_progress_bin,
    quantile_bucket, write_progress_bin_bytes,
};
use shogi_format::PackedSfenValue;

#[derive(Parser, Debug)]
#[command(name = "progress-bucket-survey")]
#[command(about = "Survey the progress8kpabs bucket distribution of a progress.bin over PSV data")]
struct Args {
    /// PSV data files (`.bin`). Pass several as a comma-separated list.
    #[arg(long)]
    data: String,

    /// progress.bin produced by progress-kpabs-train.
    #[arg(long)]
    progress: PathBuf,

    /// Number of positions to sample in total (across all --data files).
    #[arg(long, default_value_t = 50_000)]
    samples: usize,

    /// Read every N-th record (1 = dense scan).
    #[arg(long, default_value_t = 1)]
    stride: u64,

    /// Starting record offset applied to each --data file.
    #[arg(long, default_value_t = 0)]
    offset: u64,

    /// Also print a per-file histogram, not just the combined total.
    #[arg(long)]
    per_pack: bool,

    /// Number of progress buckets (LayerStack `--num-buckets`).
    /// Must be in `[1, 256]` for survey; `[2, 256]` when `--write-calibrated`.
    /// Equal-width: `floor(p * N)`. Quantile (v2 trailer): embedded thresholds.
    #[arg(long, default_value_t = 9)]
    num_buckets: usize,

    /// Write a v2 `progress.bin` with quantile thresholds derived from the
    /// sampled positions. Weights are copied from `--progress`.
    #[arg(long)]
    write_calibrated: Option<PathBuf>,
}

/// 1 PSV ファイルから最大 `max` 局面をサンプリングする。`offset` レコード目から
/// 始め、`stride` レコードごとに 1 件読む。ファイル末尾 / レコード途中の EOF は
/// そこで打ち切る (末尾の半端バイトは無視)。EOF 以外の I/O エラーは伝播する。
fn read_samples(
    path: &PathBuf,
    offset: u64,
    stride: u64,
    max: usize,
) -> io::Result<Vec<PackedSfenValue>> {
    let record = size_of::<PackedSfenValue>() as u64;
    let mut file = File::open(path)?;
    let total_records = file.metadata()?.len() / record;
    let mut out = Vec::new();
    if offset >= total_records {
        return Ok(out);
    }

    file.seek(SeekFrom::Start(offset * record))?;
    while out.len() < max {
        let mut psv = PackedSfenValue::default();
        match file.read_exact(psv.as_bytes_mut()) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        out.push(psv);
        if stride > 1 {
            let skip = i64::try_from((stride - 1).saturating_mul(record)).unwrap_or(i64::MAX);
            file.seek(SeekFrom::Current(skip))?;
        }
    }
    Ok(out)
}

/// 最多 bucket の index と占有率 (%) を返す。空ヒストグラムでは `(0, 0.0)`。
fn top_bucket(hist: &[u64]) -> (usize, f64) {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return (0, 0.0);
    }
    let mut top = 0usize;
    for (i, &count) in hist.iter().enumerate() {
        if count > hist[top] {
            top = i;
        }
    }
    (top, 100.0 * hist[top] as f64 / total as f64)
}

fn print_hist(label: &str, hist: &[u64]) {
    let total: u64 = hist.iter().sum();
    println!("\n== {label} ==");
    if total == 0 {
        println!("(no positions)");
        return;
    }
    for (i, &count) in hist.iter().enumerate() {
        let pct = 100.0 * count as f64 / total as f64;
        println!("bucket {i}: {count:>10}  ({pct:>6.2}%)");
    }
    let (top, share) = top_bucket(hist);
    println!("total {total}, top bucket {top} ({share:.2}%)");
}

fn bucket_for_survey(
    kpabs: &ShogiProgressKPAbs,
    psv: &PackedSfenValue,
    num_buckets: usize,
    calibrated_thresholds: Option<&[f32]>,
) -> u8 {
    if let Some(thresholds) = calibrated_thresholds {
        let board = psv.decode();
        let p = kpabs.progress_board(&board);
        quantile_bucket(p, num_buckets, thresholds)
    } else {
        kpabs.bucket(psv, num_buckets)
    }
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.stride == 0 {
        return Err("--stride must be >= 1".into());
    }
    let calibrating = args.write_calibrated.is_some();
    let min_buckets = if calibrating { 2 } else { 1 };
    if !(min_buckets..=256).contains(&args.num_buckets) {
        return Err(format!(
            "--num-buckets must be in [{min_buckets}, 256] (got {})",
            args.num_buckets
        )
        .into());
    }
    let data_paths: Vec<PathBuf> = args
        .data
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    if data_paths.is_empty() {
        return Err("--data is required (comma-separated PSV files)".into());
    }

    let progress_bytes = std::fs::read(&args.progress)?;
    let (weights, _) = parse_progress_bin(&progress_bytes)?;

    let kpabs = ShogiProgressKPAbs::load_from_bin(&args.progress)?;
    if !calibrating {
        ShogiProgressKPAbs::ensure_num_buckets_matches(args.num_buckets)?;
    }

    let n_buckets = args.num_buckets;
    let mut total_hist = vec![0u64; n_buckets];
    let mut grand_total = 0usize;
    let mut remaining = args.samples;
    let mut progress_values = if calibrating {
        Vec::with_capacity(args.samples)
    } else {
        Vec::new()
    };

    for path in &data_paths {
        if remaining == 0 {
            break;
        }
        let samples = read_samples(path, args.offset, args.stride, remaining)?;
        remaining -= samples.len();
        grand_total += samples.len();

        let mut pack_hist = vec![0u64; n_buckets];
        for psv in &samples {
            if calibrating {
                let board = psv.decode();
                progress_values.push(kpabs.progress_board(&board));
            }
            pack_hist[bucket_for_survey(&kpabs, psv, n_buckets, None) as usize] += 1;
        }
        for (b, &count) in pack_hist.iter().enumerate() {
            total_hist[b] += count;
        }

        println!("loaded {} positions from {}", samples.len(), path.display());
        if args.per_pack {
            print_hist(&format!("per-pack: {}", path.display()), &pack_hist);
        }
    }

    if grand_total == 0 {
        return Err("no positions read from --data files".into());
    }

    if let Some(output_path) = &args.write_calibrated {
        let mut sorted_ps = progress_values;
        sorted_ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let thresholds = compute_quantile_thresholds(&sorted_ps, n_buckets)?;
        let binning = ProgressBinning::Quantile {
            num_buckets: n_buckets,
            thresholds: thresholds.into_boxed_slice(),
        };
        let out_bytes = write_progress_bin_bytes(&weights, &binning)?;
        let mut file = File::create(output_path)?;
        file.write_all(&out_bytes)?;
        file.flush()?;
        println!(
            "wrote calibrated progress.bin: {} (num_buckets={n_buckets}, {} thresholds)",
            output_path.display(),
            n_buckets - 1
        );

        let thresholds_ref = match &binning {
            ProgressBinning::Quantile { thresholds, .. } => thresholds.as_ref(),
            ProgressBinning::EqualWidth => unreachable!(),
        };
        total_hist.fill(0);
        for &p in &sorted_ps {
            total_hist[quantile_bucket(p, n_buckets, thresholds_ref) as usize] += 1;
        }
        print_hist(
            &format!("post-calibration quantile distribution (same sample, N={n_buckets})"),
            &total_hist,
        );
    } else {
        let mode = if ShogiProgressKPAbs::loaded_binning().is_quantile() {
            "quantile"
        } else {
            "equal-width"
        };
        print_hist(
            &format!("progress-kpabs bucket distribution ({mode}, N={n_buckets})"),
            &total_hist,
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::top_bucket;
    use shogi_features::{
        ProgressBinning, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, compute_quantile_thresholds,
        quantile_bucket, write_progress_bin_bytes,
    };

    #[test]
    fn top_bucket_picks_the_largest_with_share() {
        let (idx, share) = top_bucket(&[10, 70, 20]);
        assert_eq!(idx, 1);
        assert!((share - 70.0).abs() < 1e-9);
    }

    #[test]
    fn top_bucket_empty_histogram_is_zero() {
        assert_eq!(top_bucket(&[0, 0, 0]), (0, 0.0));
    }

    #[test]
    fn top_bucket_ties_keep_the_first() {
        let (idx, _) = top_bucket(&[50, 50, 0]);
        assert_eq!(idx, 0);
    }

    #[test]
    fn calibration_produces_near_equal_histogram() {
        let ps: Vec<f32> = (0..10_000).map(|i| i as f32 / 9_999.0).collect();
        let mut sorted = ps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let num_buckets = 32usize;
        let thresholds = compute_quantile_thresholds(&sorted, num_buckets).unwrap();
        let mut hist = vec![0u64; num_buckets];
        for &p in &ps {
            hist[quantile_bucket(p, num_buckets, &thresholds) as usize] += 1;
        }
        let target = ps.len() as f64 / num_buckets as f64;
        for (i, &count) in hist.iter().enumerate() {
            let diff = (count as f64 - target).abs() / target;
            assert!(
                diff < 0.02,
                "bucket {i}: count {count} deviates more than 2% from target {target}"
            );
        }
    }

    #[test]
    fn write_calibrated_bytes_round_trip() {
        let weights = vec![0.0_f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
        let ps: Vec<f32> = (0..1000).map(|i| i as f32 / 999.0).collect();
        let thresholds = compute_quantile_thresholds(&ps, 8).unwrap();
        let binning = ProgressBinning::Quantile {
            num_buckets: 8,
            thresholds: thresholds.into_boxed_slice(),
        };
        let bytes = write_progress_bin_bytes(&weights, &binning).unwrap();
        let (_, parsed) = shogi_features::parse_progress_bin(&bytes).unwrap();
        assert_eq!(parsed, binning);
    }
}
