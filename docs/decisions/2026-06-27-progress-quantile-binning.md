# Progress quantile binning (equal-frequency buckets)

- **Status**: Accepted
- **Date**: 2026-06-27

## Context

LayerStack routes each position to one of N output buckets using a progress
estimate `p ∈ [0, 1]` from `progress.bin` (KP-abs logistic regression weights).
The default binning is equal-width: `floor(p × N)`.

When `p` is clustered on the training distribution, equal-width bins produce
skewed per-bucket sample counts. That hurts LayerStack training because some
per-bucket weight matrices see far fewer positions than others.

An equal-frequency alternative splits the calibration sample so each bucket
receives roughly `1/N` of the positions, using precomputed quantile thresholds
on `p`.

## Decision

### 1. v2 `progress.bin` trailer embeds quantile thresholds

v1 layout is unchanged: `1_003_104` bytes of f64 LE weights only → equal-width
binning.

v2 appends a trailer immediately after the weight block:

| field | type | notes |
|-------|------|-------|
| magic | `[u8; 4]` | `b"PRGQ"` |
| version | u32 LE | `1` |
| num_buckets | u32 LE | `N ∈ [2, 256]` |
| thresholds | `(N-1) × f32` LE | strictly increasing, each in `(0, 1)` |

Bucket assignment: `bucket = |{ t ∈ thresholds : p >= t }|`, clamped to `N-1`.

`--num-buckets` at `nnue-train` time must equal trailer `N` when quantile
binning is active.

### 2. Calibration is a separate offline step

`progress-kpabs-train` continues to emit v1 weights-only files. Quantile
thresholds are computed by `progress-bucket-survey --write-calibrated` from a
representative PSV sample and the chosen `progress.bin` weights.

Thresholds depend on `(weights, calibration PSV, N)`. Changing `N` requires
recalibration.

### 3. Backward compatibility

- v1 files: equal-width binning (unchanged behaviour).
- v2 files: quantile binning; `nnue-train` and survey auto-detect via trailer.
- `progress-kpabs-train --init-from` reads weights from v2 files (trailer ignored
  on write; output remains v1 unless recalibrated separately).

### 4. Engine follow-up (out of scope for tatara)

rshogi loads `progress.bin` via `LS_PROGRESS_COEFF` and must parse the v2
trailer and apply the same quantile bucket rule for inference to match training.
Until engine support lands, quantile-calibrated nets are training-only artifacts.

## Consequences

- Skewed equal-width distributions can be flattened on the calibration set.
- Calibration-set distribution drift still causes bucket skew on the full
  training PSV; recalibrate on representative data.
- Degenerate samples (all identical `p`) fail threshold validation instead of
  writing unusable trailers.
- ADR `2026-05-23-num-buckets-configurable.md` consequence “progress.bin format
  unchanged” is superseded for v2 trailer extension; v1 remains valid.
