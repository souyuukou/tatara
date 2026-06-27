**English** | [日本語](progress-bin.ja.md)

# Game-progress buckets: preparing `progress.bin`

The `layerstack` architecture routes each position to one of N game-progress
buckets, and the bucket assignment is decided by `progress.bin` — a set of
KP-abs coefficients that estimate how far a position has advanced within its
game. This page covers how to train `progress.bin` and how to check the bucket
split it produces. The `simple` architecture has no buckets and does not need
it; for the overall training flow see
[docs/training-quickstart.md](training-quickstart.md).

The idea of learning game progress and assigning it to output buckets is based
on [a post by nodchip](https://nodchip.hatenablog.com/entry/2026/02/04/000000).

## Generating progress.bin

Train the progress coefficients with `progress-kpabs-train`.

> **Do not shuffle the data.** Pass `progress-kpabs-train`'s `--data` a PSV of
> **consecutive games** (positions in game order, with games following one
> after another). The progress coefficients learn "how far a position has
> advanced within a single game", and `progress-kpabs-train` reads the data one
> game at a time (detecting game boundaries by `game_ply`) and labels each
> position with its relative position within that game. With a shuffled PSV the
> game boundaries break, the labels become meaningless, and correct coefficients
> cannot be learned. As a general rule the main NNUE training (`nnue-train`)
> benefits from a *shuffled* PSV; progress training is the opposite, so do not
> reuse the same file for both.

Specify the total number of epochs with `--epochs`. Each epoch writes a
`<run-name>.e<N>.bin` checkpoint, and the final epoch is also written to the
`--output` path.

```bash
target/release/progress-kpabs-train \
  --data <path/to/consecutive-psv.bin> \
  --output output/progress/<run-name>.bin \
  --games-per-step 1024 --epochs 5
```

Which epoch's output (`<run-name>.e<N>.bin`) to use takes some trial and error
(progress.bin is a coefficient that decides bucket assignment and is independent
of NNUE training convergence, so how many epochs you need is data-dependent).

To help decide which epoch to use, add `--val-fraction <f>` (e.g. `0.05`): it
holds out roughly that fraction of games — every Nth game in input order, since
the data must stay in consecutive-game order — and reports a held-out `val_loss`
at the end of each epoch. This adds one extra pass over the data per epoch.

Treat `val_loss` as a sanity check and an epoch-selection hint, not a precise
quality score. The progress model is simple — per-feature weights summed and
passed through a sigmoid — so it overfits little: a small `train_loss`/`val_loss`
gap is normal, and a clearly widening gap is what to watch for. Because the real
goal is a good bucket split, which a plain MSE only approximates, prefer an
epoch where `val_loss` levels off over chasing its exact minimum, and judge the
final `progress.bin` by the strength of the LayerStack NNUE trained with it.

## Checking the bucket distribution

`progress-bucket-survey` reports how a `progress.bin` assigns positions to the
progress buckets. A roughly even spread is healthy; if one bucket dominates, the
LayerStack output buckets are trained on very uneven amounts of data.

```bash
cargo build --release -p progress-bucket-survey
target/release/progress-bucket-survey \
  --data <path/to/consecutive-psv.bin> \
  --progress output/progress/<run-name>.e5.bin \
  --samples 200000
```

It prints a per-bucket count and percentage plus the top bucket's share. Only
one `progress.bin` can be loaded per run, so to compare epochs run it once per
`<run-name>.e<N>.bin` and compare the outputs.

By default, bucket assignment uses **equal-width** binning on `p ∈ [0, 1]`:
`floor(p × N)`. If the spread is badly skewed, use quantile calibration below.

## Equal-frequency calibration (v2 progress.bin)

When equal-width bins concentrate too many positions in a few buckets, calibrate
quantile thresholds from a representative PSV sample and embed them in
`progress.bin` as a trailer (magic `PRGQ`, format version 1). See
[docs/decisions/2026-06-27-progress-quantile-binning.md](decisions/2026-06-27-progress-quantile-binning.md).

```bash
target/release/progress-bucket-survey \
  --data <path/to/consecutive-psv.bin> \
  --progress output/progress/<run-name>.e5.bin \
  --num-buckets 32 --samples 500000 \
  --write-calibrated output/progress/<run-name>.e5.q32.bin
```

This copies the weights from `--progress`, computes `(N-1)` quantile thresholds
from the sampled positions, and writes a v2 file. The tool prints a post-
calibration histogram on the same sample (each bucket should be roughly `100/N`
percent). Use the calibrated file with matching `--num-buckets`:

```bash
target/release/nnue-train layerstack \
  --progress-coeff output/progress/<run-name>.e5.q32.bin \
  --num-buckets 32 ...
```

`--num-buckets` must match the `N` baked into the trailer. Recalibrate when
changing `N` or when the training PSV distribution differs materially from the
calibration sample.

Once you have a `progress.bin` you are happy with, pass it to `nnue-train` via
`--progress-coeff` when training a `layerstack` net (see
[docs/training-quickstart.md](training-quickstart.md)).
