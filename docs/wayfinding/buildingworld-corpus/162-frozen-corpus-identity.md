# #162 — Frozen corpus identity

The first **35,776 rows** of `data/real_massing_v1/real.h5` are now checked against an
immutable SHA-256 pin before row-indexed corpus reads. `FROZEN_SPLIT_N_TOTAL` and
`FROZEN_CORPUS_SHA256` live in `utils/frozen_corpus.py`.

Baseline recorded read-only on 2026-09-10 from the existing corpus (35,776 rows,
35,776 distinct `(bag_id, height_m)` pairs; all heights finite):

```text
c823614b3e6019216919490e5265557aa177d5e889f9023453dff2005d521322
```

Each row is serialized as its 64-byte, null-padded `bag_id` followed by its 4-byte
little-endian IEEE-754 `height_m`, with no padding between records. Records remain
in row order. The hash reads 2,432,768 bytes of metadata, never the SDF volumes.
Height byte order is normalized; incompatible column widths/types and non-finite
heights are rejected rather than silently converted.

## Behavior

- `open_real_corpus(path)` opens read-only and validates before returning a handle;
  failure closes the handle. `assert_frozen_corpus(handle)` checks an existing handle.
  Explicit exceptions keep the guard active under Python `-O`.
- Changed IDs, heights, reordered or missing historical rows, and inconsistent
  metadata column lengths stop the caller with a `#162` error. Restore the original
  prefix on failure. **Do not regenerate the baseline to accept a rebuilt corpus.**
- Appended rows are allowed. `Bag3dDataset` retains its historical validation/test
  permutation and adds appended rows only to training. The raw 2% test slice has
  715 rows; the recovered-surface/latent population historically used for evaluation
  contains 714. The vecset precompute and raw-corpus probes now name the frozen
  size explicitly rather than deriving the permutation from a larger corpus.
- Standalone source datasets retain their original splits. `Bag3dDataset` recognizes
  the combined corpus by the `real.h5` filename or multi-source `source_id` metadata,
  and validates again when a worker opens its lazy file handle.
- Surface extraction, vecset precompute, program recovery, height-field cache
  construction/reuse, raw-corpus evaluation/probes, and #153/#181 split loading use
  the guard. Vecset training and voxel-cache training/evaluation also check the raw
  corpus even when using existing derived caches. The precompute guard runs before
  codec loading or output creation.

Cached training/evaluation therefore requires the raw corpus to remain available:
the height-map paths use their configured `H5`, and vecset/voxel-cache training checks
the repository's `data/real_massing_v1/real.h5`. A cache alone cannot establish the
current raw corpus's identity. This change does not stamp checkpoint provenance
(#163), authenticate old cache contents, or check SDF/footprint geometry. Equal
`(bag_id, height_m)` pairs have equal identity even if their geometry differs.
The separate #153 proof split is unchanged; its loader checks identity before
constructing that split. No corpus/cache files were rewritten.

## Verification

```bash
env -u LD_PRELOAD -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  ./sdfusion/bin/python -m unittest scripts.foundations.test_frozen_corpus
```

Tests cover the serialization format and byte order, invalid metadata, drift and
row deletion, append-only split stability, legacy dataset loading, worker reopening,
cache reuse, and failure before precompute can load a codec or create output.
Acceptance tests copy only the live metadata into temporary files with placeholder
geometry; they skip when the real corpus is unavailable. Synthetic rejection tests
still run without it.
