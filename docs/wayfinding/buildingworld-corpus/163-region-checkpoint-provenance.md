# #163 — Region and checkpoint provenance guards

The height-map generator now rejects every region ID outside `[0, N_REGIONS)` before building or
reusing its cache and before constructing conditioning channels. An unknown region can no longer
be encoded as a silent all-zero one-hot vector.

Every newly saved best or final-epoch checkpoint records a compatibility contract:

- `n_regions`: the region vocabulary size;
- `conditioning_channels`: the ordered semantic names of every model input channel;
- `corpus_identity_sha256`: SHA-256 of the sorted, unique cache row IDs; and
- `region_mapping_sha256`: SHA-256 of the versioned pipeline-to-region-id mapping, per #167's
  `scripts/foundations/source_provenance.py`. Checked on every load, cache or no cache, since the
  mapping hash needs no training cache to compute.

Training diagnostics and the generation service validate checkpoints through one loader. They
compare the region count and channel order with the running code, and compare the corpus hash when
a cache is available. A mismatch raises before model weights are used. Historical checkpoints have
none of these fields, so they remain usable but emit a `RuntimeWarning` stating that compatibility
cannot be verified. A checkpoint containing only part of the contract is treated as corrupt and is
rejected.

The corpus identity deliberately describes the cache's row set rather than its row order. Duplicate
row IDs are rejected because they would make a set hash ambiguous. #162 still guards the raw frozen
corpus independently; this checkpoint stamp binds a height-map model to the derived population it
was trained on.

## Verification

```bash
env -u LD_PRELOAD ./sdfusion/bin/python \
  scripts/foundations/test_train_height_map_generator.py
```

The tests cover invalid region values and column types, channel ordering, row-set hashing, partial
or mismatched provenance, and the explicit legacy-checkpoint warning.
