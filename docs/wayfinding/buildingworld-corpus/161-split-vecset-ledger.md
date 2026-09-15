# #161 — Split the row/region/held_out/height_m ledger out of vecset_latents.h5

`train_height_map_generator.py`'s `build_cache` never touches a latent -- it only ever wanted four
small per-row facts (row identity, source region, held-out flag, height in metres) that happened to
live beside the 9.4 GB of 2048x64 fp16 Dora-VAE latents in `vecset_latents.h5`. That coupling meant a
new BuildingWorld row could not get a `held_out`/`region` assignment without either paying real
GPU-hours to Dora-encode it first, or writing a latent-less row into the latent store and risking
every other consumer of that file (the vecset-denoiser arms whose eval artifacts are pinned under
`execution/artifacts/`).

## What changed

The four columns are now also a standalone file, `data/real_massing_v1/corpus_ledger.h5`
(`scripts/foundations/corpus_ledger.py`): 471 KB against `vecset_latents.h5`'s 9.4 GB, and readable/
writable independently of the Dora codec.

- `corpus_ledger.read_ledger` / `write_ledger` / `append_ledger` -- read, wholesale-rewrite, and
  extend the ledger. `append_ledger` refuses to silently overwrite a row already present.
- `corpus_ledger.extract_from_vecset_latents` -- the one-time split itself: pulls `row`/`region`/
  `held_out`/`height_m` out of an existing `vecset_latents.h5`, never touching `latent`/`query_pos`.
- `precompute_vecset_latents.py --split_ledger` runs the extraction from the CLI (this is how the
  live `corpus_ledger.h5` was built, from the live `vecset_latents.h5`).
- `train_height_map_generator.py`'s `build_cache` now reads the ledger instead of opening
  `vecset_latents.h5` at all. A brand-new BuildingWorld row can get a `held_out`/`region`/`height_m`
  entry (once a future ticket, #177, decides its split) and be picked up by the next height-map
  cache rebuild -- with no Dora encoding involved.
- `precompute_vecset_latents.py`'s encode loop no longer decides `region`/`held_out`/`height_m`
  itself (dropping its own `test_indices`/`src_id` computation); it checks the whole batch of rows it
  is about to encode against the ledger UP FRONT, before the codec loads or any row is Dora-encoded,
  and refuses (`SystemExit`) the run if any of them has no ledger entry. This is deliberate: which
  held-out policy a BuildingWorld row gets is #177's decision, not this module's, so it is refused
  rather than guessed -- and refused before spending GPU time on it, not after. It still writes
  `region`/`held_out`/`height_m` into `vecset_latents.h5`
  alongside the latent, sourced from the ledger -- every existing reader of that file
  (`train_vecset.py`, `eval_massing_arms.py`, `prototype_voxel_editor.py`, the `probe_*` diagnostics,
  `run_aligned_retrain.py`) keeps working unmodified, because a vecset-denoiser consumer can only
  ever want a row that already has a latent, so there is nothing for it to be missing.
- `dedup_buildingworld_geometric.held_out_by_row` (flagged with a `#161:` TODO before this ticket)
  now reads the ledger instead of opening `vecset_latents.h5` for one `uint8` column.

## Why the latent store still carries its own copy

The alternative -- deleting `region`/`held_out`/`height_m` from `vecset_latents.h5` outright, and
switching `eval_massing_arms.pick_ids` and `train_vecset.LatentSet` to read the ledger instead -- was
checked, not just weighed in the abstract, and rejected: both functions are pinned by dedicated unit
tests (`test_eval_massing_arms.TestIdSet`, `test_train_vecset_solidity.TestLatentSetStorage`) that
build their OWN synthetic `latents.h5` fixtures with arbitrary row ids and embedded `held_out`/
`region` columns unrelated to the real corpus. `pick_ids` in particular has a deliberate legacy
fallback (`region is None`, exercised by
`test_ids_without_a_region_column_are_the_held_out_rows_ascending`) guarding the exact #71 incident
this file's own docstring describes: an ascending-row-order id list that turned out to be 100% one
source corpus, which voided two of this map's headline numbers. Rerouting either function through a
fixed-path external ledger would mean either breaking that tested, incident-scarred contract (a
synthetic fixture's rows are never going to be in `corpus_ledger.h5`) or growing both functions a
second, ledger-vs-embedded-column code path -- neither of which this ticket's "pure refactor... no
behavior change" mandate calls for, and both of which are strictly riskier than the alternative below.

Instead: those consumers are left reading `vecset_latents.h5`'s own copy, completely unmodified, and
the copy is now sourced from the ledger (by `precompute_vecset_latents.py`) rather than computed a
second, independent way. This is why the write side counts as "the vecset-denoiser pipeline...
read[s]" the new file even though the read side (`eval_massing_arms.py`, `train_vecset.py`) does not:
the ledger is upstream of, and the single source of truth for, every `region`/`held_out`/`height_m`
value that ends up in `vecset_latents.h5`, whether or not each individual downstream reader talks to
it directly. Those consumers are latent-scoped by construction -- a vecset denoiser has nothing to
train on for a row without a latent -- so leaving their read path alone carries no coupling to future
ledger growth: a ledger-only BuildingWorld row (#177) is invisible to them precisely because it was
never Dora-encoded, which is the corruption risk from option (b) in the original ticket, closed by
construction rather than by convention.

## Verification

The ledger was built for real from the live 9.4 GB `vecset_latents.h5`
(`precompute_vecset_latents.py --split_ledger`) and its four columns are byte-identical to the
source file's own `row`/`region`/`held_out`/`height_m` (checked directly, element-for-element).
`held_out_by_row()` against the live ledger reports 714 held-out rows, matching the pinned 714 used
throughout this project.

`train_height_map_generator.build_cache(force=True)`, pointed at a scratch path, was re-run end to
end against the new ledger and its output compared array-for-array against the pre-#161 cache
(`row`, `held`, `region`, `height_m`, `fp`, `target`, `y0`, `extent`, `ok`): identical.

```bash
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_corpus_ledger.py
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_precompute_vecset_latents.py
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_train_height_map_generator.py
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_dedup_buildingworld_geometric.py
env -u LD_PRELOAD ./sdfusion/bin/python scripts/foundations/test_frozen_corpus.py
```
