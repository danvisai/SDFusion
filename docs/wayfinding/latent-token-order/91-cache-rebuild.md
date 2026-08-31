<!-- RECOVERED FROM THE ISSUE TRACKER, 2026-08-14. -->

> **Recovered document.** The original asset `91-cache-rebuild.md` / `91-rebuild.log` was written and committed on another
> machine (`9b9bdda`) and was never pushed; it does not exist in this repository or on origin.
> This file is its findings reconstructed verbatim from GitHub issue #91 — the ticket body
> and every resolution comment. The note that the implementation was missing was true when this
> record was recovered; the 2026-08-15 resolution below records the successful rebuild. See
> `RECOVERY.md`.


# #91 — Rebuild the blockout cache in the aligned order, with a write-time guard

*State: resolved 2026-08-15 · opened 2026-08-09*


## Ticket

Part of #87

## Task

Rebuild the latent caches carrying `query_pos` (#88), and derive the **aligned** envelope cache from
them.

## 🔑 One encode pass gives BOTH arms of #92, for free

Alignment is a **permutation of the same tokens** — no re-encoding, because #89 measured the decoder as
permutation-invariant (occupancy IoU 1.000000). So a single pass produces two caches that are *literally
the same numbers in a different order*:

| cache | contents |
|---|---|
| `vecset_blockout_latents_v2.h5` | new draws, tokens **as encoded** |
| `vecset_blockout_latents_v2_aligned.h5` | the same latents, tokens **permuted** into the matched order |

That is a textbook single-variable comparison: identical data, identical seeds, identical everything
except token order. **It removes the confound that #92 originally carried** (its control was
`v4_surf@240k`, trained on the *old* cache, so a result would have mixed alignment with a different
surface draw and been unattributable).

The real-side cache is rebuilt once (`vecset_latents_v2.h5`) and shared by both arms — only the envelope
side gets permuted, since alignment maps envelope tokens onto real ones.

## The method, chosen by #90

**`models.token_alignment.greedy_match`, default `candidates=256`.** Measured cosine **0.5387** against
Hungarian's 0.5106 and Morton's 0.2112, at **~4.9 h** over the corpus — **one** pass, since only the
envelope side is permuted onto a real side that keeps its own order.

⚠️ **Expect 0.54, not 0.73.** #89's headline used a many-to-one mapping that cannot reorder a cache. If
the rebuilt cache scores far from 0.54, suspect the rebuild rather than the method. ⚠️ ~63% of tokens
will land beyond one voxel from their partner; that is the two surfaces genuinely differing, not a bug.

## Sizing

~2 h per encode pass (#78 measured 1.95 h for 35,623), so ~4 h for both sides. Deriving the aligned
cache is a permutation: minutes. `query_pos` adds 437 MB per cache at fp16.

## The guard

`verify_positions` (#88) already refuses a cache whose positions are not token-for-token its latents',
by re-encoding through the same `encode_row` the write loop uses. **Extend it for the aligned cache**:
after permuting, the stored positions must be permuted identically, so the check must still pass — if it
does not, the permutation was applied to one array and not the other, which is precisely the silent
failure #78's precedent exists for.

⚠️ Also assert the aligned cache is a **permutation** of the unaligned one, not a different set of
numbers: sorted token multisets must match per row.

## ⚠️ Do not overwrite

The existing caches stay. The shipped `v4_surf` checkpoint was trained on them and remains #92's
reference point for the *old* regime.

## Judged on

Both caches written and passing the guards; the aligned one demonstrated to be a permutation of the
unaligned one; `probe_token_order.py --from_cache` re-run on the aligned pair showing the element-wise
distance collapse from ~1.09 toward the ~0.037 matched floor.


---

## Comment — danvisai, 2026-08-10

## Done — three caches on disk, and #92's confound is gone

Writeup `docs/wayfinding/latent-token-order/91-cache-rebuild.md`, log `91-rebuild.log`
(`f4f1087`, `9b9bdda`). 11.3 h wall clock.

| cache | rows | contents |
|---|---|---|
| `vecset_latents_v2.h5` | 35,623 | real side, `query_pos`, the reference order |
| `vecset_blockout_latents_v2.h5` | 35,623 | envelope **as encoded** — #92's **control** |
| `vecset_blockout_latents_v2_aligned.h5` | 35,623 | the same latents **permuted** — #92's **treatment** |

714 held out in each. ⚠️ Original caches untouched — `v4_surf` trained on them.

🔑 **The last two are literally the same numbers in a different order**, so #92's arms differ in token
order and nothing else. The plan I first chartered used `v4_surf@240k` as control, which trained on the
*old* cache and would have mixed alignment with a different surface draw.

### Acceptance — read from disk, nothing encoded (n=60)

| cache | elementwise | matched floor | % of the way to random |
|---|---|---|---|
| unaligned | 1.1138 | 0.0423 | **99.7%** |
| **aligned** | **0.1904** | 0.0423 | **13.7%** |

Corpus-wide token cosine **0.5172**. Against the as-encoded **0.0405** (⚠️ #90's figure, on 102
held-out buildings — different sample, so the 12.8× is indicative not paired).

### Guards

| | real | envelope |
|---|---|---|
| frame (fp-IoU vs own footprint) | 0.9970 | 1.0000 |
| positions (stored vs re-encoded) | 0.0001 | 0.0002 |
| — same building **shuffled** | 1.0557 | 1.0892 |
| — a **different** building | 1.1247 | 1.1138 |

Plus, on the aligned cache: **35,623/35,623 rows are token-wise permutations**, **35,623/35,623 were
reordered**, positions followed at **0.1762**, cosine 0.5172.

### ⚠️ The guard could not have proven what I claimed it proved

The cache is correct. It was built under a check that could not establish that, and the review found
three defects:

1. **The identity check tested one row** — `idx` was a loop variable read after the loop, so "nothing
   was aligned" was asserted about building 35,622 alone.
2. **The permutation check was at the wrong granularity** — 131,072 scalars per row rather than 2,048
   token vectors, so two tokens could swap channel values and pass.
3. **This ticket's explicit ask was skipped** — `verify_positions` was never extended to the aligned
   cache, the one failure the whole effort guards against.

All three fixed, and re-run against the **already-built** cache via a new `--verify_only` mode rather
than costing another 5.68 h. They pass. Also fixed a ~70 GB memory hazard in the old guard's float32
casts, on a codebase that had already fixed `LatentSet` for exactly that.

### Timings, for the next sizing

real 2.04 h · envelope **3.04 h** · alignment 5.68 h. The envelope side is slower because it adds an
EDT and a marching-cubes pass per building — #78's 1.95 h precedent covers the real side only and
under-budgets this by an hour.

### Also delivered: checkpoint monitoring for #92

`scripts/foundations/watch_checkpoints.py` renders GT | envelope | model on fixed, region-stratified
buildings for every checkpoint as it lands, plus a `curve.json`. Runs outside training. **`vs_input` is
drawn into every caption** — a checkpoint that looks perfect by returning its own input is invisible in
a picture, and that is the failure this map exists to fix.

⚠️ Fixed a render fault the codebase already documents: GT was isosurfaced from **binary occupancy**,
which `mesh_sdf_surface`'s docstring forbids, making ground truth render as a staircased silhouette
beside two properly shaded panels.

### ⚠️ What this establishes

The training **target** is now correct. **Not** that it produces better massing — #92 is the test, and
its bar was fixed before any of this was built.


---

## Comment — danvisai, 2026-08-12

Reopening: this ticket's resolution was implemented on a cloud A100 instance that was lost before the resolving commits (cited in the comment above) were pushed. None of those commits exist on any branch or PR in this repo — verified against `git log --all` and `git ls-remote`. The written analysis and decisions above are intact and should be treated as the spec; the implementation, and #91's rebuilt caches specifically, need to be redone from scratch.


---

## Comment — Codex, 2026-08-15

## Redone — current corpus rebuilt, guarded, aligned, and exhaustively verified

The lost implementation was recovered and hardened in `74ba808` (transactional, resumable cache
writes) and `d6fcc8b` (bounded envelope-encode memory). The detached rebuild then completed at
2026-08-15T10:46:03Z.

| cache | rows | held out | contents |
|---|---:|---:|---|
| `vecset_latents_v2.h5` | 36,818 | 715 | real side, reference order |
| `vecset_blockout_latents_v2.h5` | 36,818 | 715 | envelope as encoded — #92 control |
| `vecset_blockout_latents_v2_aligned.h5` | 36,818 | 715 | the same envelope tokens, permuted — #92 treatment |

All three caches have identical, unique row IDs and complete seven-dataset schemas. The original
`vecset_latents.h5` and `vecset_blockout_latents.h5` remain untouched.

### Encode guards

| guard | real | envelope |
|---|---:|---:|
| frame fp-IoU median / min (n=16) | 0.9972 / 0.9819 | 1.0000 / 1.0000 |
| stored vs re-encoded positions (n=8) | 0.0002 | 0.0002 |
| same-building shuffle | 1.0724 | 1.0893 |
| another building | 1.1175 | 1.0978 |

### Alignment acceptance — every row read back from disk

- token-wise permutations: **36,818 / 36,818**;
- reordered: **36,818 / 36,818**;
- positions followed the latent permutation exactly: **0.0** error;
- query-position distance: **1.135835** elementwise -> **0.183678** aligned, against a
  **0.037845** nearest-neighbour floor;
- aligned distance is **13.2818%** of the random span, consistent with the lost run's 13.7%.

The current `--from_cache` probe (now on `precompute_vecset_latents.py`) independently reproduces the
collapse on the same deterministic n=64 sample: unaligned **1.1508** versus **0.0407** matched
(**99.9%** of random), aligned **0.1944** versus the same **0.0407** floor (**13.8%** of random).

Artifacts: `execution/artifacts/align_cache_v2.json` and
`execution/artifacts/91-cache-rebuild.log`. The exhaustive guard is implemented by
`align_cache.py --verify_only`; it compares whole 64-channel token vectors, proves a permutation for
every row, and proves `query_pos` followed the identical permutation.
