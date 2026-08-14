"""Encode the recovered LoD2 surfaces into vecset latents, once, for diffusion training.

Encoding is the expensive part and the corpus is fixed, so it is done ahead of time rather than in the
data loader. Everything goes through the `ShapeCodec` contract (spec #68) rather than reaching into the
autoencoder directly -- the same calls the diffusion will use, so a codec swap changes one flag here and
nothing downstream.

Also caches the conditioning the generator needs beside each latent -- footprint, height, region -- so
training reads one file and never touches the source corpus.

Usage:
    precompute_vecset_latents.py --limit 256          # smoke
    precompute_vecset_latents.py                       # the whole corpus
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.shape_codec import Building, DoraCodec                       # noqa: E402
from scripts.foundations.baseline_gate_eval import mesh_sdf_surface       # noqa: E402
from scripts.foundations.vecset_ceiling_probe import TRUNC, verts_to_world  # noqa: E402
from scripts.foundations.dora_roundtrip_probe import load_dora, H5       # noqa: E402
from scripts.foundations.dora_frozen_gate import load_surfaces           # noqa: E402
from scene.surface_sampling import to_array_frame                        # noqa: E402
from utils.numeric_guard import check_numpy  # noqa: E402
from scripts.foundations.vecset_ceiling_probe import test_indices        # noqa: E402

OUT = REPO / "data/real_massing_v1/vecset_latents.h5"


def _stratified_rows(surf: dict, rows: list, n: int) -> list:
    """`n` rows, round-robin over the three source corpora.

    ⚠️ **Ascending row order tracks the SOURCE CORPUS**, so a plain prefix is one country. That trap
    voided three headline figures once (the pinned 48 were 100% Dutch) and #95 walked into it a second
    time by re-deriving its own sample. A cache built for a probe therefore stratifies at build time,
    so no downstream reader has to remember to.
    """
    by_src: dict = {}
    for r in rows:
        by_src.setdefault(surf[r][2], []).append(r)
    order = sorted(by_src)
    out: list = []
    for k in range(max(len(v) for v in by_src.values())):
        for s in order:
            if k < len(by_src[s]) and len(out) < n:
                out.append(by_src[s][k])
        if len(out) >= n:
            break
    return sorted(out)


class IncrementalCache:
    """Append rows to the output file as they are encoded, so a crash costs minutes, not hours.

    The corpus encode is ~15 h on this box. Buffering it all in memory and writing once at the end
    means any failure at hour 14 -- OOM, a driver reset, a power cut -- loses the whole run, and the
    `[skip]` path makes a partial in-memory result unrecoverable anyway. Datasets are created with an
    unbounded first axis and extended every `flush_every` rows; `--resume` reads back the rows already
    present and encodes only the remainder.

    ⚠️ Resume is safe **only because encoding is per-row reproducible** (`encode_row` reseeds from the
    corpus row), so a row encoded in the second half of a resumed run is bit-identical to the one the
    uninterrupted run would have written. That property is #88's, and this depends on it.
    """

    SPECS = {
        "latent": (np.float16, "lzf"),
        "query_pos": (np.float16, "lzf"),
        "footprint": (np.uint8, "lzf"),
        "height_m": (np.float32, None),
        "region": (np.int32, None),
        "row": (np.int32, None),
        "held_out": (np.uint8, None),
    }
    SCHEMA_VERSION = 1

    def __init__(self, path: str, resume: bool, flush_every: int = 200):
        import h5py

        self.path, self.flush_every = path, flush_every
        self.buf: dict = {k: [] for k in self.SPECS}
        self.done: set = set()
        self.closed = False
        self.flush_failed = False
        mode = "a" if (resume and Path(path).exists()) else "w"
        self.f = h5py.File(path, mode)
        if mode == "w":
            # `committed_rows` is advanced only after every column has reached disk. On resume it is
            # therefore the last known-good boundary even if the process died halfway through a
            # multi-column append.
            self.f.attrs["incremental_schema"] = self.SCHEMA_VERSION
            self.f.attrs["committed_rows"] = 0
            self.f.flush()
        else:
            schema = int(self.f.attrs.get("incremental_schema", 0))
            if schema != self.SCHEMA_VERSION:
                self.f.close()
                raise SystemExit(
                    f"[precompute] cannot resume {path}: it is not an incremental cache (schema "
                    f"{schema}, need {self.SCHEMA_VERSION}). Choose a new --out; an old fixed-size "
                    "cache cannot be safely extended."
                )
            committed = int(self.f.attrs.get("committed_rows", 0))
            present = [k for k in self.SPECS if k in self.f]
            if present and len(present) != len(self.SPECS):
                # A first append may have died while creating the datasets. Discard it back to the
                # committed boundary; missing columns will be recreated by the next flush.
                if committed:
                    self.f.close()
                    raise SystemExit(
                        f"[precompute] cannot resume {path}: only {len(present)}/{len(self.SPECS)} "
                        f"columns exist below a non-zero committed boundary ({committed} rows)"
                    )
                for k in present:
                    self.f[k].resize(committed, axis=0)
            elif present:
                for k in self.SPECS:
                    actual = self.f[k].shape[0]
                    if actual < committed:
                        self.f.close()
                        raise SystemExit(
                            f"[precompute] cannot resume {path}: {k} has {actual} rows, "
                            f"below the committed boundary {committed}"
                        )
                    self.f[k].resize(committed, axis=0)
            self.f.flush()
            if "row" in self.f:
                self.done = {int(r) for r in self.f["row"]}
            print(f"[precompute] resuming: {len(self.done)} rows already in {path}")

    def add(self, **cols) -> None:
        for k, v in cols.items():
            self.buf[k].append(v)
        if len(self.buf["row"]) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        n = len(self.buf["row"])
        if not n:
            return
        arrays = {k: np.asarray(self.buf[k], dt) for k, (dt, _) in self.SPECS.items()}
        if any(len(arr) != n for arr in arrays.values()):
            self.flush_failed = True
            raise ValueError("every cache column must receive exactly one value per row")

        committed = int(self.f.attrs["committed_rows"])
        try:
            for k, (_, comp) in self.SPECS.items():
                arr = arrays[k]
                if k not in self.f:
                    self.f.create_dataset(k, data=arr, maxshape=(None,) + arr.shape[1:],
                                          compression=comp, chunks=True)
                else:
                    d = self.f[k]
                    d.resize(committed + n, axis=0)
                    d[committed:] = arr
            self.f.flush()
            self.f.attrs.modify("committed_rows", committed + n)
            self.f.flush()
        except BaseException:
            # Keep the on-disk boundary honest even when the process survives a failed flush. A later
            # `--resume` can restart from `committed`; this process must not retry on uneven columns.
            self.flush_failed = True
            for k in self.SPECS:
                if k in self.f:
                    self.f[k].resize(committed, axis=0)
            self.f.flush()
            raise

        self.done.update(int(r) for r in arrays["row"])
        for values in self.buf.values():
            values.clear()

    def close(self, attrs: dict | None = None) -> None:
        if self.closed:
            return
        try:
            if not self.flush_failed:
                self.flush()
            for k, v in (attrs or {}).items():
                self.f.attrs[k] = v
        finally:
            self.f.close()
            self.closed = True


def encode_row(codec, building: Building, row: int):
    """Encode one building reproducibly -> (latent, query positions).

    **Per-row seeding is the point.** A codec seeded once at construction makes row `k`'s draw a
    function of every row before it, so a latent cannot be reproduced without replaying the whole
    pass -- and the `[skip]` path below desynchronises everything after any failure. Seeding from the
    corpus row makes each building independent, which is what lets `verify_positions` re-encode a
    sample and what lets a later pass rebuild one row without rebuilding the cache (#88).
    """
    codec.reseed(row)
    return codec.encode_with_positions(building)


def _building_for_row(source_h5, surf: dict, row: int, blockout: bool) -> tuple[Building, str]:
    """Reconstruct the exact encoder input for one corpus row."""
    v, fc, src = surf[row]
    if blockout:
        # The same extrusion the generator is handed at inference -- encoding it here is what
        # removes the train/inference distribution gap the diagnostic identified.
        from scripts.foundations.eval_massing_arms import blockout_sdf

        g = np.asarray(source_h5["sdf"][row], np.float32)
        ys = np.nonzero((g <= 0).any(axis=(0, 2)))[0]
        if len(ys) == 0:
            raise ValueError("empty source SDF")
        bo = blockout_sdf(np.asarray(source_h5["footprint"][row], np.uint8),
                          int(ys.min()), int(ys.max()))
        if bo is None:
            raise ValueError("empty blockout")
        bv, bf = mesh_sdf_surface(np.clip(bo, -TRUNC, TRUNC))
        if bv is None:
            raise ValueError("blockout has no surface")
        return Building(verts=verts_to_world(bv), faces=bf), src

    # The corpus is in Frame-N; everything else in this pipeline -- the blockout above,
    # `grid_points`, `decode_grid`, the eval harness -- speaks the ARRAY frame. Encoding the corpus
    # verts raw put every real latent in a frame x<->z-transposed from its own aligned blockout, so
    # pair training learned "blockout -> transposed building" (#70).
    av, af = to_array_frame(v, fc)
    return Building(verts=av, faces=af), src


def _pos_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-token L2 between two (T, 3) position sets, compared index for index."""
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32), axis=-1).mean())


def verify_positions(codec, blds: dict, rows: list, P: np.ndarray, tol: float = 0.1) -> None:
    """Refuse to write a cache whose stored positions are not the ones the encoder actually chose.

    ⚠️ **The obvious guard does not work, and #89 is the reason.** Decoding a latent at its own stored
    positions and requiring |sdf| ~ 0 reads convincingly (it separates buildings), but it **cannot see
    a within-building permutation** -- the decoder is permutation-invariant at occupancy IoU 1.000000,
    which is exactly the property that makes token order pure gauge. Token *i* <-> position *i* is
    precisely what #90/#91 consume, so a guard blind to it is no guard.

    So this **re-encodes** sampled rows through `encode_row` -- the same function the write loop uses,
    so it cannot validate its own copy of the answer -- and scores the stored positions against three
    references: themselves, themselves shuffled, and another building. Shuffled is the discriminating
    one; it is invisible to the decode-based check.
    """
    if not blds:
        print("[verify_pos] SKIPPED -- no rows sampled")
        return

    idx_of = {r: i for i, r in enumerate(rows)}
    rng = np.random.default_rng(0)
    same, shuf, cross = [], [], []
    for r, b in blds.items():
        i = idx_of[r]
        _, re = encode_row(codec, b, r)
        stored = np.asarray(P[i], np.float32)
        same.append(_pos_dist(stored, re))
        shuf.append(_pos_dist(stored[rng.permutation(len(stored))], re))
        j = (i + 1) % len(P)
        if j != i:
            cross.append(_pos_dist(np.asarray(P[j], np.float32), re))

    s, sh = float(np.median(same)), float(np.median(shuf))
    c = float(np.median(cross)) if cross else float("nan")
    print(f"[verify_pos] n={len(same)}  stored-vs-re-encoded {s:.4f}  shuffled {sh:.4f}  "
          f"another building {c:.4f}")
    if not s < tol * sh:
        raise SystemExit(f"[verify_pos] FAILED: stored positions {s:.4f} are not tightly reproduced "
                         f"(shuffled reference {sh:.4f}); the cache is NOT written")


def verify_frame(codec, L: np.ndarray, fps: np.ndarray, n: int = 16, tol: float = 0.85) -> None:
    """Refuse to write a cache whose latents decode into the wrong frame.

    Decode sampled latents and check they reproduce **their own footprints**. A frame error --
    transpose, flip, axis swap -- moves the mass off the footprint and tanks this instantly, while a
    correct cache scores ~1.0 because the codec round-trips at ~0.999.

    This exists because the bug class has now bitten twice and **both times the existing verification
    passed**: #62 aligned surfaces at IoU 1.0000 while every normal was inverted (it validated position,
    not orientation), and #70 found every real latent x<->z transposed with nothing checking frame at all.
    Costs seconds against a multi-hour encode, and it is checked against the footprint stored *beside each
    latent in this very file*, so it cannot drift out of sync with what it validates.

    ⚠️ **Gates on the MEDIAN, not the minimum** -- an earlier version used the minimum and cried wolf on
    the known-good cache. Some real buildings genuinely do not fill their rasterised footprint in
    projection (overhangs, setbacks, courtyards): across the 48 held-out buildings of #71's baseline the
    median is 0.9967 but the floor is **0.7639**, with 2 of 48 under 0.85. At ~4% below tolerance a
    4-sample minimum fails spuriously about one call in five. The median cannot be moved by those
    outliers and still catches a frame error decisively -- the transposed cache scored 0.1735 median,
    the fixed one 0.9966. The minimum is still printed, and a *separate*, much lower floor catches the
    "everything is wrong" case without inheriting the false-positive rate.
    """
    import torch
    from scripts.foundations.baseline_gate_eval import fp_iou
    from scripts.foundations.vecset_ceiling_probe import RES

    if n <= 0:
        print("[verify] SKIPPED -- no frame guard was run on this cache")
        return

    # `ShapeCodec._device` reads the LATENT's device, not the model's, so it cannot place the latent for
    # us. Find the codec's own module instead; the contract does not expose one, so this stays local.
    dev = "cpu"
    for attr in vars(codec).values():
        if isinstance(attr, torch.nn.Module):
            dev = next(attr.parameters()).device
            break

    idx = np.linspace(0, len(L) - 1, min(n, len(L))).astype(int)
    scores = []
    for i in idx:
        z = torch.from_numpy(L[i].astype(np.float32))[None].to(dev)
        fld = codec.decode_grid(z, RES).cpu().numpy()[0, 0]
        scores.append(fp_iou(fld <= 0, fps[i]))
    lo, med = float(np.min(scores)), float(np.median(scores))
    floor = 0.40          # a frame error puts EVERY sample well under this (transposed range 0.17-0.42)
    print(f"[verify] fp-IoU of decoded latent vs its own footprint on {len(idx)} samples: "
          f"median {med:.4f} (need >= {tol})  min {lo:.4f} (floor {floor})")
    if med < tol or lo < floor:
        why = (f"median {med:.4f} < {tol}" if med < tol
               else f"a sample scored {lo:.4f}, under the {floor} floor")
        raise SystemExit(
            f"[verify] FRAME CHECK FAILED ({why}). The latents do not decode onto their own "
            f"footprints, so the mesh frame is wrong -- see scene.surface_sampling.to_array_frame. "
            f"NOT writing the cache.")


def _unit(x: np.ndarray) -> np.ndarray:
    """Row-normalise, so token agreement is measured as cosine and not as latent scale."""
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


def _unit_t(dset_row):
    """Read one cached row and row-normalise it **in torch**.

    In torch because the numpy form was where the numpy-2.2.6-on-3.14 defect showed itself (0.6011
    vs 0.0045 on row 9 of the smoke pair). The defect is fixed; this stays as the second
    implementation `_verify_agrees` checks against, which is worth more than saving one conversion.
    """
    import torch

    return torch.nn.functional.normalize(torch.tensor(np.array(dset_row, np.float32)), dim=-1)


def _nearest(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """For each row of `pa`, the index of the nearest row of `pb`.

    A KD-tree rather than the obvious `cdist().argmin()`: it avoids a (2048, 2048, 3) float32
    temporary, and it is faster.

    🔑 **This function used to carry a long warning about numpy corrupting live arrays. The cause is
    now known and fixed: numpy 2.2.6 on Python 3.14, a pair numpy never supported** (3.14 support
    landed in numpy 2.3). It silently turned matched-token agreement of 0.6660 into 0.0057. See
    `utils/numeric_guard.py`, which refuses to start a long run on such a pair, and
    `docs/wayfinding/latent-token-order/RECOVERY.md` for how it was found.
    """
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(pb, np.float64)).query(np.asarray(pa, np.float64))[1]


def _row_agreement(a, b, i: int, j: int, rng) -> tuple:
    """One building's (as-encoded, matched, random) token agreement, read fresh.

    ⚠️ **Everything this touches is read inside this call and dies with it, on purpose.** Holding the
    two latents across the surrounding loop's allocations is what exposed the corruption described in
    `_nearest`: the arrays were byte-identical to their source when assigned and wrong by the time
    they were used, turning 0.6660 into 0.0057. Keeping the lifetime inside one small frame is the
    mitigation that actually holds; `_verify_agrees` re-reads from disk to confirm it per run.
    """
    import torch

    za = _unit_t(a["latent"][i])
    zb = _unit_t(b["latent"][j])
    m = _nearest(np.array(a["query_pos"][i], np.float32), np.array(b["query_pos"][j], np.float32))
    perm = rng.permutation(zb.shape[0])
    pa = torch.tensor(np.array(a["query_pos"][i], np.float32))
    pb = torch.tensor(np.array(b["query_pos"][j], np.float32))
    return (float((za * zb).sum(-1).mean()),
            float((za * zb[m]).sum(-1).mean()),
            float((za * zb[perm]).sum(-1).mean()),
            float((pa - pb).norm(dim=-1).mean()),
            float((pa - pb[m]).norm(dim=-1).mean()),
            float((pa - pb[perm]).norm(dim=-1).mean()))


def _verify_agrees(a, b, i: int, j: int, claimed: float, tol: float = 1e-3) -> None:
    """Recompute one row from disk through torch, and refuse to report a disagreement.

    Deliberately re-reads rather than reusing the caller's arrays: the failure mode is the *arrays*
    going bad, so a check that shares them agrees with the corruption instead of catching it. That
    mistake was made once here already -- it reported 0.0045 vs -0.0047 and called it consensus.
    """
    ta = _unit_t(a["latent"][i])
    tb = _unit_t(b["latent"][j])
    m = _nearest(np.array(a["query_pos"][i], np.float32), np.array(b["query_pos"][j], np.float32))
    second = float((ta[None] @ tb[m][None].transpose(1, 2)).diagonal(dim1=1, dim2=2).mean())
    if abs(second - claimed) > tol:
        raise SystemExit(f"[from_cache] ABORTED: two independent paths disagree on row {i} "
                         f"({claimed:.4f} vs {second:.4f}) -- the memory-corruption signature "
                         f"documented in _nearest. The numbers cannot be trusted; do not report them")


def measure_from_cache(real: str, blockout: str, n: int = 64, seed: int = 0) -> dict:
    """Reproduce the token-order measurement from disk, encoding nothing (#88's acceptance).

    Three token agreements between a building's real latent and its blockout partner: **as-encoded**
    (token i to token i, which is what the pair loss actually bridges), **matched** (token i to the
    blockout token whose query position is nearest), and **random** (a shuffled pairing, the floor).
    The premise holds if as-encoded sits at the random floor while matched is far above it.

    Reported as mean per-token **cosine**, which is what #89/#90 measured and therefore the only
    scale on which surviving reference values exist: as-encoded **0.0405**, nn bound **0.7079**.

    🔑 **The premise doc's figures were QUERY-POSITION distances, not latent distances**, which is
    why they are reported on a second line. `docs/research/latent-token-order.md` was lost with the
    machine that wrote it, so its metric had to be re-derived; on the smoke pair this form gives
    elementwise **1.0929** against the doc's **1.093**, matched 0.038 against 0.037, and the two are
    the same measurement. Read on the latent scale those numbers look unreproducible, which is how
    the confusion started -- a latent distance of 0.037 would mean the blockout token equals the real
    token, and #90 already showed the two surfaces genuinely differ.

    ⚠️ `matched` is `argmin` over positions -- many-to-one, so it is a *bound*, not a usable
    reordering. #90 measured what survives an actual bijection (**0.5387**), and that is the number
    to plan against.
    """
    import h5py

    with h5py.File(real, "r") as a, h5py.File(blockout, "r") as b:
        for f, name in ((a, real), (b, blockout)):
            if "query_pos" not in f:
                raise SystemExit(f"{name} has no query_pos -- it predates #88 and cannot be matched; "
                                 "rebuild it (positions are not recoverable after the fact)")
        ra, rb = np.asarray(a["row"]), np.asarray(b["row"])
        ia = {int(r): i for i, r in enumerate(ra)}
        ib = {int(r): i for i, r in enumerate(rb)}
        common = np.intersect1d(ra, rb)
        rng = np.random.default_rng(seed)
        pick = rng.choice(common, size=min(n, len(common)), replace=False)

        ae, mt, rd, pe, pm, pr = [], [], [], [], [], []
        for r in pick:
            i, j = ia[int(r)], ib[int(r)]
            e, m_, d, qe, qm, qr = _row_agreement(a, b, i, j, rng)
            ae.append(e), mt.append(m_), rd.append(d)
            pe.append(qe), pm.append(qm), pr.append(qr)
            if len(mt) == 1:
                _verify_agrees(a, b, i, j, mt[0])

    out = {"n": len(pick), "as_encoded": float(np.mean(ae)), "matched": float(np.mean(mt)),
           "random": float(np.mean(rd))}
    out["pct_of_random"] = 100.0 * (out["matched"] - out["as_encoded"]) / max(
        1e-9, out["matched"] - out["random"])
    out.update(pos_elementwise=float(np.mean(pe)), pos_matched=float(np.mean(pm)),
               pos_random=float(np.mean(pr)))
    out["pos_pct_of_random"] = 100.0 * (out["pos_elementwise"] - out["pos_matched"]) / max(
        1e-9, out["pos_random"] - out["pos_matched"])
    print(f"[from_cache] n={out['n']}  token cosine: as-encoded {out['as_encoded']:.4f} · "
          f"matched {out['matched']:.4f} · random {out['random']:.4f}  -> as-encoded is "
          f"{out['pct_of_random']:.1f}% of the way from matched to random")
    print(f"             query-position distance: elementwise {out['pos_elementwise']:.4f} · "
          f"matched {out['pos_matched']:.4f} · random {out['pos_random']:.4f}  -> "
          f"{out['pos_pct_of_random']:.1f}% of the way to random "
          f"(premise doc: 1.093 · 0.037 · 1.089 -> 100.3%)")
    return out


def main() -> None:
    check_numpy()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--start", type=int, default=0,
                     help="row offset into the sorted id list, for chunked runs")
    ap.add_argument("--n_coarse", type=int, default=8192)
    ap.add_argument("--n_sharp", type=int, default=8192)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", type=int, default=4,
                    help="samples for the write-time frame guard; 0 disables it (don't)")
    ap.add_argument("--verify_tol", type=float, default=0.85,
                    help="minimum fp-IoU of a decoded latent against its own footprint")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: keep the rows already in --out and encode "
                         "only the rest (safe because encoding is per-row reproducible, #88)")
    ap.add_argument("--stratify", type=int, default=0,
                    help="build a sample of N rows, round-robin over sources, instead of a prefix "
                         "(a prefix is one country -- see _stratified_rows)")
    ap.add_argument("--verify_pos", type=int, default=4,
                    help="rows to RE-ENCODE to prove the stored query positions are the encoder's own "
                         "(0 disables); a decode-based check cannot see a token permutation")
    ap.add_argument("--from_cache", nargs=2, metavar=("REAL", "BLOCKOUT"),
                    help="encode nothing: reproduce the token-order measurement from two caches")
    ap.add_argument("--pairs", type=int, default=64,
                    help="--from_cache: buildings to measure")
    ap.add_argument("--blockout", action="store_true",
                    help="encode the footprint EXTRUSION instead of the real surface, giving the "
                         "aligned-pair partner: what the generator is handed at inference")
    args = ap.parse_args()

    import h5py
    if args.from_cache:
        measure_from_cache(*args.from_cache, n=args.pairs)
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = DoraCodec(load_dora(dev), n_coarse=args.n_coarse, n_sharp=args.n_sharp)

    surf = load_surfaces()
    rows = sorted(surf)
    if args.stratify:
        rows = _stratified_rows(surf, rows, args.stratify)
        print(f"[precompute] stratified sample: {args.stratify} rows, round-robin over sources")
    else:
        if args.start:
            rows = rows[args.start:]
        if args.limit:
            rows = rows[:args.limit]
    held = set(int(i) for i in test_indices(35776))
    print(f"[precompute] {len(rows)} buildings -> {args.out}")

    # Verification samples span the whole requested output, not just the tail encoded after a
    # resume. Otherwise a crash before the original guard ran would leave the already-committed
    # prefix permanently unchecked.
    step = max(1, len(rows) // max(1, args.verify_pos))
    vrows = set(rows[::step][:args.verify_pos]) if args.verify_pos > 0 else set()
    cache = IncrementalCache(args.out, args.resume)
    written_rows = set(cache.done)
    if cache.done:
        rows = [r for r in rows if r not in cache.done]
        print(f"[precompute] {len(rows)} rows left to encode")
    src_id = {"bag3d": 0, "nrw": 1, "plateau": 2}
    vblds: dict = {}
    t0 = time.time()
    attrs = {"codec": codec.name, "n_coarse": args.n_coarse, "n_sharp": args.n_sharp}
    try:
        with h5py.File(H5, "r") as f:
            for n, r in enumerate(rows):
                try:
                    bld, src = _building_for_row(f, surf, r, args.blockout)
                    z, pos = encode_row(codec, bld, r)
                except Exception as e:
                    print(f"  [skip] row {r}: {type(e).__name__}"); continue
                if r in vrows:
                    vblds[r] = bld
                cache.add(
                    latent=z[0].cpu().numpy().astype(np.float16),  # fp16: 2048x64 per building
                    # positions live in [-1,1], where fp16 resolves ~1e-3 -- 30x finer than the 2/63
                    # voxel pitch, so the storage is exact enough to match on and halves 875 MB -> 437 MB
                    query_pos=pos.astype(np.float16),
                    footprint=np.asarray(f["footprint"][r], np.uint8),
                    height_m=float(f["height_m"][r]),
                    region=src_id[src],
                    row=r,
                    held_out=1 if r in held else 0,             # 1 = held out, never trained on
                )
                written_rows.add(r)
                if (n + 1) % 200 == 0:
                    el = time.time() - t0
                    eta = el / (n + 1) * (len(rows) - n - 1)
                    print(f"  {n+1}/{len(rows)}  {el:.0f}s  eta {eta:.0f}s", flush=True)
                if (n + 1) % 50 == 0 and torch.cuda.is_available():
                    # The blockout path's marching-cubes meshes vary far more in size than the tiny
                    # (median ~20-face) recovered corpus meshes, so the caching allocator's reserved
                    # pool keeps growing; on this unified-memory box that reserved-but-idle memory
                    # counts against system RAM, not a separate VRAM pool -- release it periodically.
                    torch.cuda.empty_cache()

            # On a resumed run the verification rows may all be in the committed prefix. Rebuild
            # only those few encoder inputs so the final position guard covers that prefix too.
            for r in sorted(vrows - set(vblds)):
                if r not in written_rows:
                    continue
                try:
                    vblds[r], _ = _building_for_row(f, surf, r, args.blockout)
                except Exception as e:
                    print(f"  [verify_pos skip] row {r}: {type(e).__name__}")
    finally:
        # KeyboardInterrupt, a driver error, or another unexpected failure still leaves every
        # successfully encoded row committed and the HDF5 handle closed for `--resume`.
        cache.close(attrs)

    # The guards read the finished file, so they check what a consumer will actually get -- including
    # anything a resumed segment wrote. They are the last word: a cache that fails them is not sound
    # merely because it is complete.
    with h5py.File(args.out, "r") as o:
        if "latent" not in o or not len(o["latent"]):
            raise SystemExit("nothing encoded")
        L = o["latent"]
        rows_out = [int(r) for r in o["row"]]
        held_n = int(np.sum(np.asarray(o["held_out"])))
        verify_frame(codec, L, o["footprint"], n=args.verify, tol=args.verify_tol)
        verify_positions(codec, vblds, rows_out, o["query_pos"])
        latent_shape = L.shape
        latent_mb = L.size * L.dtype.itemsize / 1e6
    print(f"[precompute] {latent_shape[0]} latents {latent_shape} ({latent_mb:.0f} MB fp16), "
          f"{held_n} held out -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
