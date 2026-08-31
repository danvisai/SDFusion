"""Seam: put a blockout latent's tokens in the same order as its real partner's (#90).

The Dora encoder picks each token by farthest-point sampling over a random surface draw, so token *i*
of one latent and token *i* of another describe unrelated places. The pair training target subtracts
them element-wise, which is why the bridge it learns is a difference between unrelated locations
(#87). Alignment is the repair: a **permutation of the same tokens**, so the aligned cache is derived
from the unaligned one with no second encode.

**The decision (#90): greedy matching, k=256.** Measured on 102 region-stratified buildings:

| method | token cosine | note |
|---|---|---|
| as encoded | 0.0405 | the training target today -- at the random floor |
| morton | 0.2112 | ❌ rejected: weakest, least stable |
| hungarian | 0.5106 | minimises total distance, and loses |
| **greedy** | **0.5387** | ✅ adopted |
| nn (`argmin`) | 0.7079 | ⚠️ many-to-one, NOT a permutation -- a bound, not a method |

🔑🔑 **Hungarian wins total position error and LOSES latent agreement.** Total distance was never the
objective, only the proxy: minimising the *sum* degrades a token that has an excellent partner in
order to rescue one that has none. Greedy genuinely matches **36.7%** of tokens against Hungarian's
**21.2%** -- 73% more real correspondences. Optimising the proxy harder made the real objective worse.

⚠️ **~63% of tokens have no partner** (beyond a voxel; mean 0.2783 against 0.0186 for the matched
ones). The blockout and the real building are genuinely different surfaces, which is why the ceiling
is 0.71 and not 1.0. #92 must expect the smaller number: #89's 0.73 came from `cdist().argmin`, which
is many-to-one and cannot reorder anything.

⚠️ **The k restriction changes the algorithm, not just its speed.** Unrestricted greedy scored 0.5596
and restricted-to-16 scored 0.4881, and reporting those as two findings of one method is how this
nearly got chosen wrong twice. The sweep settles it -- 16: 0.4783 · 64: 0.5141 · 256: 0.5329 ·
2048: 0.5392 -- and the default sits at the knee, where the last doubling buys 0.006.
"""
from __future__ import annotations

import numpy as np

#: the 64^3 grid pitch. A pair further apart than this is not a correspondence, it is a coincidence.
VOXEL = 2.0 / 63
METHODS = ("as_encoded", "morton", "greedy", "hungarian", "nn")


def align(pa: np.ndarray, pb: np.ndarray, method: str = "greedy", k: int = 256) -> np.ndarray:
    """-> `perm`, so that `zb[perm]` is `zb` reordered to stand beside `za` token for token.

    `pa`/`pb` are the two token query-position sets, (T, 3) each, from `DoraCodec.encode_with_positions`.
    Every method except `nn` returns a genuine **permutation** -- `np.sort(perm) == arange(T)` -- because
    a cache written from a many-to-one map would duplicate some tokens and drop others.
    """
    pa = np.asarray(pa, np.float64)
    pb = np.asarray(pb, np.float64)
    if pa.shape != pb.shape:
        raise ValueError(f"token sets differ in shape: {pa.shape} vs {pb.shape}")
    if method == "as_encoded":
        return np.arange(len(pb))
    if method == "nn":
        return _nn(pa, pb)
    if method == "morton":
        return _morton(pa, pb)
    if method == "greedy":
        return _greedy(pa, pb, k)
    if method == "hungarian":
        return _hungarian(pa, pb)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def _nn(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Each token's nearest partner, independently. NOT a permutation -- an upper bound only."""
    from scipy.spatial import cKDTree
    return cKDTree(pb).query(pa)[1]


def _morton(pa: np.ndarray, pb: np.ndarray, bits: int = 10) -> np.ndarray:
    """Sort both sets into a canonical space-filling order and pair them off by rank.

    Needs no partner at inference, which was its one advantage -- and #95 removed it by measuring a
    canonical order at inference at **-0.0150** against as-encoded, slightly worse and inside the noise.
    """
    ra, rb = np.argsort(_morton_code(pa, bits)), np.argsort(_morton_code(pb, bits))
    perm = np.empty(len(pb), np.int64)
    perm[ra] = rb
    return perm


def _morton_code(p: np.ndarray, bits: int) -> np.ndarray:
    q = np.clip(((p + 1.0) * 0.5 * ((1 << bits) - 1)).astype(np.int64), 0, (1 << bits) - 1)
    code = np.zeros(len(p), np.int64)
    for b in range(bits):
        for axis in range(3):
            code |= ((q[:, axis] >> b) & 1) << (3 * b + axis)
    return code


def _greedy(pa: np.ndarray, pb: np.ndarray, k: int) -> np.ndarray:
    """Take the closest available pair, repeatedly, over each token's k nearest candidates.

    Unlike Hungarian this optimises nothing globally, which is exactly why it wins: it never spends a
    good correspondence to improve a hopeless one.
    """
    from scipy.spatial import cKDTree

    t = len(pa)
    k = int(min(max(1, k), t))
    d, idx = cKDTree(pb).query(pa, k=k)
    d, idx = np.atleast_2d(d), np.atleast_2d(idx)
    if k == 1:
        d, idx = d.reshape(t, 1), idx.reshape(t, 1)

    order = np.argsort(d, axis=None, kind="stable")
    rows, cols = np.divmod(order, k)
    cand_b = idx[rows, cols]

    perm = np.full(t, -1, np.int64)
    used_a = np.zeros(t, bool)
    used_b = np.zeros(t, bool)
    for i, j in zip(rows.tolist(), cand_b.tolist()):
        if not used_a[i] and not used_b[j]:
            perm[i] = j
            used_a[i] = used_b[j] = True

    # Tokens whose candidates were all taken get the leftovers. They are not correspondences and are
    # counted as unmatched by `report`; they exist so the result stays a permutation.
    left_a = np.nonzero(~used_a)[0]
    left_b = np.nonzero(~used_b)[0]
    perm[left_a] = left_b
    return perm


def _hungarian(pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """The assignment that minimises TOTAL position distance -- the proxy, not the objective."""
    from scipy.optimize import linear_sum_assignment
    cost = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)
    return linear_sum_assignment(cost)[1]


def report(za, zb, pa: np.ndarray, pb: np.ndarray, perm: np.ndarray) -> dict:
    """Score a permutation: token cosine, how many pairs are real, and how far apart the rest are.

    Reductions run in **torch**. That began as a workaround for wrong numpy results (numpy 2.2.6 on
    Python 3.14, since fixed -- see `utils/numeric_guard.py`) and is kept because it is the
    independent second implementation the guards compare against: this function's output is the
    number the method choice is made on, so it is worth computing it somewhere other than the code
    being checked.
    """
    import torch

    ta = torch.nn.functional.normalize(torch.as_tensor(np.asarray(za, np.float32)), dim=-1)
    tb = torch.nn.functional.normalize(torch.as_tensor(np.asarray(zb, np.float32)), dim=-1)
    p = torch.as_tensor(np.asarray(perm, np.int64))
    cos = float((ta * tb[p]).sum(-1).mean())

    dist = np.linalg.norm(pa - np.asarray(pb)[perm], axis=-1)
    matched = dist <= VOXEL
    return {
        "cosine": cos,
        "matched_frac": float(matched.mean()),
        "dist_matched": float(dist[matched].mean()) if matched.any() else float("nan"),
        "dist_unmatched": float(dist[~matched].mean()) if (~matched).any() else float("nan"),
        "is_permutation": bool(len(np.unique(perm)) == len(perm)),
    }
