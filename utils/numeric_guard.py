"""Refuse to run long numeric jobs on an interpreter/numpy pair that computes wrong answers.

**This exists because of a real, silent, three-hour bug.** On Python 3.14.4 with numpy 2.2.6 --
a combination numpy 2.2.x never supported, since Python 3.14 support landed in numpy 2.3 -- a
freshly allocated, owning numpy array could be modified by a *later, unrelated* expression. The
observable was a live array that was byte-identical to its source when assigned and different from it
a few statements later, with nothing touching it.

It was:

* **silent** -- no warning, no exception, no allocator complaint. `PYTHONMALLOC=debug`,
  `MALLOC_CHECK_=3` and `MALLOC_PERTURB_` all saw nothing, which is how it was finally distinguished
  from heap corruption: a use-after-free does not survive `MALLOC_PERTURB_` unchanged;
* **deterministic**, so it did not look like memory corruption;
* **layout-sensitive**, so every attempt to observe it from inside the process made it vanish -- and
  so a green-looking debug session proved nothing;
* **wrong in the direction that ends investigations.** It turned matched-token agreement of 0.6660
  into 0.0057. Believed, it would have said this map's whole premise was dead.

Torch was unaffected throughout, which is what made the disagreement visible at all.

⚠️ A version check is a proxy, not a proof. Keep the cross-implementation guards
(`_verify_agrees`, `verify_positions`, `align_cache.verify`) regardless: they check the *answer*,
which is the thing that actually matters, and they cost seconds against runs measured in hours.
"""
from __future__ import annotations

import sys

#: numpy gained Python 3.14 support in 2.3.0. Below that, on 3.14, results are silently unreliable.
MIN_NUMPY_ON_314 = (2, 3)


def check_numpy(strict: bool = True) -> str | None:
    """Return a complaint string, or None when the pair is sane. Raises when `strict` (the default).

    Called from the entry point of anything that runs long enough that a silent wrong answer would be
    expensive -- the precompute, the alignment pass, the probes, the checkpoint watcher.
    """
    import numpy as np

    if sys.version_info[:2] < (3, 14):
        return None
    ver = tuple(int(p) for p in np.__version__.split(".")[:2])
    if ver >= MIN_NUMPY_ON_314:
        return None

    msg = (f"numpy {np.__version__} on Python {sys.version.split()[0]} computes silently wrong "
           f"results: numpy gained 3.14 support in 2.3.0. A live array can be modified by a later, "
           f"unrelated expression -- it turned a real 0.6660 into 0.0057 here, with no error. "
           f"Upgrade numpy to >= {'.'.join(str(v) for v in MIN_NUMPY_ON_314)} before running this.")
    if strict:
        raise SystemExit(f"[numeric_guard] {msg}")
    return msg
