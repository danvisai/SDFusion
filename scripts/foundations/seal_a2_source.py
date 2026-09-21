"""#188 step 4: freeze the accepted A2 checkpoint, hash it, and record what #119 will seal.

[#188](https://github.com/danvisai/SDFusion/issues/188): "Freeze and hash the accepted checkpoint,
and record the generation operating point (strength / steps / guidance) that #119 will seal
alongside it."

Three properties this enforces, none of them ceremony:

  * **Accepted means the pre-registered bar says so.** `build_seal` refuses a checkpoint whose
    verdict is not PASS. The escape hatch exists -- #183's arm 3 ran past a KILL on an explicit
    owner override and the ladder record says so in as many words -- but it must carry a recorded
    reason, because an unattributed override is indistinguishable from not having run the bar.
  * **The digest is of the file the consumer will load**, not of the training checkpoint it came
    from. Freezing strips optimizer state, which is ~70% of a vecset checkpoint and is needed only
    to *resume* training, so the two files have different hashes and only one of them is the source.
  * **A sealed source is immutable.** `freeze_checkpoint` refuses to overwrite, because every
    citation of a digest (#119's manifest, `prototype_voxel_editor.A2_CHECKPOINT_SHA256`, the
    wayfinding record) silently becomes wrong if the file behind it changes.

    ⚠️ Immutability is load-bearing here precisely because the freeze is **not** reproducible:
    `torch.save` is not byte-stable, so re-freezing the same training checkpoint yields a different
    file digest (pinned by a test). A reader therefore cannot re-derive a sealed digest -- the
    sealed file itself is the artifact, and overwriting it destroys the only copy that matches
    what was recorded.

What survives the strip is what a consumer needs: the weights, `latent_mu`/`latent_sd` (load-bearing
-- the denoiser trains on globally normalised latents and decodes to noise without them),
`latent_channels`, `footprint_res`, and `n_regions` (#188: the conditioning width has to travel with
the weights, or every consumer goes back to guessing the legacy 3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SEAL_PATH = REPO / "execution/artifacts/188_seal.json"

# Only needed to RESUME training. `stage_weights_for_transfer.py` drops the same pair, for the same
# reason; this keeps that decision in one shape rather than two.
DROP_KEYS = ("opt", "sched")

# The source this one replaces. #115's amendment: "Only the identity of the frozen checkpoint
# changes" -- so the seal names its predecessor rather than pretending to be the first.
REPLACES = {
    "path": "logs_building/vecset_v4_surf/vecset_denoiser_step240000.pth",
    "sha256": "643aed0896e2edc36ab3ecb073da63847881dc2ba95459eed896a19b65fed04d",
    "why": "conditions through nn.Embedding(3, 512); every BuildingWorld row is region id 3-8, so "
           "it raises IndexError. Structural, not a measured quality failure.",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_of(path) -> str:
    """SHA-256 of a file, read in bounded chunks (a vecset checkpoint is ~600 MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_checkpoint(src, dest) -> str:
    """Strip the optimizer state, write `dest`, and return the digest OF THE FILE WRITTEN.

    Refuses to overwrite: a sealed source is immutable, and every recorded digest that points at
    this path would quietly start describing different weights.
    """
    import torch

    src, dest = Path(src), Path(dest)
    if dest.exists():
        raise FileExistsError(
            f"{dest} already exists. A sealed source is immutable -- every citation of its digest "
            f"would silently start describing different weights. Seal to a new path instead.")
    blob = torch.load(src, map_location="cpu", weights_only=False)
    frozen = {k: v for k, v in blob.items() if k not in DROP_KEYS}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frozen, dest)
    return sha256_of(dest)


def build_seal(checkpoint, digest: str, step: int, region_free: bool, operating_point: dict,
               cohort: dict, verdict: dict, eval_artifact: str,
               override: bool = False, override_reason: str = "") -> dict:
    """The record #119 seals: which weights, at which operating point, accepted on what evidence."""
    if not _SHA256.match(str(digest)):
        raise ValueError(f"not a sha256 digest: {digest!r}")
    got = verdict.get("verdict")
    if got != "PASS":
        if not override:
            raise ValueError(
                f"the bar returned {got!r}, not PASS. #188 seals the ACCEPTED checkpoint; sealing "
                f"this one needs an explicit override with a recorded reason.")
        if not str(override_reason).strip():
            raise ValueError(
                "an override must carry a recorded reason (#183's arm 3 named its owner ruling); "
                "an unattributed override is indistinguishable from not having run the bar")

    return {
        "meta": {
            "ticket": 188,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_rev": _git_rev(),
            "for": "https://github.com/danvisai/SDFusion/issues/119",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": str(digest),
            "step": int(step),
            "region_free": bool(region_free),
        },
        # #115's method, carried over verbatim: one source per global row, row-keyed reseeding of
        # both the envelope encoding and the SDEdit noise, no sweeping, no retries, no best-of-k.
        "operating_point": dict(operating_point),
        "cohort": dict(cohort),
        "acceptance": {
            "verdict": got,
            "override": bool(override),
            "override_reason": str(override_reason),
            "clauses": verdict.get("clauses", []),
            "eval_artifact": str(eval_artifact),
            "registration": "execution/artifacts/188_preregistration.json",
        },
        "replaces": dict(REPLACES),
    }


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="#188: freeze, hash and seal the accepted A2 source")
    ap.add_argument("--checkpoint", required=True, help="the trained checkpoint to freeze")
    ap.add_argument("--dest", required=True, help="where the frozen source is written (immutable)")
    ap.add_argument("--verdict", required=True,
                    help="the bar's verdict JSON, from a2_buildingworld_bar.py score --out")
    ap.add_argument("--eval", required=True, help="the eval artifact the verdict was read from")
    ap.add_argument("--cohort", default=str(REPO / "execution/artifacts/188_cohort.json"))
    ap.add_argument("--out", default=str(SEAL_PATH))
    ap.add_argument("--override", action="store_true",
                    help="seal despite a non-PASS verdict; requires --override_reason")
    ap.add_argument("--override_reason", default="")
    args = ap.parse_args()

    import torch

    verdict = json.loads(Path(args.verdict).read_text())
    cohort = json.loads(Path(args.cohort).read_text())
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    digest = freeze_checkpoint(args.checkpoint, args.dest)
    seal = build_seal(
        checkpoint=args.dest, digest=digest, step=int(blob["step"]),
        region_free=not int(blob.get("n_regions", 3)),
        operating_point=json.loads(
            Path(REPO / "execution/artifacts/188_preregistration.json").read_text()
        )["operating_point"],
        cohort={"digest": cohort["train"]["digest"], "n": cohort["train"]["n"],
                "manifest": "execution/artifacts/188_cohort.json"},
        verdict=verdict, eval_artifact=args.eval,
        override=args.override, override_reason=args.override_reason)

    Path(args.out).write_text(json.dumps(seal, indent=1))
    print(f"[188] frozen  -> {args.dest}")
    print(f"[188] sha256     {digest}")
    print(f"[188] step       {seal['checkpoint']['step']}  "
          f"region_free={seal['checkpoint']['region_free']}")
    print(f"[188] operating  {seal['operating_point']}")
    print(f"[188] verdict    {seal['acceptance']['verdict']}"
          + ("  (OVERRIDDEN)" if seal["acceptance"]["override"] else ""))
    print(f"[188] seal    -> {args.out}")
    print("[188] hand this digest to #119; it is what that ticket seals as its source.")


if __name__ == "__main__":
    main()
