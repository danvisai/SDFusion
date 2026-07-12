"""Ticket 11: train the real full-data monolith baseline for the C2 kill-gate.

One `MonolithUNet` (`models/networks/monolith_unet.py`) trained from scratch as a
`GaussianDiffusion` (`models/monolith_diffusion.py`) on ticket 07's real (coarse, target)
`train_100` pairs (`datasets/monolith_pair_dataset.py`) -- coarse SDF channel-concat
conditioning, DDIM sampling for deterministic inference. A held-out-from-gradients slice of
`train_100` (never the sealed `data/splits_v1/test.json`) is tracked for convergence
monitoring only; the headline C2 comparison against the sealed test set is tickets 12/13.

Follows the PRD's experiment-run-artifact contract: every run is resumable (`--resume`,
checkpoints record step/model/optimizer state), a manifest records configuration, git
provenance, checkpoint identity (a content hash, so later tickets can verify which exact
checkpoint a result traces to), and the measured train/val loss history -- partial runs are
distinguishable from complete ones via `manifest["status"]`.

Out: <out>/ckpt/monolith_steps-{N}.pth, <out>/ckpt/monolith_steps-latest.pth,
     <out>/history.jsonl, <out>/manifest.json
Run:  env -u LD_PRELOAD -u LD_LIBRARY_PATH TORCH_HOME=external/torch_hub \
        ./sdfusion/bin/python scripts/foundations/train_monolith.py [--steps N] [--resume auto]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
for _p in ("datasets", "models", "models/networks"):
    sys.path.insert(0, str(REPO / _p))

from monolith_unet import MonolithUNet  # noqa: E402
from monolith_diffusion import GaussianDiffusion  # noqa: E402
from monolith_pair_dataset import MonolithPairDataset, train_val_ids  # noqa: E402


def save_checkpoint(path: Path, model, optimizer, step: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, config=config), tmp)
    tmp.replace(path)  # atomic on the same filesystem -- never leaves a half-written ckpt


def load_checkpoint(path: Path, model, optimizer, device: str) -> int:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def checkpoint_digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _git_provenance():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return dict(git_rev=None, dirty_digest=None)
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    digest = hashlib.sha1(status.encode()).hexdigest()[:12] if status.strip() else None
    return dict(git_rev=rev, dirty_digest=digest)


def evaluate(diffusion, loader, device) -> float:
    diffusion.model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            target, coarse = batch["target"].to(device), batch["coarse"].to(device)
            loss = diffusion.p_losses(target, coarse)
            total += float(loss) * target.shape[0]
            n += target.shape[0]
    diffusion.model.train()
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-json", default=str(REPO / "data/monolith_pairs_v1/pairs.json"))
    ap.add_argument("--out", default=str(REPO / "logs_building/monolith_v1"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--channel-mults", default="1,2,4")
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--surface-band", type=float, default=0.3,
                    help="|x0|<band gets extra loss weight (see GaussianDiffusion docstring)")
    ap.add_argument("--surface-weight", type=float, default=0.0,
                    help="0 = plain MSE; measured surface-band voxel fraction on train_100 is "
                         "~2.9%%, so e.g. 20 raises its share of the loss to ~38%% -- see ticket "
                         "11 answer for the pre-registered derivation")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-batches", type=int, default=8, help="cap val batches (speed)")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap dataset size")
    ap.add_argument("--resume", default=None, help="'auto' = <out>/ckpt/monolith_steps-latest.pth")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    out = Path(a.out)
    ckpt_dir = out / "ckpt"
    history_path = out / "history.jsonl"

    ids = json.load(open(a.pairs_json))
    if a.limit:
        ids = ids[: a.limit]
    train_ids, val_ids = train_val_ids(ids, val_frac=a.val_frac, seed=a.seed)

    train_ds = MonolithPairDataset(train_ids, augment=True, device="cpu")
    val_ds = MonolithPairDataset(val_ids, augment=False, device="cpu")
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers, drop_last=True, persistent_workers=a.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=0)

    channel_mults = tuple(int(x) for x in a.channel_mults.split(","))
    net = MonolithUNet(base_channels=a.base_channels, channel_mults=channel_mults).to(a.device)
    n_params = sum(p.numel() for p in net.parameters())
    diffusion = GaussianDiffusion(net, timesteps=a.timesteps, device=a.device,
                                  surface_band=a.surface_band, surface_weight=a.surface_weight)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr)

    config = dict(base_channels=a.base_channels, channel_mults=list(channel_mults),
                 timesteps=a.timesteps, lr=a.lr, batch_size=a.batch_size,
                 surface_band=a.surface_band, surface_weight=a.surface_weight,
                 val_frac=a.val_frac, seed=a.seed, n_train=len(train_ids), n_val=len(val_ids),
                 n_params=n_params)

    start_step = 0
    resume_path = None
    if a.resume == "auto":
        latest = ckpt_dir / "monolith_steps-latest.pth"
        resume_path = latest if latest.exists() else None
    elif a.resume:
        resume_path = Path(a.resume)
    if resume_path is not None:
        start_step = load_checkpoint(resume_path, net, optimizer, a.device)
        print(f"[resume] {resume_path} -> step {start_step}")

    out.mkdir(parents=True, exist_ok=True)
    history_f = open(history_path, "a")

    def log(rec: dict):
        history_f.write(json.dumps(rec) + "\n")
        history_f.flush()

    print(f"[*] n_params={n_params:,}  n_train={len(train_ids)}  n_val={len(val_ids)}  "
          f"device={a.device}  start_step={start_step}/{a.steps}")

    net.train()
    train_iter = iter(train_loader)
    t0 = time.time()
    step = start_step
    while step < a.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        target, coarse = batch["target"].to(a.device), batch["coarse"].to(a.device)
        loss = diffusion.p_losses(target, coarse)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1

        if step % a.log_every == 0 or step == a.steps:
            elapsed = time.time() - t0
            rate = (step - start_step) / max(elapsed, 1e-6)
            print(f"  step {step}/{a.steps}  loss={float(loss):.4f}  "
                  f"{rate:.3f} it/s", flush=True)
            log(dict(step=step, kind="train", loss=float(loss), it_per_sec=rate))

        if step % a.val_every == 0 or step == a.steps:
            capped = []
            for i, b in enumerate(val_loader):
                if i >= a.val_batches:
                    break
                capped.append(b)
            val_loss = evaluate(diffusion, capped, a.device) if capped else None
            if val_loss is not None:
                print(f"  [val] step {step}  loss={val_loss:.4f}", flush=True)
                log(dict(step=step, kind="val", loss=val_loss))

        if step % a.ckpt_every == 0 or step == a.steps:
            numbered = ckpt_dir / f"monolith_steps-{step}.pth"
            latest = ckpt_dir / "monolith_steps-latest.pth"
            save_checkpoint(numbered, net, optimizer, step, config)
            save_checkpoint(latest, net, optimizer, step, config)
            log(dict(step=step, kind="ckpt", path=str(numbered), digest=checkpoint_digest(numbered)))

    history_f.close()
    final_ckpt = ckpt_dir / "monolith_steps-latest.pth"
    manifest = dict(
        status="complete" if step >= a.steps else "partial",
        step=step, steps_requested=a.steps, config=config,
        pairs_json=a.pairs_json, val_ids=val_ids,
        final_checkpoint=str(final_ckpt),
        final_checkpoint_digest=checkpoint_digest(final_ckpt) if final_ckpt.exists() else None,
        history_path=str(history_path),
        command=" ".join(sys.argv),
        **_git_provenance(),
    )
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"[done] step={step}  final_ckpt={final_ckpt}  -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
