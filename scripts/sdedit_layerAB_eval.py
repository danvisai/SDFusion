"""LOCALIZED (inpainting) before/after: BASELINE (no context) vs LAYER A (geometric context) vs
LAYER A+B (+ element-type token). The ablation eval from docs/CONTEXT_CONDITIONED_SNAP_BUILD_SPEC_
2026-06-30.md sec.5/7: does context conditioning kill blob/suppression (A), and does the added
element-type token additionally fix wrong-vocabulary (A+B)? Per sec.7 acceptance: a pronounced
tower-like mass, a flat slab (balcony probe), and a thin wall patch (window probe), on NL/DE/JP.

Body stays bit-exact via the locality-mask blend (refine.py:_edit_locality_mask), exactly the
production snap_volume(local=True) path. For A/A+B, REAL context channels (known_body, edit_mask,
primitive) are built from the ACTUAL edit (not the null/self-supervised ones) and passed via
sdedit(ctx_channels=...); A+B additionally gets the primitive's classified element_type via
sdedit(elem_type_id=...) (models/stage3a_model.py:_classify_element_type, the same voxel-space
SHAPE->ARCH rule the training used).

  env -u LD_PRELOAD -u LD_LIBRARY_PATH PYTHONPATH=. ./sdfusion/bin/python scripts/sdedit_layerAB_eval.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
import h5py, numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "server"))
from datasets.bag3d_dataset import Bag3dDataset
from models.stage3a_model import Stage3aModel
from sdedit_bag3d_test import mesh, TRUNC
from refine import _edit_locality_mask
from scene.sdf_edit import EditOp, _primitive

REAL = REPO / "data/real_massing_v1/real.h5"
BASELINE = REPO / "logs_building/continue-stage3a-xcultural-warmstart-ft-final/ckpt/stage3a_steps-latest.pth"
LAYER_A = REPO / "logs_building/continue-stage3a-layerA-context/ckpt/stage3a_steps-latest.pth"
LAYER_AB = REPO / "logs_building/continue-stage3a-layerAB-context-elemtype/ckpt/stage3a_steps-latest.pth"
MODELS = [("BASELINE (no context)", BASELINE, False, False),
          ("LAYER A (geom context)", LAYER_A, True, False),
          ("LAYER A+B (+elem-type)", LAYER_AB, True, True)]

STRENGTHS = [0.3, 0.5, 0.7]
INNER, BAND = 0.2, 0.5            # cube units: same locality mask as prior localized eval scripts
DS_FACTOR = 4                     # 64 -> 16 latent res, matches Stage3aModel._build_context

# sec.7 acceptance probes: tower (pronounced protrusion), slab (balcony probe), patch (window probe).
# EditOp size = half-extents, center/size in cube coords [-1,1] (scene/sdf_edit.py convention).
EDITS = [
    ("tower",  {"kind": "box", "center": [0.5, 0.35, 0.5],  "size": [0.12, 0.7, 0.12], "mode": "add"}),
    ("slab",   {"kind": "box", "center": [0.5, 0.15, 0.75], "size": [0.35, 0.06, 0.15], "mode": "add"}),
    ("patch",  {"kind": "box", "center": [0.5, 0.15, 0.1],  "size": [0.15, 0.15, 0.04], "mode": "add"}),
]

# (label, tag, source) — NL via Bag3dDataset (matches sdedit_localized_ab.py sample 5),
# DE/JP via direct real.h5 row (matches sdedit_localized_dejp.py rows).
BUILDINGS = [("NL", "nl", ("dataset", 5)),
             ("DE (NRW)", "de", ("row", 12500)),
             ("JP (PLATEAU)", "jp", ("row", 25000))]


def cube_pts(dev):
    g = torch.linspace(-1, 1, 64, device=dev)
    Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")           # (D,H,W)
    return torch.stack([X, Y, Z], -1).reshape(-1, 3)


def load_building(spec, dev):
    kind, key = spec
    if kind == "dataset":
        ds = Bag3dDataset(); ds.initialize(SimpleNamespace(bag3d_h5=str(REAL), trunc_thres=TRUNC,
                                                            augment=False), "train")
        it = ds[key]
        sdf = it["sdf"].view(1, 1, 64, 64, 64).to(dev)
        fp = it["fp"].view(1, 1, 64, 64).to(dev)
        return sdf, fp, {"class_id": it["class_id"].view(1).to(dev),
                         "style_id": it["style_id"].view(1).to(dev),
                         "height": it["height"].view(1).to(dev)}
    with h5py.File(REAL, "r") as f:
        sdf_np = f["sdf"][key].astype(np.float32)
        fp_np = f["footprint"][key].astype(np.uint8)
    sdf = torch.from_numpy(sdf_np).view(1, 1, 64, 64, 64).to(dev)
    fp = torch.from_numpy(fp_np).float().view(1, 1, 64, 64).to(dev)
    occ_y = (sdf_np <= 0).any(axis=(0, 2))
    ys = np.where(occ_y)[0]
    h_n = float((ys.max() - ys.min() + 1) * (2.0 / 63.0)) if ys.size else 0.0
    meta = {"class_id": torch.zeros(1, dtype=torch.long, device=dev),
            "style_id": torch.full((1,), 8, dtype=torch.long, device=dev),
            "height": torch.tensor([h_n], dtype=torch.float32, device=dev)}
    return sdf, fp, meta


def load_model(ckpt, dev, use_context, use_element_type):
    opt = SimpleNamespace(isTrain=False, device=dev, df_cfg=str(REPO / "configs/stage3a_sdf_diffusion.yaml"),
                          vq_cfg=str(REPO / "configs/vqvae_bnet.yaml"),
                          vq_ckpt=str(REPO / "logs_building/2025-05-19T19-58-28-vqvae-building-all-res64-LR1e-4-T0.2-release/ckpt/vqvae_steps-latest.pth"),
                          ckpt=str(ckpt), ddim_steps=50, debug="0", gpu_ids=[0] if dev == "cuda" else [],
                          ckpt_dir="/tmp", latent_size_HW=(16, 16), latent_size_D=16,
                          use_context=use_context, use_element_type=use_element_type)
    print(f"[load] {Path(ckpt).parent.parent.name} (use_context={use_context}, use_element_type={use_element_type})")
    m = Stage3aModel(); m.initialize(opt)
    return m


def build_edit_context(sdf0, box_sdf, w, dev):
    """REAL Layer-A/B context from the actual edit (mirrors Stage3aModel._build_context, but from
    a KNOWN edit rather than a random self-supervised region). Returns (ctx_channels, elem_id)."""
    region = (w > 0.05).float()                                    # dilated edit region (D,H,W)
    known_full = torch.where(region > 0.5, torch.full_like(sdf0[0, 0], TRUNC), sdf0[0, 0])
    occ_box = (box_sdf <= 0)
    prim_full = torch.where(occ_box, torch.full_like(box_sdf, -TRUNC), torch.full_like(box_sdf, TRUNC))
    mask_full = region
    ds = lambda v: F.avg_pool3d(v.view(1, 1, 64, 64, 64), kernel_size=DS_FACTOR, stride=DS_FACTOR)
    ctx_channels = (ds(known_full), ds(mask_full), ds(prim_full))

    idx = torch.nonzero(occ_box, as_tuple=False)
    if idx.numel() == 0:
        elem_id = torch.zeros(1, dtype=torch.long, device=dev)
    else:
        lo = idx.min(0).values.tolist(); hi = (idx.max(0).values + 1).tolist()
        hz = (hi[0] - lo[0]) / 2.0; hy = (hi[1] - lo[1]) / 2.0; hw = (hi[2] - lo[2]) / 2.0
        y_center = (lo[1] + hi[1]) / 2.0
        cls = Stage3aModel._classify_element_type(hz, hy, hw, y_center, 64.0)
        elem_id = torch.full((1,), cls, dtype=torch.long, device=dev)
    return ctx_channels, elem_id


def panel(fig, nr, nc, r, c, title, sdf):
    ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
    ax.set_title(title, fontsize=8); ax.set_axis_off(); ax.view_init(elev=18, azim=-55)
    ax.set_box_aspect((1, 1, 1))
    if sdf is None: return
    mm = mesh(sdf if sdf.dim() == 5 else sdf[None, None])
    if mm is None: return
    v, fc = np.asarray(mm.vertices), np.asarray(mm.faces)
    ax.plot_trisurf(v[:, 0], v[:, 2], fc, v[:, 1], color="#c9b790", edgecolor="none", shade=True)
    lim = [v.min(), v.max()]; ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pts = cube_pts(dev)
    models = [(label, load_model(ckpt, dev, uc, uet)) for label, ckpt, uc, uet in MODELS]
    out_dir = REPO / "outputs/sdedit_xcultural/layerAB_eval"; out_dir.mkdir(parents=True, exist_ok=True)

    for blabel, btag, bspec in BUILDINGS:
        sdf0, fp, meta = load_building(bspec, dev)
        for etag, edit in EDITS:
            box = _primitive(EditOp.from_dict(edit))(pts).reshape(64, 64, 64)
            w = _edit_locality_mask([edit], pts, INNER, BAND).reshape(64, 64, 64).to(dev)
            edited = torch.minimum(sdf0[0, 0], box).clamp(-TRUNC, TRUNC)[None, None]
            ctx_channels, elem_id = build_edit_context(sdf0, box, w, dev)
            elem_name = Stage3aModel._ELEMENT_TYPES[int(elem_id.item())]
            print(f"\n=== {blabel} / {etag} -> classified '{elem_name}'  "
                  f"mask>0.5 frac={float((w > 0.5).float().mean()):.3f} ===")

            base = dict(meta); base["fp"] = fp
            edit_data = dict(base); edit_data["sdf"] = edited

            nc = 2 + len(STRENGTHS); nr = len(models)
            fig = plt.figure(figsize=(3.0 * nc, 3.2 * nr))
            for r, (label, m) in enumerate(models):
                use_ctx = MODELS[r][2]; use_et = MODELS[r][3]
                panel(fig, nr, nc, r, 0, f"{label}\n{blabel} body", sdf0)
                panel(fig, nr, nc, r, 1, f"edited (+{etag})", edited)
                for j, s in enumerate(STRENGTHS):
                    torch.manual_seed(0)
                    kwargs = dict(strength=s, ddim_steps=50, uc_scale=1.0)
                    if use_ctx:
                        kwargs["ctx_channels"] = ctx_channels
                    if use_et:
                        kwargs["elem_type_id"] = elem_id
                    out = m.sdedit(edit_data, **kwargs)
                    local = edited[0, 0] * (1 - w) + out[0, 0] * w
                    occ = float((local <= 0).float().mean())
                    print(f"  {label}  s={s:.1f}  localized occ={occ:.3f}")
                    panel(fig, nr, nc, r, 2 + j, f"s={s:.1f}", local[None, None])

            fig.suptitle(f"LOCALIZED snap: {blabel} + {etag} edit (classified '{elem_name}') — "
                          "body bit-exact\nrow0: baseline (no ctx)  row1: Layer A  row2: Layer A+B",
                          fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            out_p = out_dir / f"{btag}_{etag}.png"
            fig.savefig(out_p, dpi=88); plt.close(fig)
            print(f"[saved] {out_p}")


if __name__ == "__main__":
    main()
