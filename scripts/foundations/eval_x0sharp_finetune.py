"""#60 eval — x0-sharp finetune vs map-#24 baseline, paired on the same held-out footprints.
Reports roughness (the crispness metric) + #27 gate (footprint-IoU / LCC / collapse) for each, and a
shaded GT | map-#24 | finetuned montage. Loads the two 15GB models SEQUENTIALLY (one freed before the
next) so it fits."""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/scratch/gilbreth/dsimhadr/GenerativeTowns/SDFusion")
sys.path.insert(0, str(REPO))
from scripts.foundations.baseline_gate_eval import build_opt, mesh_sdf_surface, fp_iou, lcc_frac, score_gate
from scripts.foundations.refiner_prototype import surface_roughness

MAP24 = "logs_building/2026-07-16-stage3a-lod2-fromscratch-region/ckpt/stage3a_steps-latest.pth"
X0SHARP = "logs_building/x0sharp-w05-clip/ckpt/stage3a_steps-latest.pth"
OUT = REPO / "outputs/x0sharp_eval_w05"
N = 24
NM = 6
TAN = (0.82, 0.75, 0.60)


def rough(v): return surface_roughness(torch.from_numpy(np.asarray(v, np.float32)))


def run_model(ckpt, items, device):
    from models.stage3a_model import Stage3aModel
    opt = build_opt(device, ckpt=ckpt, use_region=True, use_extra_cond=False, use_ema=True)
    opt.ddim_steps = 100
    model = Stage3aModel(); model.initialize(opt)
    rows, sdfs = [], []
    with torch.no_grad():
        for it in items:
            data = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
                    for k, v in it.items() if torch.is_tensor(v)}
            sdf = model.inference(data, ddim_steps=100, max_sample=1).detach().cpu().numpy()[0, 0]
            occ = sdf <= 0
            real_fp = it["fp"].numpy()[0]
            rows.append(dict(region=int(it["region_id"]), gen_occ=float(occ.mean()),
                             collapsed=bool(occ.mean() < 1e-4), lcc=lcc_frac(occ),
                             fp_iou=fp_iou(occ, real_fp),
                             real_fp_self_iou=fp_iou((it["sdf"].numpy()[0] <= 0), real_fp),
                             roughness=rough(sdf)))
            sdfs.append(sdf)
    del model; gc.collect(); torch.cuda.empty_cache()
    return rows, sdfs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from datasets.bag3d_dataset import Bag3dDataset
    opt = build_opt(device, ckpt=MAP24, use_region=True, use_extra_cond=False, use_ema=True)
    ds = Bag3dDataset(); ds.initialize(opt, phase="test")
    pick = np.random.default_rng(0).choice(len(ds), size=min(N, len(ds)), replace=False)
    items = [ds[int(i)] for i in pick]
    gts = [it["sdf"].numpy()[0] for it in items]

    print(f"[eval] map-#24 baseline (n={len(items)})", flush=True)
    r24, s24 = run_model(MAP24, items, device)
    print(f"[eval] x0-sharp finetuned (n={len(items)})", flush=True)
    rx, sx = run_model(X0SHARP, items, device)

    def summ(rows):
        return dict(roughness_mean=float(np.mean([r["roughness"] for r in rows])),
                    gate=score_gate(rows))
    out = dict(gt_roughness_mean=float(np.mean([rough(g) for g in gts])),
               map24=summ(r24), x0sharp=summ(rx),
               map24_rough_per=[r["roughness"] for r in r24],
               x0sharp_rough_per=[r["roughness"] for r in rx])
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print("\n=== ROUGHNESS (GT floor {:.5f}) ===".format(out["gt_roughness_mean"]))
    print(f"  map-#24 : {out['map24']['roughness_mean']:.5f}  fp_iou_med={out['map24']['gate']['fp_iou_median']:.3f} "
          f"p10={out['map24']['gate']['fp_iou_p10']:.3f} LCC>=.9={out['map24']['gate']['lcc_ge90_frac']:.2f} "
          f"collapse={out['map24']['gate']['collapse_rate']:.2f} PASS={out['map24']['gate']['OVERALL_SCALAR_PASS']}")
    print(f"  x0-sharp: {out['x0sharp']['roughness_mean']:.5f}  fp_iou_med={out['x0sharp']['gate']['fp_iou_median']:.3f} "
          f"p10={out['x0sharp']['gate']['fp_iou_p10']:.3f} LCC>=.9={out['x0sharp']['gate']['lcc_ge90_frac']:.2f} "
          f"collapse={out['x0sharp']['gate']['collapse_rate']:.2f} PASS={out['x0sharp']['gate']['OVERALL_SCALAR_PASS']}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=50)
    cols = [("GT (real LoD2)", gts), ("map-#24", s24), ("x0-sharp finetuned", sx)]
    fig = plt.figure(figsize=(10, 3.2 * NM))
    for ri in range(NM):
        for ci, (title, vols) in enumerate(cols):
            ax = fig.add_subplot(NM, 3, ri * 3 + ci + 1, projection="3d"); ax.set_axis_off()
            v, f = mesh_sdf_surface(vols[ri])
            if v is not None:
                ax.plot_trisurf(v[:, 2], v[:, 0], f, v[:, 1], color=TAN, shade=True, lightsource=ls,
                                edgecolor="none", linewidth=0, antialiased=False)
                ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 64)
            ax.view_init(elev=24, azim=-58)
            if ri == 0: ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "montage.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    wf = REPO / "docs/wayfinding/diffusion-latent-accuracy/x0sharp-w05-vs-map24-montage.png"
    import shutil; shutil.copyfile(OUT / "montage.png", wf)
    print(f"SAVED: {OUT}/montage.png (+ {wf}) , {OUT}/metrics.json")


if __name__ == "__main__":
    main()
