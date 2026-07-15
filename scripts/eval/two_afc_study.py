"""Ticket 17: prototype a minimal blinded two-alternative forced-choice (2AFC) study comparing
the monolith and decomposition arms' detail fidelity, on the same held-out population and neutral
render pipeline ticket 13 already established.

PROTOTYPE per this ticket's own type -- a small, concrete artifact meant to validate the sampling,
randomization, consent, and analysis contract WITH THE PROJECT OWNER before any real participant
recruitment. Determining "the feasible participant plan" is a separate, later decision this ticket
explicitly does not make (see the ticket's own Question). n is small by design.

Why this pairing (monolith vs decomposition, not vs real): PRD lines 39-40 frame the two-AFC study
as a human complement to detail FID. Ticket 13's FID result and its accompanying montage disagreed
in direction -- decomposition looked more building-like to the eye despite a worse (higher) FID.
A monolith-vs-decomposition head-to-head is the direct way to check whether that disagreement is
a real human-preference signal or an artifact of judging shape realism through neutral-render +
ImageNet-Inception features. Comparing each arm against the real building instead would answer a
different (also valid, not attempted here) question -- "which one fools a viewer" -- not this one.

Reuses, not duplicates: render_facades.py's orbit_cameras/render_sdf_neutral -- the SAME neutral
shader and camera set ticket 13's own FID comparison used, so the human and automated judgments
look at literally the same kind of image. Population is the same monolith_arm/decomposition_arm
intersection ticket 13 used.

Blinding: which side (left/right) shows which arm is randomized independently per pair via
`assign_blind_sides` and is NEVER written into any participant-facing file -- only into a separate
answer key the analysis script (analyze_two_afc.py) reads afterward. The consent/instructions text
shown alongside the study states plainly what the images are and that no personal data is
collected; it does not name the two systems being compared (that would break blinding).

Out: outputs/two_afc_study/{pairs/<building>_L.png, pairs/<building>_R.png, index.html},
     execution/artifacts/two_afc_answer_key.json (NOT participant-facing -- keep separate from
     anything a participant might see)
Run:  TORCH_HOME=external/torch_hub env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        ./sdfusion/bin/python scripts/eval/two_afc_study.py [--n-pairs 20] [--seed 0]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in ("scripts/eval", "scripts/foundations"):
    sys.path.insert(0, str(REPO / _p))

from transform_vs_noise import git_provenance  # noqa: E402

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def sample_pair_ids(ids, n, seed=0):
    """Deterministic, seed-reproducible sample of `n` ids (without replacement) from `ids`.
    Returns all of `ids` (sorted) if `n` meets or exceeds the population."""
    ids = sorted(ids)
    if n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(ids), size=n, replace=False).tolist())
    return [ids[i] for i in idx]


def assign_blind_sides(building_ids, seed=0):
    """For each building, independently randomize which side (left/right) shows which arm --
    the blinding step. Returns [{building, left, right}], left/right in
    {"monolith", "decomposition"}. Deterministic for a given seed (so a study run is
    reproducible), but a participant sees no pattern across buildings within one run."""
    rng = np.random.default_rng(seed)
    out = []
    for bid in building_ids:
        if rng.random() < 0.5:
            left, right = "monolith", "decomposition"
        else:
            left, right = "decomposition", "monolith"
        out.append(dict(building=bid, left=left, right=right))
    return out


def wilson_ci(k, n, z=Z95):
    """Wilson score interval for a binomial proportion k/n -- more accurate than the normal
    approximation at the small n a prototype study necessarily has."""
    if n == 0:
        return 0.0, 0.0, 1.0
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def two_afc_result(responses, answer_key):
    """`responses`: dict building_id -> "left"/"right" (which side the participant picked).
    `answer_key`: the list `assign_blind_sides` produced. Un-blinds each response, counts how
    often "decomposition" was picked, and reports a Wilson CI against the 50% no-preference null.
    Response ids absent from the answer key are reported, not silently dropped."""
    by_id = {a["building"]: a for a in answer_key}
    n, k, missing = 0, 0, []
    for bid, side in responses.items():
        if bid not in by_id:
            missing.append(bid)
            continue
        n += 1
        if by_id[bid][side] == "decomposition":
            k += 1
    if n == 0:
        return dict(n=0, n_preferred_decomposition=0, proportion=None, ci95=None,
                    missing_ids=missing, significant_vs_chance=False)
    phat, lo, hi = wilson_ci(k, n)
    return dict(n=n, n_preferred_decomposition=k, proportion=phat, ci95=[lo, hi],
                missing_ids=missing, significant_vs_chance=(lo > 0.5 or hi < 0.5))


def render_mesh_image(grid, out_path: Path, res=384, azim=-58, elev=18):
    """Marching-cubes + flat-beige matplotlib render -- the SAME visual convention every prior
    qualitative montage in this project used (decide_c2_kill_gate.py, generate_decomposition_arm.py,
    eval_visual.py), and what the project owner has actually been judging "building-likeness" from
    throughout this research thread. Deliberately NOT render_sdf_neutral's sphere-traced normal map:
    that rendering is right for FEATURE-EXTRACTION parity with ticket 13's FID pipeline, but a
    rainbow normal map is not what an untrained human would read as "looks like a real building" --
    using it here would confound the very question a 2AFC study is meant to test cleanly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    fig = plt.figure(figsize=(res / 100, res / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    if (grid <= 0).sum() > 8:
        v, f, *_ = measure.marching_cubes(grid, 0.0)
        ax.plot_trisurf(v[:, 2], v[:, 0], f, v[:, 1], color="#c9b790", edgecolor="none", shade=True)
        ax.set_xlim(0, grid.shape[2]); ax.set_ylim(0, grid.shape[0]); ax.set_zlim(0, grid.shape[1])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


CONSENT_TEXT = (
    "This is a short visual comparison. You will see pairs of 3D building renders, side by "
    "side. For each pair, click the one that looks more like a real building. There are no "
    "right or wrong answers, no personal data is collected, and your choices are saved only "
    "as which SIDE (left/right) you picked per pair -- not who you are. You can stop at any "
    "time; partial responses are still useful. This is a research prototype, not a finished "
    "study -- your feedback on the format itself is as valuable as your picks."
)


def _build_html(pairs_meta, out_dir: Path):
    """A self-contained static page: no server, no network calls. Images are embedded as
    base64 data URIs so the whole thing is one folder a participant can open directly or you
    can hand off. Responses are held in-page and exported as a JSON file via a download link
    (no backend to receive them)."""
    import base64

    cards = []
    for i, row in enumerate(pairs_meta):
        left_b64 = base64.b64encode((out_dir / "pairs" / f"{row['building']}_L.png").read_bytes()).decode()
        right_b64 = base64.b64encode((out_dir / "pairs" / f"{row['building']}_R.png").read_bytes()).decode()
        cards.append(f"""
<div class="pair" data-building="{row['building']}" id="pair-{i}">
  <div class="pair-label">Pair {i + 1} of {len(pairs_meta)}</div>
  <div class="imgs">
    <img src="data:image/png;base64,{left_b64}" data-side="left" onclick="pick(this)">
    <img src="data:image/png;base64,{right_b64}" data-side="right" onclick="pick(this)">
  </div>
</div>""")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Two-AFC detail-fidelity study (prototype)</title>
<style>
body {{ font-family: sans-serif; max-width: 900px; margin: 2em auto; }}
.consent {{ background: #f4f4f4; padding: 1em; border-radius: 6px; margin-bottom: 2em; }}
.pair {{ margin-bottom: 2em; }}
.pair-label {{ font-weight: bold; margin-bottom: 0.5em; }}
.imgs {{ display: flex; gap: 1em; }}
.imgs img {{ width: 45%; cursor: pointer; border: 4px solid transparent; }}
.imgs img.picked {{ border-color: #2ca02c; }}
#done {{ display: none; background: #eef; padding: 1em; border-radius: 6px; }}
button {{ padding: 0.5em 1em; font-size: 1em; }}
</style></head>
<body>
<h1>Detail-fidelity comparison (prototype)</h1>
<div class="consent">{CONSENT_TEXT}</div>
<div id="pairs">{''.join(cards)}</div>
<div id="done">
  <p>All pairs answered. Click below to download your responses as JSON, then send that file
  back for analysis.</p>
  <button onclick="download_responses()">Download responses.json</button>
</div>
<script>
const responses = {{}};
const total = {len(pairs_meta)};
function pick(img) {{
  const pairDiv = img.closest('.pair');
  pairDiv.querySelectorAll('img').forEach(i => i.classList.remove('picked'));
  img.classList.add('picked');
  responses[pairDiv.dataset.building] = img.dataset.side;
  if (Object.keys(responses).length === total) {{
    document.getElementById('done').style.display = 'block';
  }}
}}
function download_responses() {{
  const blob = new Blob([JSON.stringify(responses, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'responses.json';
  a.click();
}}
</script>
</body></html>
"""
    (out_dir / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=int, default=1, help="renders per building per arm (normal style only)")
    ap.add_argument("--img-res", type=int, default=384)
    ap.add_argument("--render-style", choices=["mesh", "normal"], default="mesh",
                    help="mesh: flat-beige marching-cubes render, matching every prior "
                         "qualitative montage a human would recognize as a building (default). "
                         "normal: sphere-traced normal map, matching ticket 13's FID pipeline "
                         "exactly -- useful for testing whether results depend on render style.")
    ap.add_argument("--out-dir", default=str(REPO / "outputs/two_afc_study"))
    ap.add_argument("--answer-key-out", default=str(REPO / "execution/artifacts/two_afc_answer_key.json"))
    a = ap.parse_args()

    import torch
    import render_facades as rf
    device = "cuda" if torch.cuda.is_available() else "cpu"

    monolith_manifest = json.load(open(REPO / "data/monolith_arm_v1/manifest.json"))
    decomp_manifest = json.load(open(REPO / "data/decomposition_arm_v1/manifest.json"))
    monolith_ids = {r["building"] for r in monolith_manifest["per_building"]}
    decomp_ids = {r["building"] for r in decomp_manifest["per_building"]}
    population = sorted(monolith_ids & decomp_ids)
    print(f"[*] {len(population)} buildings with output from both arms")

    ids = sample_pair_ids(population, a.n_pairs, seed=a.seed)
    answer_key = assign_blind_sides(ids, seed=a.seed)
    print(f"[*] sampled {len(ids)} pairs (seed={a.seed})")

    monolith_grids_dir = Path(monolith_manifest["grids_dir"])
    decomp_grids_dir = Path(decomp_manifest["grids_dir"])
    cams = rf.orbit_cameras(n_views=a.views)

    out_dir = Path(a.out_dir)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(answer_key):
        bid = row["building"]
        mono_grid = np.load(monolith_grids_dir / f"{bid}.npy").astype(np.float32)
        decomp_grid = np.load(decomp_grids_dir / f"{bid}.npy").astype(np.float32)
        grids = dict(monolith=mono_grid, decomposition=decomp_grid)
        for side in ("left", "right"):
            arm = row[side]
            out_path = pairs_dir / f"{bid}_{'L' if side == 'left' else 'R'}.png"
            if a.render_style == "mesh":
                render_mesh_image(grids[arm], out_path, res=a.img_res)
            else:
                img = rf.render_sdf_neutral(grids[arm], cameras=cams, res=a.img_res, device=device)[0]
                from PIL import Image
                Image.fromarray(img).save(out_path)
        print(f"  [{i + 1}/{len(answer_key)}] {bid} rendered (left={row['left']}, right={row['right']})", flush=True)

    _build_html(answer_key, out_dir)

    manifest = dict(
        n_pairs=len(answer_key), seed=a.seed, population_size=len(population),
        render_style=a.render_style, consent_text=CONSENT_TEXT, answer_key=answer_key,
        out_dir=str(out_dir), **git_provenance(),
    )
    Path(a.answer_key_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(a.answer_key_out, "w"), indent=2)
    print(f"\n[done] {len(answer_key)} pairs -> {out_dir / 'index.html'}")
    print(f"[save] answer key (NOT participant-facing) -> {a.answer_key_out}")


if __name__ == "__main__":
    main()
