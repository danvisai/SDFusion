"""Stage A — core inference engine for the Option B+ generative head.

Wraps the trained B+.6 recipe-param diffusion (`outputs/recipe_param_diffusion_b6/`) +
the differentiable recipes into a small, stateless engine the FastAPI service
(`inference_service.py`) and any host plugin can call. No web framework here so it stays
unit-testable and embeddable.

Pipeline per building:
    (footprint_polygon_m, class, height_m, style, seed)
        -> scale-invariant conditioning  (recipe_param_space.raw_conditioning)
        -> sample recipe params           (diffusion ddim_sample -> inverse-normalize)
        -> DiffRecipe.forward on a world-meter grid -> marching cubes -> mesh -> glb

`params_to_mesh` is the FAST path (no model call) used for slider edits in the host.
`generate_building` is the GENERATIVE path (samples params first).

Frames: polygons + heights are in METERS (OSM/world frame), matching how the B+.4
synthetic data — and therefore the recipe params — were defined. The conditioning fed to
the diffusion is scale-invariant, so the same head serves any metric footprint.

Sampling defaults (guidance=2.0, eta=1.0) come from scripts/sweep_recipe_diffusion_sampling.py:
guidance acts as a diversity knob here, so a little is desirable for a varied city.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.networks import recipe_param_space as ps
from models.networks.diff_recipe import build_diff_recipe
from models.networks.recipe_param_diffusion import ConditionalDenoiser, GaussianDiffusion
from scene.sdf_primitives import polygon_bbox_with_pad, grid_to_mesh

DEFAULT_CKPT = REPO / "outputs/recipe_param_diffusion_b6"

# guidance presets exposed to the host as a "diversity" dial (from the B+.6 sweep).
DIVERSITY_PRESETS = {"low": 1.0, "medium": 2.0, "high": 3.0}


@dataclass
class BuildingResult:
    style: str
    recipe_params: np.ndarray     # raw, trimmed to the style's dimensionality
    glb: bytes                    # binary glTF mesh (local frame, base at y=0)
    n_vertices: int
    n_faces: int
    position_xz: tuple            # world centroid of the footprint (for host placement)


class RecipeInferenceEngine:
    def __init__(self, ckpt_dir: Path = DEFAULT_CKPT, device: Optional[str] = None,
                 grid_res: int = 64):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_res = grid_res
        ck = torch.load(Path(ckpt_dir) / "denoiser.pth", map_location=self.device)
        a = ck["args"]
        self.denoiser = ConditionalDenoiser(hidden=a["hidden"], depth=a["depth"]).to(self.device)
        self.denoiser.load_state_dict(ck["model"])
        self.denoiser.eval()
        self.diffusion = GaussianDiffusion(ck["timesteps"], device=self.device)
        self.feat, self.pnorm = ps.load_scalers(Path(ckpt_dir) / "scalers.npz")
        self._recipes = {s: build_diff_recipe(s)[0].to(self.device) for s in ps.STYLES}

    # -- conditioning ------------------------------------------------------
    def _conditioning(self, polygon_xz, height_m, class_name, style):
        ci = ps.CLASS_TO_IDX.get(class_name.upper(), ps.CLASS_TO_IDX["RESIDENTIAL"])
        si = ps.STYLE_TO_IDX[style]
        cond = ps.raw_conditioning(np.asarray(polygon_xz, np.float32), float(height_m), ci, si)
        return self.feat.transform(cond[None])

    # -- generative path: sample recipe params -----------------------------
    @torch.no_grad()
    def sample_params(self, polygon_xz, height_m, class_name, style, seed=None,
                      guidance=2.0, eta=1.0, steps=50) -> np.ndarray:
        if style not in ps.STYLE_TO_IDX:
            raise ValueError(f"unknown style '{style}'; known: {ps.STYLES}")
        if seed is not None:
            torch.manual_seed(int(seed))
        cond = torch.tensor(self._conditioning(polygon_xz, height_m, class_name, style),
                            device=self.device)
        x0 = self.diffusion.ddim_sample(self.denoiser, cond, steps=steps, eta=eta,
                                        guidance=guidance)
        si = np.array([ps.STYLE_TO_IDX[style]])
        raw = self.pnorm.inverse(x0.cpu().numpy(), si)[0]
        return ps.unpad_params(raw, style)

    # -- fast path: params -> mesh (no model) ------------------------------
    @torch.no_grad()
    def params_to_mesh(self, params, style, polygon_xz, height_m):
        """Evaluate the recipe on a world-meter grid and marching-cubes it. Returns
        (trimesh.Trimesh | None, bbox)."""
        params = np.asarray(params, dtype=np.float32)
        n_exp = ps.STYLE_DIMS[style]
        if params.shape[-1] != n_exp:
            raise ValueError(f"style '{style}' expects {n_exp} params, got {params.shape[-1]}")
        poly = np.asarray(polygon_xz, dtype=np.float32)
        height = float(height_m)
        # bbox with headroom for roof/ornaments (matches B+.4 synthetic sampling intent).
        bbox = polygon_bbox_with_pad(poly, height * 1.6, pad=0.12)
        x0, y0, z0, x1, y1, z1 = bbox
        R = self.grid_res
        dev = self.device
        xs = torch.linspace(x0, x1, R, device=dev)
        ys = torch.linspace(y0, y1, R, device=dev)
        zs = torch.linspace(z0, z1, R, device=dev)
        Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")
        pts = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        module = self._recipes[style]
        p = torch.tensor(np.asarray(params, np.float32), device=dev)
        poly_t = torch.tensor(poly, device=dev)
        h_t = torch.tensor(height, device=dev)
        sdf = module(p, poly_t, h_t, pts).reshape(R, R, R)  # (D, H, W)
        mesh = grid_to_mesh(sdf, bbox, iso=0.0)
        return mesh, bbox

    @staticmethod
    def mesh_to_glb(mesh) -> bytes:
        if mesh is None or len(mesh.faces) == 0:
            return b""
        try:
            mesh.fix_normals()           # consistent OUTWARD normals -> correct lighting
        except Exception:
            pass
        return mesh.export(file_type="glb")

    # -- end-to-end one building -------------------------------------------
    def generate_building(self, polygon_xz, class_name, height_m, style, seed=None,
                          guidance=2.0, eta=1.0, steps=50, detail=True) -> BuildingResult:
        poly = np.asarray(polygon_xz, dtype=np.float32)
        centroid = poly.mean(axis=0)
        local = poly - centroid  # build the mesh in a local frame; host places at centroid
        params = self.sample_params(local, height_m, class_name, style, seed,
                                    guidance, eta, steps)
        if detail:
            mesh = self._detailed_mesh(params, style, class_name, local, height_m, seed)
        else:
            mesh, _ = self.params_to_mesh(params, style, local, height_m)
        nv = 0 if mesh is None else len(mesh.vertices)
        nf = 0 if mesh is None else len(mesh.faces)
        return BuildingResult(style=style, recipe_params=params,
                              glb=self.mesh_to_glb(mesh), n_vertices=nv, n_faces=nf,
                              position_xz=(float(centroid[0]), float(centroid[1])))

    @torch.no_grad()
    def _detailed_mesh(self, params, style, class_name, local_poly, height, seed):
        """Compose recipe base + COMPOSER-DRIVEN detail (facade + door + roof + landmarks) -> mesh.

        The part-composer (trained on real BuildingNet part layouts) decides which elements
        this massing gets (glazing, roof type, dome/towers/steps); sdf_detail instantiates
        them. Falls back to per-class random sampling if the composer is unavailable."""
        from scene.sdf_edit import recipe_base_sdf
        from scene import sdf_detail as det
        from scene.sdf_primitives import sample_grid, grid_to_mesh
        rng = np.random.default_rng(0 if seed is None else int(seed))
        base = recipe_base_sdf(style, params, local_poly, height, device=self.device)
        n_towers = 0
        try:
            from scene.composer_detail import compose_detail, get_composer
            sdf, _layout, dec = compose_detail(base, local_poly, height, class_name, style=style,
                                               seed=seed, composer=get_composer(self.device))
            n_towers = dec["n_towers"]
        except Exception as exc:   # robust fallback to the original random path
            print(f"[recipe_inference] composer unavailable ({exc}); random detail fallback")
            dp = det.ground_glazing(det.vector_to_params(det.sample_detail_vector(style, rng)), class_name)
            sdf = det.add_facade_detail(base, local_poly, height, dp)
            sdf = det.add_door(sdf, local_poly, height)
            sdf = det.apply_roof_shape(sdf, local_poly, height, det.sample_roof_shape(class_name, None, rng))
            lm = det.sample_landmarks(class_name, rng); n_towers = lm["n_towers"]
            if lm["dome"] or lm["n_towers"] or lm["steps"]:
                sdf = det.add_landmarks(sdf, local_poly, height, dome=lm["dome"],
                                        n_towers=lm["n_towers"], steps=lm["steps"])
        p = np.asarray(local_poly)
        x0, z0, x1, z1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
        pad = 0.12 * max(x1 - x0, z1 - z0) + 1.0
        head = height * (1.9 if n_towers else 1.5)
        bbox = (x0 - pad, 0.0, z0 - pad, x1 + pad, head, z1 + pad)
        mesh = grid_to_mesh(sample_grid(sdf, self.grid_res, bbox, device=self.device), bbox, 0.0)
        try:
            from scene.mesh_cleanup import cleanup_mesh
            mesh = cleanup_mesh(mesh)                     # weld + drop floating fragments
        except Exception:
            pass
        return mesh
