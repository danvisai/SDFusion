"""Place a per-asset GaussianSet at an OSM polygon's world coordinates.

Mirrors the transform recipe from scene/run_demo.py:213-239 `place_mesh`:
    - polygon centroid (XZ) translates the asset's XZ center
    - uniform XZ scale  s_xz = polygon_max_extent / mesh_max_extent_xz
    - independent Y scale s_y = target_height / mesh_extent_y
    - base lifts to ground_y

For Gaussians, the world transform is applied to:
    - means : non-uniform diagonal scale + translation
    - scales: elementwise multiplied by (s_xz, s_y, s_xz) on the world axes
    - quats : unchanged (approximation — assumes Gaussian principal axes are
              roughly aligned to world axes; cheap and standard in 3DGS placement)
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

from scene.gsplat_common import GaussianSet


def _xz_extent_from_means(means: torch.Tensor,
                          raw_opac: torch.Tensor = None,
                          opac_threshold: float = 0.05,
                          q_low: float = 0.005, q_high: float = 0.995,
                          ) -> Tuple[float, float, float]:
    """Robust per-axis extent of a GaussianSet.

    Uses (q_low, q_high) percentiles instead of min/max so sparse outlier
    Gaussians produced by densification don't blow up the bbox. If raw_opac is
    provided, also drops Gaussians whose activated opacity is below threshold
    (they won't contribute to the rendered surface anyway).
    """
    mns = means
    if raw_opac is not None and opac_threshold > 0:
        keep = torch.sigmoid(raw_opac) > opac_threshold
        if keep.sum() > max(64, int(0.05 * means.shape[0])):
            mns = means[keep]
    lo = torch.quantile(mns, q_low, dim=0)
    hi = torch.quantile(mns, q_high, dim=0)
    ext = (hi - lo).cpu().tolist()
    return float(ext[0]), float(ext[1]), float(ext[2])


def place_gsplat(
    g: GaussianSet,
    polygon_xz: np.ndarray,
    target_height_m: float,
    ground_y: float = 0.0,
    aspect_preserve: bool = False,
) -> GaussianSet:
    """Frame-N GaussianSet -> Frame-W placed GaussianSet.

    Args:
        g               : GaussianSet in Frame N (centered, max-extent ~1).
        polygon_xz      : (P, 2) polygon vertices in world XZ meters.
        target_height_m : building height in meters.
        ground_y        : world Y to lift the asset's base onto.
        aspect_preserve : if True, use a single uniform scale s = min(s_xz, s_y)
                          so the building keeps its native aspect ratio. The
                          asset will under-fill either the polygon footprint or
                          the target height (whichever is the looser constraint),
                          but will not be flattened/stretched. Default False
                          (matches the original behaviour shared with place_mesh).

    Returns:
        A new GaussianSet (does not mutate input) in Frame W.
    """
    g = g.clone()
    poly = np.asarray(polygon_xz, dtype=np.float64)
    px_min, pz_min = poly.min(axis=0)
    px_max, pz_max = poly.max(axis=0)
    pw, pd = px_max - px_min, pz_max - pz_min
    pcx, pcz = (px_min + px_max) / 2.0, (pz_min + pz_max) / 2.0

    ext_x, ext_y, ext_z = _xz_extent_from_means(g.means, g.raw_opac)
    s_xz = float(max(pw, pd) / max(max(ext_x, ext_z), 1e-6))
    s_y = float(target_height_m / max(ext_y, 1e-6))

    device = g.means.device
    if aspect_preserve:
        s = min(s_xz, s_y)
        scale_vec = torch.tensor([s, s, s], device=device, dtype=g.means.dtype)
    else:
        scale_vec = torch.tensor([s_xz, s_y, s_xz], device=device, dtype=g.means.dtype)

    # 1) scale means about origin
    g.means = g.means * scale_vec

    # 2) scale raw_scales and raw_quats:
    if aspect_preserve:
        # Uniform scaling: simple and exact to scale principal axes directly
        log_scale = torch.log(scale_vec.clamp_min(1e-12))
        g.raw_scales = g.raw_scales + log_scale
    else:
        # Non-uniform scaling: requires transforming the covariance matrix and re-decomposing it.
        # This preserves the true ellipsoid geometry under affine shear/stretching.
        from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

        # Reconstruct covariance Sigma = R * S^2 * R^T
        q_norm = g.raw_quats / (g.raw_quats.norm(dim=-1, keepdim=True) + 1e-8)
        R = quaternion_to_matrix(q_norm) # (N, 3, 3)
        s = torch.exp(g.raw_scales) # (N, 3)
        S_sq = torch.diag_embed(s**2) # (N, 3, 3)
        cov = R @ S_sq @ R.transpose(-1, -2) # (N, 3, 3)

        # Apply scale_vec to covariance: Sigma_prime = M * Sigma * M
        cov_prime = scale_vec.unsqueeze(-1) * cov * scale_vec.unsqueeze(-2)

        # Eigendecomposition of Sigma_prime
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_prime)

        # Compute new scale parameters
        s_prime = torch.sqrt(eigenvalues.clamp(min=1e-12))
        g.raw_scales = torch.log(s_prime)

        # Ensure proper rotation matrix (det = +1)
        det = torch.linalg.det(eigenvectors)
        flip = torch.ones_like(det)
        flip[det < 0] = -1.0
        eigenvectors = eigenvectors.clone()
        eigenvectors[..., 0] = eigenvectors[..., 0] * flip.unsqueeze(-1)

        # Convert back to quaternions
        g.raw_quats = matrix_to_quaternion(eigenvectors)

    # 3) recompute new bbox in world frame (robust to outliers, same convention as init)
    keep_w = (torch.sigmoid(g.raw_opac) > 0.05) if g.raw_opac is not None else None
    mns_w = g.means[keep_w] if (keep_w is not None and keep_w.sum() > 64) else g.means
    lo = torch.quantile(mns_w, 0.005, dim=0).cpu().numpy()
    hi = torch.quantile(mns_w, 0.995, dim=0).cpu().numpy()
    cx = (lo[0] + hi[0]) / 2.0
    cz = (lo[2] + hi[2]) / 2.0
    dy = ground_y - float(lo[1])

    # 4) translate centroid to polygon centroid; base to ground_y
    t = torch.tensor([pcx - cx, dy, pcz - cz], device=device, dtype=g.means.dtype)
    g.means = g.means + t

    return g
