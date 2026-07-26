"""Neutral-shader FID measurement for the transform+composition proof (ticket 05).

Detail fidelity is DISTRIBUTIONAL (ADR 0002): compare a set of neutral-shader facade renders against
real facade renders via Fréchet Inception Distance. This module is the FID math + a pinned Inception
feature extractor with recorded provenance; rendering is in `render_facades.py`. Both share the locked
96^3 / neutral-normal-shader operating point (ADR 0004).

Feature-extractor provenance (pinned): torchvision `Inception_V3_Weights.DEFAULT` (IMAGENET1K_V1),
2048-d pre-logits avgpool features, images resized 299x299 bilinear + ImageNet normalization.

Run the tests: env -u LD_PRELOAD -u LD_LIBRARY_PATH ./sdfusion/bin/python scripts/eval/test_fid.py
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg

EXTRACTOR_PROVENANCE = dict(
    name="torchvision.inception_v3",
    weights="Inception_V3_Weights.DEFAULT (IMAGENET1K_V1)",
    feature="pre-logits avgpool 2048-d",
    preprocess="resize 299 bilinear + ImageNet mean/std",
    feature_dim=2048,
)


def _mu_cov(feats):
    feats = np.asarray(feats, dtype=np.float64)
    return feats.mean(axis=0), np.atleast_2d(np.cov(feats, rowvar=False))


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """FID = ||mu1-mu2||^2 + tr(cov1 + cov2 - 2*sqrt(cov1 cov2)), numerically stabilized."""
    mu1, mu2 = np.asarray(mu1, np.float64), np.asarray(mu2, np.float64)
    cov1, cov2 = np.atleast_2d(np.asarray(cov1, np.float64)), np.atleast_2d(np.asarray(cov2, np.float64))
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if not np.isfinite(covmean).all():                       # singular product -> jitter
        offset = np.eye(cov1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((cov1 + offset) @ (cov2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * np.trace(covmean))


def undersampled(feats_a, feats_b):
    """True if either set has fewer rows than the feature dimensionality — the covariance estimate
    is then rank-deficient (rank <= N-1 in a D-dim space) and FID is known to be substantially
    biased (Heusel et al. 2017 recommend >=10,000 samples for 2048-d Inception features; our own
    sanity run at N=144-288 empirically reproduced this: the point estimate fell OUTSIDE its own
    bootstrap CI, confirmed via synthetic same-distribution data where the true FID is 0 but the
    naive estimate was in the thousands, and did NOT shrink proportionally up to N=720). Ticket 05
    finding — see the ticket answer before trusting a headline FID computed on too few samples."""
    d = np.asarray(feats_a).shape[-1]
    return len(feats_a) < d or len(feats_b) < d


def fid_from_features(feats_a, feats_b, warn_undersampled=True):
    """FID between two feature sets of shape (N_a, D) and (N_b, D). Warns (not an error, so callers
    can still inspect the number) when either set is smaller than the feature dimensionality — see
    `undersampled`."""
    if warn_undersampled and undersampled(feats_a, feats_b):
        warnings.warn(
            f"FID computed with N_a={len(feats_a)}, N_b={len(feats_b)} samples in a "
            f"{np.asarray(feats_a).shape[-1]}-d feature space (N < D): the estimate is known to be "
            f"substantially biased at this scale (see ticket 05 / fid.undersampled docstring). "
            f"Use more samples before trusting a headline comparison.", stacklevel=2)
    mu1, cov1 = _mu_cov(feats_a)
    mu2, cov2 = _mu_cov(feats_b)
    return frechet_distance(mu1, cov1, mu2, cov2)


def _resample_indices(rng, n, groups=None):
    """Bootstrap resample indices: row-level by default, or GROUP-level if `groups` (a per-row
    group-id array, e.g. building id per rendered view) is given — resamples unique groups with
    replacement and returns every row belonging to a resampled group, so correlated rows (multiple
    camera views of the same building) are resampled as a unit instead of as independent samples."""
    if groups is None:
        return rng.integers(0, n, n)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    picked = uniq[rng.integers(0, len(uniq), len(uniq))]
    return np.concatenate([np.where(groups == g)[0] for g in picked])


def bootstrap_fid_ci(feats_a, feats_b, n_boot=100, seed=0, alpha=0.05, groups_a=None, groups_b=None):
    """(point, lo, hi): full-sample FID + a percentile bootstrap CI resampling BOTH sets with
    replacement. Reports finite-sample variation so small FID gaps are not over-claimed (PRD #19).

    `groups_a`/`groups_b`: optional per-row group ids (e.g. building id per view). When given,
    resampling happens at the GROUP level so multiple correlated views of one building aren't
    double-counted as independent samples (which would understate the CI)."""
    fa, fb = np.asarray(feats_a, np.float64), np.asarray(feats_b, np.float64)
    # warn ONCE for the whole call (fid_from_features would otherwise repeat it n_boot+1 times);
    # bootstrap resamples are even more undersampled than the full sets (grouped resampling drops
    # ~37% of unique groups per draw), so this check on the full sets is the meaningful gate.
    point = fid_from_features(fa, fb, warn_undersampled=True)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ia = _resample_indices(rng, len(fa), groups_a)
        ib = _resample_indices(rng, len(fb), groups_b)
        boots.append(fid_from_features(fa[ia], fb[ib], warn_undersampled=False))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


class InceptionExtractor:
    """Lazy, pinned Inception-v3 2048-d feature extractor. Weights load on first use
    (set TORCH_HOME=external/torch_hub). Deterministic: eval mode, no grad, dropout off."""

    def __init__(self, device="cuda", batch=32):
        self.device = device
        self.batch = batch
        self._model = None
        self._tf = None
        self.weights_url = None   # resolved concrete checkpoint URL, set on first _load()

    def _load(self):
        import torch
        import torchvision
        import torchvision.transforms as T
        from torchvision.models import Inception_V3_Weights
        weights = Inception_V3_Weights.DEFAULT
        # pin provenance to the ACTUAL checkpoint (its filename embeds a content hash, e.g.
        # inception_v3_google-0cc3c7bd.pth) rather than the floating DEFAULT alias, which could
        # silently repoint if torchvision changes its default in a future release.
        self.weights_url = weights.url
        m = torchvision.models.inception_v3(weights=weights)
        m.fc = torch.nn.Identity()               # expose the 2048-d pre-logits avgpool feature
        self._model = m.eval().to(self.device)
        self._tf = T.Compose([
            T.Resize((299, 299), antialias=True),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def features(self, images_uint8):
        """images_uint8: (N, H, W, 3) uint8 -> (N, 2048) float32 features."""
        import torch
        if self._model is None:
            self._load()
        x = torch.from_numpy(np.asarray(images_uint8)).float().permute(0, 3, 1, 2) / 255.0
        x = self._tf(x).to(self.device)
        out = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch):
                out.append(self._model(x[i:i + self.batch]).cpu().numpy())
        return np.concatenate(out, 0).astype(np.float32)
