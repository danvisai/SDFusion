"""Part-LAYOUT PLANNER — detail-plan step 3 (the OmniPart stage-1 analog).

Autoregressive transformer: (64^3 massing SDF, building class) -> a variable-length sequence
of part tokens (type, bbox center+extent), trained on the BuildingNet part-instance dataset
(outputs/part_layouts_full/part_instances.npz). This upgrades the count-only PartComposer to
"WHICH elements, WHERE" — the prerequisite for coherent add/replace editing (step 4).

Token = type_emb + box_proj(box6). Sequence = [COND] p_1 ... p_n [STOP], causal transformer,
per-position joint heads: next-type logits (incl. STOP) + next-box regression.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# contiguous type ids (raw BuildingNet label -> planner type) — keep in sync with the npz
RAW_TYPES = [2, 4, 6, 7, 12, 14, 16, 17, 22, 15]
TYPE_NAMES = ["window", "roof", "door", "tower", "column",
              "balcony", "balcony_upper", "stairs", "dome", "chimney"]
N_TYPES = len(RAW_TYPES)
STOP = N_TYPES                       # extra class in the type head
N_CLASSES = 4                        # COMMERCIAL/PUBLIC/RELIGIOUS/RESIDENTIAL


class MassingEncoder(nn.Module):
    """64^3 truncated SDF -> global feature."""

    def __init__(self, out_dim=256):
        super().__init__()
        c = [16, 32, 64, 128]
        layers, ci = [], 1
        for co in c:
            layers += [nn.Conv3d(ci, co, 3, stride=2, padding=1), nn.GroupNorm(8, co), nn.SiLU()]
            ci = co
        self.net = nn.Sequential(*layers)                 # 64 -> 4
        self.head = nn.Linear(c[-1], out_dim)

    def forward(self, sdf):                                # (B,1,64,64,64)
        h = self.net(sdf)
        return self.head(h.mean(dim=(2, 3, 4)))           # (B, out_dim)


class MassingEncoderSpatial(nn.Module):
    """64^3 SDF -> (global feature, 8^3 SPATIAL memory tokens w/ 3D position) — v2.

    v1's global pooling threw away WHERE the walls are, so sampled boxes collapsed to the
    center. Here the part decoder cross-attends to 512 located tokens instead."""

    def __init__(self, dim=256):
        super().__init__()
        c = [16, 32, 64]
        layers, ci = [], 1
        for co in c:
            layers += [nn.Conv3d(ci, co, 3, stride=2, padding=1), nn.GroupNorm(8, co), nn.SiLU()]
            ci = co
        self.net = nn.Sequential(*layers)                  # 64 -> 8, 64ch
        self.tok = nn.Linear(c[-1] + 3, dim)               # feature + its (x,y,z) in [-1,1]
        self.glob = nn.Linear(c[-1], dim)

    def forward(self, sdf):                                 # (B,1,64,64,64) axes (D=z,H=y,W=x)
        h = self.net(sdf)                                   # (B,C,8,8,8)
        B, C, D, H, W = h.shape
        g = torch.linspace(-1, 1, D, device=h.device)
        Z, Y, X = torch.meshgrid(g, g, g, indexing="ij")
        pos = torch.stack([X, Y, Z], -1).reshape(1, -1, 3).expand(B, -1, -1)
        feat = h.flatten(2).transpose(1, 2)                 # (B, 512, C)
        mem = self.tok(torch.cat([feat, pos], -1))          # (B, 512, dim)
        return self.glob(h.mean(dim=(2, 3, 4))), mem


class PartLayoutPlannerV2(nn.Module):
    """v2: causal part decoder CROSS-ATTENDS to the spatial massing grid (OmniPart-style),
    and boxes are DISCRETIZED + SAMPLED (PolyGen/MeshGPT-style) — L1 regression collapses
    multimodal positions (a window fits any facade) to their mean = the center (v1/v2-L1
    both hit identical val box loss ~0.182 = the mean-predictor optimum)."""

    BINS = 32          # per-coordinate bins over [-1, 1]

    def __init__(self, dim=256, depth=4, heads=4, max_len=40, stop_weight=2.0):
        super().__init__()
        self.max_len = max_len
        self.enc = MassingEncoderSpatial(dim)
        self.cls_emb = nn.Embedding(N_CLASSES, dim)
        self.type_emb = nn.Embedding(N_TYPES + 1, dim)
        self.box_proj = nn.Linear(6, dim)
        self.pos_emb = nn.Embedding(max_len + 1, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerDecoder(layer, depth)
        self.type_head = nn.Linear(dim, N_TYPES + 1)
        self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                      nn.Linear(dim, 6 * self.BINS))   # 6 coord classifications
        w = torch.ones(N_TYPES + 1); w[STOP] = stop_weight  # count calibration (v1 over-generated)
        self.register_buffer("type_w", w)

    def _to_bins(self, boxes):
        return ((boxes.clamp(-1, 1) + 1) * 0.5 * (self.BINS - 1)).round().long()

    def _from_bins(self, idx):
        return idx.float() / (self.BINS - 1) * 2.0 - 1.0

    def _hidden(self, sdf, cls_id, types, boxes):
        """tgt = [BOS] p_1..p_{n-1}; memory = spatial massing tokens."""
        B, L = types.shape
        glob, mem = self.enc(sdf)
        bos = (glob + self.cls_emb(cls_id)).unsqueeze(1)
        tok = self.type_emb(types) + self.box_proj(boxes)
        seq = torch.cat([bos, tok[:, :-1]], dim=1) if L > 0 else bos
        seq = seq + self.pos_emb(torch.arange(seq.shape[1], device=seq.device))
        mask = torch.triu(torch.full((seq.shape[1], seq.shape[1]), float("-inf"),
                                     device=seq.device), diagonal=1)
        return self.tr(seq, mem, tgt_mask=mask)

    def forward(self, sdf, cls_id, types, boxes, lens):
        h = self._hidden(sdf, cls_id, types, boxes)         # (B, L, D): predicts p_1..p_n/STOP
        B, L1, _ = h.shape
        tgt_type = torch.full((B, L1), -100, device=types.device, dtype=torch.long)
        for b in range(B):
            n = int(lens[b])
            tgt_type[b, :n] = types[b, :n]
            if n < L1:
                tgt_type[b, n] = STOP
        type_loss = F.cross_entropy(self.type_head(h).reshape(-1, N_TYPES + 1),
                                    tgt_type.reshape(-1), ignore_index=-100,
                                    weight=self.type_w)
        logits = self.box_head(h).view(B, L1, 6, self.BINS)
        tgt_bins = self._to_bins(boxes)                      # (B, L1, 6)
        box_mask = torch.arange(L1, device=types.device)[None] < lens[:, None]
        tgt_bins = torch.where(box_mask[..., None], tgt_bins,
                               torch.full_like(tgt_bins, -100))
        box_loss = F.cross_entropy(logits.reshape(-1, self.BINS), tgt_bins.reshape(-1),
                                   ignore_index=-100)
        return type_loss, box_loss

    @torch.no_grad()
    def sample(self, sdf, cls_id, temperature=0.7):
        B = sdf.shape[0]
        dev = sdf.device
        types = torch.zeros(B, 0, dtype=torch.long, device=dev)
        boxes = torch.zeros(B, 0, 6, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        out = [[] for _ in range(B)]
        for _ in range(self.max_len):
            h = self._hidden(sdf, cls_id, types, boxes)[:, -1]
            logits = self.type_head(h) / max(temperature, 1e-4)
            t = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
            bl = self.box_head(h).view(B, 6, self.BINS) / max(temperature, 1e-4)
            bins = torch.multinomial(F.softmax(bl, -1).reshape(B * 6, self.BINS), 1).view(B, 6)
            b6 = self._from_bins(bins)                       # SAMPLED positions (multimodal kept)
            for i in range(B):
                if not done[i]:
                    if t[i].item() == STOP:
                        done[i] = True
                    else:
                        out[i].append((int(t[i]), b6[i].cpu().numpy()))
            if done.all():
                break
            t = torch.where(done, torch.zeros_like(t), t)
            types = torch.cat([types, t[:, None]], 1)
            boxes = torch.cat([boxes, b6[:, None]], 1)
        return out


class PartLayoutPlanner(nn.Module):
    def __init__(self, dim=256, depth=4, heads=4, max_len=40):
        super().__init__()
        self.max_len = max_len
        self.enc = MassingEncoder(dim)
        self.cls_emb = nn.Embedding(N_CLASSES, dim)
        self.type_emb = nn.Embedding(N_TYPES + 1, dim)     # +1: BOS token reuses STOP slot
        self.box_proj = nn.Linear(6, dim)
        self.pos_emb = nn.Embedding(max_len + 2, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, depth)
        self.type_head = nn.Linear(dim, N_TYPES + 1)       # + STOP
        self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 6))

    def _tokens(self, sdf, cls_id, types, boxes):
        """[COND][BOS] p_1..p_{n-1} -> hidden states predicting p_1..p_n / STOP."""
        B, L = types.shape
        cond = (self.enc(sdf) + self.cls_emb(cls_id)).unsqueeze(1)          # (B,1,D)
        bos = self.type_emb(torch.full((B, 1), STOP, device=types.device, dtype=torch.long))
        tok = self.type_emb(types) + self.box_proj(boxes)                   # (B,L,D)
        seq = torch.cat([cond, bos, tok[:, :-1]], dim=1) if L > 0 else torch.cat([cond, bos], dim=1)
        seq = seq + self.pos_emb(torch.arange(seq.shape[1], device=seq.device))
        mask = torch.triu(torch.full((seq.shape[1], seq.shape[1]), float("-inf"),
                                     device=seq.device), diagonal=1)
        return self.tr(seq, mask=mask)

    def forward(self, sdf, cls_id, types, boxes, lens):
        """Teacher-forced losses. types (B,L) long, boxes (B,L,6), lens (B,)."""
        h = self._tokens(sdf, cls_id, types, boxes)[:, 1:]                  # drop COND position
        B, L1, _ = h.shape                                                   # L1 = L (BOS..p_{n-1})
        tgt_type = torch.full((B, L1), -100, device=types.device, dtype=torch.long)
        for b in range(B):
            n = int(lens[b])
            tgt_type[b, :n] = types[b, :n]
            if n < L1:
                tgt_type[b, n] = STOP
        type_loss = F.cross_entropy(h.reshape(-1, h.shape[-1]) @ self.type_head.weight.T
                                    + self.type_head.bias, tgt_type.reshape(-1),
                                    ignore_index=-100)
        box_pred = self.box_head(h)
        box_mask = (torch.arange(L1, device=types.device)[None] < lens[:, None]).float()
        box_loss = (F.l1_loss(box_pred, boxes, reduction="none").mean(-1) * box_mask).sum() \
            / box_mask.sum().clamp_min(1)
        return type_loss, box_loss

    @torch.no_grad()
    def sample(self, sdf, cls_id, temperature=0.8):
        B = sdf.shape[0]
        dev = sdf.device
        types = torch.zeros(B, 0, dtype=torch.long, device=dev)
        boxes = torch.zeros(B, 0, 6, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        out = [[] for _ in range(B)]
        for _ in range(self.max_len):
            h = self._tokens(sdf, cls_id, types, boxes)[:, -1]
            logits = self.type_head(h) / max(temperature, 1e-4)
            t = torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
            b6 = self.box_head(h)
            for i in range(B):
                if not done[i]:
                    if t[i].item() == STOP:
                        done[i] = True
                    else:
                        out[i].append((int(t[i]), b6[i].cpu().numpy()))
            if done.all():
                break
            t = torch.where(done, torch.zeros_like(t), t)
            types = torch.cat([types, t[:, None]], 1)
            boxes = torch.cat([boxes, b6[:, None]], 1)
        return out
