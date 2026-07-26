from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FootprintEmbedNet(nn.Module):
    """Small footprint encoder for BuildingNet retrieval.

    Input:
        footprint: (B, 1, 64, 64), float in [0, 1]
        class_id:  (B,), long, optional

    Output:
        embedding: L2-normalized (B, embedding_dim)
        logits:    (B, num_classes)
    """

    def __init__(self, num_classes: int, embedding_dim: int = 256, class_dim: int = 32):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.class_emb = nn.Embedding(num_classes, class_dim)
        self.proj = nn.Sequential(
            nn.Linear(256 + class_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, footprint: torch.Tensor, class_id: torch.Tensor | None = None):
        feat = self.backbone(footprint)
        feat = self.pool(feat).flatten(1)
        if class_id is None:
            class_feat = torch.zeros(
                feat.shape[0],
                self.class_emb.embedding_dim,
                dtype=feat.dtype,
                device=feat.device,
            )
        else:
            class_feat = self.class_emb(class_id.to(feat.device))
        embedding = self.proj(torch.cat([feat, class_feat], dim=1))
        embedding = F.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return embedding, logits

