import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import directed_hausdorff
from torchvision.models import resnet50
import clip
import numpy as np

def compute_mask_metrics(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: binary numpy arrays (H, W), dtype=bool or uint8
    Returns: dict of IoU and Hausdorff
    """
    pred = pred_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)

    iou = jaccard_score(gt, pred)
    hausdorff = max(
        directed_hausdorff(gt_mask.astype(np.uint8), pred_mask.astype(np.uint8))[0],
        directed_hausdorff(pred_mask.astype(np.uint8), gt_mask.astype(np.uint8))[0]
    )

    return {"iou_score": iou, "hausdorff_distance": hausdorff}

    


def clip_image_similarity(img1_pil, img2_pil, device="cuda"):
    """
    Computes cosine similarity between two images using CLIP.
    img1_pil, img2_pil: PIL Images
    """
    model, preprocess = clip.load("ViT-B/32", device=device)

    img1 = preprocess(img1_pil).unsqueeze(0).to(device)
    img2 = preprocess(img2_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = model.encode_image(img1)
        emb2 = model.encode_image(img2)

    emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
    emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)

    similarity = (emb1 @ emb2.T).item()
    return similarity
