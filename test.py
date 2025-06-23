import torch
import clip
model, preprocess = clip.load("ViT-B/32")
print("CLIP loaded successfully!")