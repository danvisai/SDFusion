from models.networks.clip_networks.network import CLIPImageEncoder
import torch

# instantiate and move the encoder to GPU
encoder = CLIPImageEncoder(model="ViT-B/32").cuda()

# create a dummy input on GPU
dummy = torch.zeros(1, 3, 224, 224).cuda()

# forward pass
emb = encoder(dummy)

print(emb.shape)  # e.g. torch.Size([1, 512])
