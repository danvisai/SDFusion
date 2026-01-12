# # import torch
# # #import clip
# # # model, preprocess = clip.load("ViT-B/32")
# # # print("CLIP loaded successfully!")
# # print(torch.cuda.is_available())
# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("Device count:", torch.cuda.device_count())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import torch
import os
#import pytorch3d
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
#print("pytorch3d:", pytorch3d.__version__)
print("cuda.is_available():", torch.cuda.is_available())
print("Looking for libc10_cuda.so…", 
      os.path.exists(os.path.join(torch._C.__file__, os.pardir, "lib", "libc10_cuda.so")))

