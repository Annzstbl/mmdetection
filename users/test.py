from projects.DiffusionDet.diffusiondet.head import cosine_beta_schedule
import torch
from torch.nn import functional as F

# timestemp = 1000
# betas = cosine_beta_schedule(timestemp)
# alphas = 1. - betas #\alpha_t
# alphas_cumprod = torch.cumprod(alphas, dim=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
# print(alphas_cumprod_prev)



x_feat = torch.randn(8, 256, 768)
mask = torch.randn(8, 256)
# change mask to true and false
mask = mask > 0.5

indices = torch.nonzero(mask == True)
output = [x_feat[i,mask[i,:],:] for i in range(x_feat.shape[0])]
output : list #[8, x, 768]

x_feat_extract = x_feat(indices)
