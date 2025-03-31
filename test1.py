import lpips 
import torch 


img0 = torch.randn(4,3,512,512)
img1 = torch.randn(4,3,512,512)


img0 = (img0-img0.min()) / (img0.max()-img0.min())
img1 = (img1-img1.min()) / (img1.max()-img1.min())

import pyiqa

# list all available metrics
# print(pyiqa.list_models())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# create metric with default setting
metric_psnr = pyiqa.create_metric('psnr', device=device)
metric_ssim = pyiqa.create_metric('ssimc', device=device)
metric_lpips = pyiqa.create_metric('lpips', device=device)
# metric_fid = pyiqa.create_metric('fid', device=device)
# metric_musiq = pyiqa.create_metric('musiq', device=device)
# metric_maniqa = pyiqa.create_metric('maniqa', device=device)

psnr = metric_psnr(img0, img1)
ssim = metric_ssim(img0, img1)
lpip = metric_lpips(img0, img1)
# fid = metric_fid(img0, img1)

# musiq = metric_musiq(img0, img1)
# maniqa = metric_maniqa(img0, img1)

# psnr metric 
from diffbir.utils.common import calculate_psnr_pt 
psnr2 = calculate_psnr_pt(img0, img1, crop_border=0)

# # ssim metric 
# from diffbir.utils.common import calculate_ssim_pt
# ssim2 = calculate_ssim_pt(img0, img1, crop_border=0)

# # lpips metric 
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
lpip2 = loss_fn_alex(img0, img1, normalize=True)


from IQA_pytorch import SSIM, GMSD, LPIPSvgg, DISTS
D = SSIM(channels=3)
# Calculate score of the image X with the reference Y
# X: (N,3,H,W) 
# Y: (N,3,H,W) 
# Tensor, data range: 0~1
ssim3  = D(img0, img1, as_loss=False) 

# from basicsr.metrics import calculate_psnr_pt, calculate_ssim

# psnr3=basicsr.metrics.calculate_psnr_pt(img0, img1, crop_border=0)
# ssim3=basicsr.metrics.calculate_ssim_pt(img0, img1, crop_border=0)


breakpoint()