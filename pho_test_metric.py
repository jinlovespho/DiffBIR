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
metric_dists = pyiqa.create_metric('dists', device=device)
# metric_fid = pyiqa.create_metric('fid', device=device)
metric_niqe = pyiqa.create_metric('niqe', device=device)
metric_musiq = pyiqa.create_metric('musiq', device=device)
metric_maniqa = pyiqa.create_metric('maniqa', device=device)
metric_clipiqa = pyiqa.create_metric('clipiqa', device=device)

psnr = metric_psnr(img0, img1)
ssim = metric_ssim(img0, img1)
lpip = metric_lpips(img0, img1)
dists = metric_dists(img0, img1)
# fid = metric_fid(img0, img1)
niqe = metric_niqe(img0, img1)
musiq = metric_musiq(img0, img1)
maniqa = metric_maniqa(img0, img1)
clipiqa = metric_clipiqa(img0, img1)

breakpoint()