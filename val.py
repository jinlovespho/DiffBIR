from argparse import ArgumentParser
import copy
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.model import ControlLDM, Diffusion
from diffbir.sampler import SpacedSampler
import initialize
import os
import cv2 
import numpy as np
import wandb
import pyiqa
from torchvision.utils import save_image 
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from diffbir.dataset.pho_utils import encode, decode 
from accelerate.utils import DistributedDataParallelKwargs
from PIL import Image 



def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(25, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    cfg = OmegaConf.load(args.config)


    # load logging tools and ckpt directory
    if accelerator.is_main_process:
        exp_dir, ckpt_dir, exp_name, writer = initialize.load_experiment_settings(accelerator, cfg)


    # load data
    if cfg.dataset.val_dataset_name == 'div2k':
        gt_imgs = sorted(os.listdir(f'{cfg.dataset.dataset_path}/gt'))  # len(gt_imgs) = 3000
        lq_imgs = sorted(os.listdir(f'{cfg.dataset.dataset_path}/lq'))  # len(lq_imgs) = 3000
        
        gt_imgs_path = sorted([f'{cfg.dataset.dataset_path}/gt/{img}' for img in gt_imgs])
        lq_imgs_path = sorted([f'{cfg.dataset.dataset_path}/lq/{img}' for img in lq_imgs])
        len_val_ds = len(gt_imgs)
    
    elif cfg.dataset.val_dataset_name == 'realsr' or cfg.dataset.val_dataset_name == 'drealsr':
        gt_imgs = sorted(os.listdir(f'{cfg.dataset.dataset_path}/test_HR'))  # len(gt_imgs) = 3000
        lq_imgs = sorted(os.listdir(f'{cfg.dataset.dataset_path}/test_LR'))  # len(lq_imgs) = 3000
        
        gt_imgs_path = sorted([f'{cfg.dataset.dataset_path}/test_HR/{img}' for img in gt_imgs])
        lq_imgs_path = sorted([f'{cfg.dataset.dataset_path}/test_LR/{img}' for img in lq_imgs])
        len_val_ds = len(gt_imgs)
        
    elif cfg.dataset.val_dataset_name == 'sam':
        _, val_ds, _, val_loader = initialize.load_data(accelerator, cfg)
        val_batch_transform = instantiate_from_config(cfg.val_batch_transform)
        val_loader = accelerator.prepare(val_loader)
        len_val_ds = len(val_ds)

    

    # load models
    models, resume_ckpt_path = initialize.load_model(accelerator, device, args, cfg)
    

    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup model accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # SR metrics
    metric_psnr = pyiqa.create_metric('psnr', device=device)
    metric_ssim = pyiqa.create_metric('ssimc', device=device)
    metric_lpips = pyiqa.create_metric('lpips', device=device)
    metric_dists = pyiqa.create_metric('dists', device=device)
    # metric_fid = pyiqa.create_metric('fid', device=device)
    metric_niqe = pyiqa.create_metric('niqe', device=device)
    metric_musiq = pyiqa.create_metric('musiq', device=device)
    metric_maniqa = pyiqa.create_metric('maniqa', device=device)
    metric_clipiqa = pyiqa.create_metric('clipiqa', device=device)



    # print Validation Info
    if accelerator.is_main_process:
        print('='*100)
        print(f'Experiment name: {exp_name}')
        print('-'*50)
        print(f"Save val directory: {exp_dir}")
        print('-'*50)
        print(f"Num val_dataset: {len_val_ds}")
        print('-'*50)
        print(f'Loaded models: {list(models.keys())}')
        print('-'*50)
        print(f'Resume training ckpt: ', resume_ckpt_path)
        print('='*100)



    tot_val_psnr=[]
    tot_val_ssim=[]
    tot_val_lpips=[]
    tot_val_dists=[]
    tot_val_niqe=[]
    tot_val_musiq=[]
    tot_val_maniqa=[]
    tot_val_clipiqa=[]
    

    # set seed for identical generation for validation sampling noise
    gen.manual_seed(25)
    
    
    # put model on eval
    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()


    # Validation
    if  cfg.dataset.val_dataset_name == 'div2k' or \
        cfg.dataset.val_dataset_name == 'realsr' or \
        cfg.dataset.val_dataset_name == 'drealsr':
        
        # For val_gt (range [-1, 1])
        preprocess_gt = T.Compose([
            T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # For val_lq (range [0, 1])
        preprocess_lq = T.Compose([
            T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ])
        
        for val_batch_idx, (gt_img_path, lq_img_path) in enumerate(tqdm(zip(gt_imgs_path, lq_imgs_path), desc='val', total=len(gt_imgs_path))):
            
            gt_id = gt_img_path.split('/')[-1].split('.')[0]
            lq_id = lq_img_path.split('/')[-1].split('.')[0]
            assert gt_id == lq_id, f"gt_img_path: {gt_img_path}, lq_img_path: {lq_img_path} do not match"
            

            gt_img = Image.open(gt_img_path)     # size: 512
            lq_img = Image.open(lq_img_path)     # size: 128
            

            val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)  # 1 3 512 512
            val_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)  # 1 3 512 512
            

            val_bs, _, val_H, val_W = val_gt.shape
            val_prompt=[""]
            
            
            with torch.no_grad():
                # val_z_0 = pure_cldm.vae_encode(val_gt)
                val_clean = models['swinir'](val_lq)    # b 3 512 512
                val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

                M=1
                pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
                # print(pure_noise)
                
                # sampling
                val_z, val_sampled_unet_feats = sampler.sample(     # b 4 56 56
                    model=models['cldm'],
                    device=device,
                    steps=50,
                    x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),   # manual shape adjustment
                    cond=val_cond,
                    uncond=None,
                    cfg_scale=1.0,
                    x_T = pure_noise,
                    progress=accelerator.is_main_process,
                    cfg=cfg
                )

                # =========================== OCR ===========================
                # if cfg.exp_args.model_name == 'diffbir_testr':
                if False:

                    # process annotations for OCR val loss 
                    val_targets=[]
                    for i in range(val_bs):
                        num_box=len(val_boxes[i])
                        tmp_dict={}
                        tmp_dict['labels'] = torch.tensor([0]*num_box).cuda()  # 0 for text
                        tmp_dict['boxes'] = torch.tensor(val_boxes[i]).cuda()
                        tmp_dict['texts'] = val_text_encs[i]
                        tmp_dict['ctrl_points'] = val_polys[i]
                        val_targets.append(tmp_dict)


                    # evaluate diffusion features for different timesteps
                    for sampled_iter, sampled_timestep, unet_feats in val_sampled_unet_feats:

                        # OCR model forward pass
                        sampling_val_ocr_loss_dict, sampling_val_ocr_results = models['testr'](unet_feats, val_targets)
                        # val ocr total loss
                        sampling_val_ocr_tot_loss = sum(sampling_val_ocr_loss_dict.values())


                        # log sampling train loss and box to wandb
                        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                            for ocr_key, ocr_val in sampling_val_ocr_loss_dict.items():
                                wandb.log({f"sampling_val_LOSS_iter{sampled_iter}_timestep{sampled_timestep}/{ocr_key}": ocr_val.item()})
                            wandb.log({f"sampling_val_LOSS_iter{sampled_iter}_timestep{sampled_timestep}/ocr_tot_loss": sampling_val_ocr_tot_loss.item()})


                        # vis poly and text
                        for i in range(M):
                            vis_val_gt = val_gt[i]                                  # 3 512 512 [-1,1]
                            vis_val_gt = (vis_val_gt + 1)/2 * 255.0                 # 3 512 512 [0,255]
                            vis_val_gt = vis_val_gt.permute(1,2,0).detach().cpu().numpy().astype(np.uint8).copy()  # 512 512 3

                            results_per_img = sampling_val_ocr_results[i]

                            for j in range(len(results_per_img.polygons)):
                                val_ctrl_pnt= results_per_img.polygons[j].view(16,2).cpu().detach().numpy().astype(np.int32)    # 32 -> 16 2
                                val_score = results_per_img.scores[j]                     # 1
                                val_rec = results_per_img.recs[j]
                                val_pred_text = decode(val_rec)

                                cv2.polylines(vis_val_gt, [val_ctrl_pnt], True, (0,255,0), 2)
                                cv2.putText(vis_val_gt, val_pred_text, (val_ctrl_pnt[0][0], val_ctrl_pnt[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            # cv2.imwrite(f'./tmp{i}.jpg', vis_val_gt[...,::-1])
                            if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                                wandb.log({f'sampling_val_VIS_iter{sampled_iter}_timestep{sampled_timestep}/{val_batch_idx}_poly{i}': wandb.Image(vis_val_gt, caption=f'draw sampled val ocr results on gt')})
                                
                
                

                restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
                # restored_img = torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
                
                
                # save sampled images   
                img_save_path = f'{cfg.exp_args.save_val_img_dir}/{cfg.exp_args.log_additional_msg}'
                os.makedirs(img_save_path, exist_ok=True)
                restored_img_pil = TF.to_pil_image(restored_img.squeeze().cpu())
                restored_img_pil.save(f'{img_save_path}/{gt_id}.png')
                
                
                # log total psnr, ssim, lpips for val
                tot_val_psnr.append(torch.mean(metric_psnr(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_ssim.append(torch.mean(metric_ssim(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_lpips.append(torch.mean(metric_lpips(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_dists.append(torch.mean(metric_dists(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                # tot_val_fid.append(torch.mean(metric_fid(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_niqe.append(torch.mean(metric_niqe(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_musiq.append(torch.mean(metric_musiq(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_maniqa.append(torch.mean(metric_maniqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_clipiqa.append(torch.mean(metric_clipiqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                


                # log sampling val imgs to wandb
                if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':

                    # log sampling val metrics 
                    wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_dists': torch.mean(metric_dists(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            # f'sampling_val_METRIC/val_fid': torch.mean(metric_fid(
                            #                                                                 restored_img, 
                            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_niqe': torch.mean(metric_niqe(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_musiq': torch.mean(metric_musiq(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_maniqa': torch.mean(metric_maniqa(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            })
                    
                    # log sampling val images 
                    wandb.log({ f'sampling_val_FINAL_VIS/{val_batch_idx}_val_gt': wandb.Image((val_gt + 1) / 2, caption=f'gt_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_lq': wandb.Image(val_lq, caption=f'lq_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_cleaned': wandb.Image(val_clean, caption=f'cleaned_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_sampled': wandb.Image(torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), caption=f'sampled_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_prompt': wandb.Image(log_txt_as_img((256, 256), val_prompt), caption=f'prompt'),
                            })
                    wandb.log({f'sampling_val_FINAL_VIS/{val_batch_idx}_val_all': wandb.Image(torch.concat([val_lq, val_clean, torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), val_gt], dim=2), caption='lq_clean_sample,gt')})
        
        
    else:
        for val_batch_idx, val_batch in enumerate(val_loader):

            # load val data
            to(val_batch, device)
            val_batch = val_batch_transform(val_batch)
            val_gt, val_lq, val_prompt, val_texts, val_boxes, val_polys, val_text_encs, val_img_name = val_batch 
            val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float()   # 1 3 512 512 [-1,1]
            val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float()   # 1 3 512 512 [0,1]
            
            val_bs, _, val_H, val_W = val_gt.shape
            

            # prepare vae, condition
            with torch.no_grad():
                # val_z_0 = pure_cldm.vae_encode(val_gt)
                val_clean = models['swinir'](val_lq)    # b 3 512 512
                val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

                M=1
                pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
                # print(pure_noise)
                
                # sampling
                val_z, val_sampled_unet_feats = sampler.sample(     # b 4 56 56
                    model=models['cldm'],
                    device=device,
                    steps=50,
                    x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),   # manual shape adjustment
                    cond=val_cond,
                    uncond=None,
                    cfg_scale=1.0,
                    x_T = pure_noise,
                    progress=accelerator.is_main_process,
                    cfg=cfg
                )

                # =========================== OCR ===========================
                if cfg.exp_args.model_name == 'diffbir_testr':

                    # process annotations for OCR val loss 
                    val_targets=[]
                    for i in range(val_bs):
                        num_box=len(val_boxes[i])
                        tmp_dict={}
                        tmp_dict['labels'] = torch.tensor([0]*num_box).cuda()  # 0 for text
                        tmp_dict['boxes'] = torch.tensor(val_boxes[i]).cuda()
                        tmp_dict['texts'] = val_text_encs[i]
                        tmp_dict['ctrl_points'] = val_polys[i]
                        val_targets.append(tmp_dict)


                    # evaluate diffusion features for different timesteps
                    for sampled_iter, sampled_timestep, unet_feats in val_sampled_unet_feats:

                        # OCR model forward pass
                        sampling_val_ocr_loss_dict, sampling_val_ocr_results = models['testr'](unet_feats, val_targets)
                        # val ocr total loss
                        sampling_val_ocr_tot_loss = sum(sampling_val_ocr_loss_dict.values())


                        # log sampling train loss and box to wandb
                        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                            for ocr_key, ocr_val in sampling_val_ocr_loss_dict.items():
                                wandb.log({f"sampling_val_LOSS_iter{sampled_iter}_timestep{sampled_timestep}/{ocr_key}": ocr_val.item()})
                            wandb.log({f"sampling_val_LOSS_iter{sampled_iter}_timestep{sampled_timestep}/ocr_tot_loss": sampling_val_ocr_tot_loss.item()})


                        # vis poly and text
                        for i in range(M):
                            vis_val_gt = val_gt[i]                                  # 3 512 512 [-1,1]
                            vis_val_gt = (vis_val_gt + 1)/2 * 255.0                 # 3 512 512 [0,255]
                            vis_val_gt = vis_val_gt.permute(1,2,0).detach().cpu().numpy().astype(np.uint8).copy()  # 512 512 3

                            results_per_img = sampling_val_ocr_results[i]

                            for j in range(len(results_per_img.polygons)):
                                val_ctrl_pnt= results_per_img.polygons[j].view(16,2).cpu().detach().numpy().astype(np.int32)    # 32 -> 16 2
                                val_score = results_per_img.scores[j]                     # 1
                                val_rec = results_per_img.recs[j]
                                val_pred_text = decode(val_rec)

                                cv2.polylines(vis_val_gt, [val_ctrl_pnt], True, (0,255,0), 2)
                                cv2.putText(vis_val_gt, val_pred_text, (val_ctrl_pnt[0][0], val_ctrl_pnt[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                            # cv2.imwrite(f'./tmp{i}.jpg', vis_val_gt[...,::-1])
                            if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                                wandb.log({f'sampling_val_VIS_iter{sampled_iter}_timestep{sampled_timestep}/{val_batch_idx}_poly{i}': wandb.Image(vis_val_gt, caption=f'draw sampled val ocr results on gt')})
                                

                restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
                # restored_img = torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
                
                # log total psnr, ssim, lpips for val
                tot_val_psnr.append(torch.mean(metric_psnr(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_ssim.append(torch.mean(metric_ssim(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_lpips.append(torch.mean(metric_lpips(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_dists.append(torch.mean(metric_dists(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                # tot_val_fid.append(torch.mean(metric_fid(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_niqe.append(torch.mean(metric_niqe(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_musiq.append(torch.mean(metric_musiq(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_maniqa.append(torch.mean(metric_maniqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                tot_val_clipiqa.append(torch.mean(metric_clipiqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
                



                # log sampling val imgs to wandb
                if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':

                    # log sampling val metrics 
                    wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_dists': torch.mean(metric_dists(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            # f'sampling_val_METRIC/val_fid': torch.mean(metric_fid(
                            #                                                                 restored_img, 
                            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_niqe': torch.mean(metric_niqe(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_musiq': torch.mean(metric_musiq(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_maniqa': torch.mean(metric_maniqa(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            f'sampling_val_METRIC/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                    restored_img, 
                                                                                    torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                            })
                    
                    # log sampling val images 
                    wandb.log({ f'sampling_val_FINAL_VIS/{val_batch_idx}_val_gt': wandb.Image((val_gt + 1) / 2, caption=f'gt_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_lq': wandb.Image(val_lq, caption=f'lq_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_cleaned': wandb.Image(val_clean, caption=f'cleaned_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_sampled': wandb.Image(torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), caption=f'sampled_img'),
                                f'sampling_val_FINAL_VIS/{val_batch_idx}_val_prompt': wandb.Image(log_txt_as_img((256, 256), val_prompt), caption=f'prompt'),
                            })
                    wandb.log({f'sampling_val_FINAL_VIS/{val_batch_idx}_val_all': wandb.Image(torch.concat([val_lq, val_clean, torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), val_gt], dim=2), caption='lq_clean_sample,gt')})

        
    # average using numpy
    tot_val_psnr = np.array(tot_val_psnr).mean()
    tot_val_ssim = np.array(tot_val_ssim).mean()
    tot_val_lpips = np.array(tot_val_lpips).mean()
    tot_val_dists = np.array(tot_val_dists).mean()
    # tot_val_fid = np.array(tot_val_fid).mean()
    tot_val_niqe = np.array(tot_val_niqe).mean()
    tot_val_musiq = np.array(tot_val_musiq).mean()
    tot_val_maniqa = np.array(tot_val_maniqa).mean()
    tot_val_clipiqa = np.array(tot_val_clipiqa).mean()


    # log total val metrics 
    if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
        wandb.log({
            f'sampling_val_METRIC/tot_val_psnr': tot_val_psnr,
            f'sampling_val_METRIC/tot_val_ssim': tot_val_ssim,
            f'sampling_val_METRIC/tot_val_lpips': tot_val_lpips,
            f'sampling_val_METRIC/tot_val_dists': tot_val_dists,
            # f'sampling_val_METRIC/tot_val_fid': tot_val_fid,
            f'sampling_val_METRIC/tot_val_niqe': tot_val_niqe,
            f'sampling_val_METRIC/tot_val_musiq': tot_val_musiq,
            f'sampling_val_METRIC/tot_val_maniqa': tot_val_maniqa,
            f'sampling_val_METRIC/tot_val_clipiqa': tot_val_clipiqa,
        })
        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
