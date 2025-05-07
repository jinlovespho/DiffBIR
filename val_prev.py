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
from diffbir.dataset.pho_utils import encode, decode 
from accelerate.utils import DistributedDataParallelKwargs


def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)


    # load logging tools and ckpt directory
    if accelerator.is_main_process:
        exp_dir, ckpt_dir, exp_name, writer = initialize.load_experiment_settings(accelerator, cfg)


    # load data
    _, val_ds, _, val_loader = initialize.load_data(accelerator, cfg)
    batch_transform = instantiate_from_config(cfg.batch_transform)


    # load models
    models, resume_ckpt_path = initialize.load_model(accelerator, device, args, cfg)
    

    # # set training params
    # train_params, train_model_names = initialize.set_training_params(accelerator, models, cfg)


    # # setup optimizer
    # opt = torch.optim.AdamW(train_params, lr=cfg.train.learning_rate)


    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}
    val_loader = accelerator.prepare(val_loader)


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # SR metrics
    metric_psnr = pyiqa.create_metric('psnr', device=device)
    metric_ssim = pyiqa.create_metric('ssimc', device=device)
    metric_lpips = pyiqa.create_metric('lpips', device=device)


    # print Training Info
    if accelerator.is_main_process:
        print('='*50)
        print(f'Experiment name: {exp_name}')
        # print(f"Training steps: {cfg.train.train_steps}")
        print(f"Experiment directory: {exp_dir}")
        # print(f"Num train_dataset: {len(train_ds):,}")
        print(f"Num val_dataset: {len(val_ds):,}")
        print(f'Loaded models: {list(models.keys())}')
        print(f'Finetuning Method: {cfg.exp_args.finetuning_method}')
        print(f'Resume training ckpt: ', resume_ckpt_path)
        print(f'OCR pretrained ckpt: {cfg.exp_args.testr_ckpt_dir}')
        print('='*50)



    tot_val_psnr=[]
    tot_val_ssim=[]
    tot_val_lpips=[]

    # Validation
    for val_batch in val_loader:

        # load val data
        to(val_batch, device)
        val_batch = batch_transform(val_batch)
        val_gt, val_lq, val_prompt, val_texts, val_boxes, val_polys, val_text_encs, val_img_name = val_batch 
        val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float()   # b 3 512 512
        val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
        val_bs, _, val_H, val_W = val_gt.shape


        # val_prompt is null prompts !!


        # put models on evaluation for sampling
        for model in models.values():
            if isinstance(model, nn.Module):
                model.eval()

        # prepare vae, condition
        with torch.no_grad():
            # val_z_0 = pure_cldm.vae_encode(val_gt)
            val_clean = models['swinir'](val_lq)
            val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

            # set number of val imgs to log
            M = cfg.val.log_num_val_img
            val_log_clean = val_clean[:M]
            val_log_cond = {k: v[:M] for k, v in val_cond.items()}
            val_log_gt, val_log_lq = val_gt[:M], val_lq[:M]
            val_log_prompt = val_prompt[:M]
            
            # sampling
            val_z, val_sampled_unet_feats = sampler.sample(     # 6 4 56 56
                model=models['cldm'],
                device=device,
                steps=50,
                x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),   # manual shape adjustment
                cond=val_log_cond,
                uncond=None,
                cfg_scale=1.0,
                progress=accelerator.is_main_process,
                cfg=cfg
            )

            # =========================== OCR ===========================
            if cfg.exp_args.model_name == 'diffbir_onlybox' or cfg.exp_args.model_name == 'diffbir_testr':

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
                        vis_val_gt = val_gt[i]                                # 3 512 512 [-1,1]
                        vis_val_gt = (vis_val_gt + 1)/2 * 255.0         # 3 512 512 [0,255]
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
                            wandb.log({f'sampling_val_VIS_iter{sampled_iter}_timestep{sampled_timestep}/poly{i}': wandb.Image(vis_val_gt, caption=f'draw sampled val ocr results on gt')})



                        # vis_val_box=[]
                        # for i in range(M):
                        #     vis_val_gt = val_gt[i]              # 3 512 512
                        #     vis_val_gt = (vis_val_gt + 1) / 2 * 255.0

                        #     # only label bbox for gt img
                        #     vis_val_gt = vis_val_gt.permute(1,2,0).detach().cpu().numpy().astype(np.uint8).copy()  # 512 512 3

                        #     for cx, cy, w, h in val_enc_box_points[i]:
                        #         # Convert cx, cy, w, h (normalized) to x1, y1, x2, y2 (absolute coordinates)
                        #         x1 = int((cx - w / 2) * 512)
                        #         y1 = int((cy - h / 2) * 512)
                        #         x2 = int((cx + w / 2) * 512)
                        #         y2 = int((cy + h / 2) * 512)

                        #         # Draw the rectangle on the image
                        #         cv2.rectangle(vis_val_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                        #     vis_val_gt = torch.tensor(vis_val_gt)
                        #     vis_val_gt = vis_val_gt.cuda().permute(2,0,1).float() / 255.0
                        #     vis_val_box.append(vis_val_gt)

                        # vis_val_box = torch.stack(vis_val_box)        # b 3 512 512
                        # wandb.log({f'sampling_val_iter{sampled_iter}_timestep{sampled_timestep}/val_vis_box{i}': wandb.Image(vis_val_box, caption=f'draw sampled val pred box on gt')})


            # log total psnr, ssim, lpips for val
            tot_val_psnr.append(torch.mean(metric_psnr(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
            tot_val_ssim.append(torch.mean(metric_ssim(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
            tot_val_lpips.append(torch.mean(metric_lpips(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())


            # log sampling val imgs to wandb
            if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':

                # log sampling val metrics 
                wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
                                                                                torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
                                                                                torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
                                                                                torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                        })
                
                # log sampling val images 
                wandb.log({ f'sampling_val_FINAL_VIS/val_gt': wandb.Image((val_log_gt + 1) / 2, caption=f'gt_img'),
                            f'sampling_val_FINAL_VIS/val_lq': wandb.Image(val_log_lq, caption=f'lq_img'),
                            f'sampling_val_FINAL_VIS/val_cleaned': wandb.Image(val_log_clean, caption=f'cleaned_img'),
                            f'sampling_val_FINAL_VIS/val_sampled': wandb.Image(torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), caption=f'sampled_img'),
                            f'sampling_val_FINAL_VIS/val_prompt': wandb.Image(log_txt_as_img((256, 256), val_log_prompt), caption=f'prompt'),
                        })
                wandb.log({f'sampling_val_FINAL_VIS/val_all': wandb.Image(torch.concat([val_log_lq, val_log_clean, torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), val_log_gt], dim=2), caption='lq_clean_sample,gt')})

        # put models back to training 
        for model in models.values():
            if isinstance(model, nn.Module):
                model.train()


    # average using numpy
    tot_val_psnr = np.array(tot_val_psnr).mean()
    tot_val_ssim = np.array(tot_val_ssim).mean()
    tot_val_lpips = np.array(tot_val_lpips).mean()


    # log total val metrics 
    if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
        wandb.log({
            f'sampling_val_METRIC/tot_val_psnr': tot_val_psnr,
            f'sampling_val_METRIC/tot_val_ssim': tot_val_ssim,
            f'sampling_val_METRIC/tot_val_lpips': tot_val_lpips,
        })
        

    # print end of experiment
    if accelerator.is_main_process:
        print("FINISH !!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
