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
import cv2 
import numpy as np
import wandb
import pyiqa
from torchvision.utils import save_image 
from torchvision.transforms.functional import to_pil_image


def main(args):


    # set accelerator, seed, device, config
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    

    # load logging tools and ckpt directory
    if accelerator.is_main_process:
        exp_dir, ckpt_dir, writer = initialize.load_experiment_settings(accelerator, cfg)


    # load data
    train_ds, val_ds, train_loader, val_loader = initialize.load_data(accelerator, cfg)
    batch_transform = instantiate_from_config(cfg.batch_transform)


    # load models
    models = initialize.load_model(accelerator, device, args, cfg)
    

    # set training params
    train_params, train_model_names = initialize.set_training_params(accelerator, models, cfg)


    # setup optimizer
    opt = torch.optim.AdamW(train_params, lr=cfg.train.learning_rate)


    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}
    opt, train_loader, val_loader = accelerator.prepare(opt, train_loader, val_loader)


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # SR metrics
    metric_psnr = pyiqa.create_metric('psnr', device=device)
    metric_ssim = pyiqa.create_metric('ssimc', device=device)
    metric_lpips = pyiqa.create_metric('lpips', device=device)


    # print Training Info
    if accelerator.is_main_process:
        print('='*50)
        print(f"Training steps: {cfg.train.train_steps}")
        print(f"Experiment directory: {exp_dir}")
        print(f"Num train_dataset: {len(train_ds):,}")
        print(f"Num val_dataset: {len(val_ds):,}")
        print(f'Loaded models: {list(models.keys())}')
        print(f'Finetuning Method: {cfg.exp_args.finetuning_method}')
        print(f'Ctrl_point type: ', {cfg.dataset.train.params.data_args['ctrl_point']})
        print(f'Bbox_format: ', {cfg.dataset.train.params.data_args['bbox_format']})
        print('='*50)


    # setup variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    globalstep_total_loss = []

    epoch = 0
    epoch_total_loss = []

    diffusion_loss = []

    ocr_enc_cls_loss=[]
    ocr_enc_box_loss=[]
    ocr_enc_giou_loss=[]

    # train_psnr=0.0
    # train_ssim=0.0
    # train_lpips=0.0

    # Training Loop
    while global_step < max_steps:

        # TRAINING
        pbar = tqdm( iterable=None, disable=not accelerator.is_main_process, unit="batch", total=len(train_loader), )
        for batch in train_loader:


            # log basic info while training
            if accelerator.is_main_process:
                if cfg.log_args.log_tool == 'wandb':
                    wandb.log({'global_step': global_step, 'epoch': epoch,'learning_rate': opt.param_groups[0]['lr'], })


            # load training data
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, train_prompt, texts, boxes, text_encs, img_name = batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()   # b 3 512 512
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()   # b 3 512 512
            train_bs = gt.shape[0]


            # use null prompt for now
            train_prompt=["" for i in range(train_bs)]


            # # JLP - set box format to xywh and visualize box
            # for i in range(cfg.train.batch_size):
            #     vis_gt = gt[i]  # has range [-1,1]
            #     vis_lq = lq[i]  # has range [0,1]
            #     vis_gt = (vis_gt + 1) / 2.0 * 255.0
            #     vis_lq = vis_lq * 255.0 
            #     vis_gt = vis_gt.permute(1,2,0).detach().cpu().numpy().copy()
            #     vis_lq = vis_lq.permute(1,2,0).detach().cpu().numpy().copy()
            #     # draw box and text
            #     for box_coord, txt in zip(boxes[i], texts[i]):
            #         x,y,w,h = list(map(lambda x: int(x), box_coord)) 
            #         cv2.rectangle(vis_gt, (x,y), (x+w,y+h), (0,255,0), 2)
            #         cv2.putText(vis_gt, txt, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            #     # cv2.imwrite(f'./vis/textocr_img_{i}_gt.jpg', vis_gt[...,::-1])
            #     # cv2.imwrite(f'./vis/textocr_img_{i}_lq.jpg', vis_lq[:,:,::-1])
            #     cv2.imwrite(f'textocr_img_{i}_gt.jpg', vis_gt[...,::-1])
            #     cv2.imwrite(f'textocr_img_{i}_lq.jpg', vis_lq[:,:,::-1])


            # prepare VAE, condition, timestep
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)                              # b 4 64 64
                clean = models['swinir'](lq)                                          # b 3 512 512
                cond = pure_cldm.prepare_condition(clean, train_prompt)     # cond['c_txt'], cond['c_img']
                # noise augmentation
                cond_aug = copy.deepcopy(cond)


            # sample random training timesteps and obtain diffusion loss
            t = torch.randint(0, diffusion.num_timesteps, (train_bs,), device=device)
            diff_loss, extracted_feats = diffusion.p_losses(models['cldm'], z_0, t, cond_aug)
            

            # =========================== OCR ===========================
            if cfg.exp_args.model_name == 'diffbir_onlybox':

                # process annotations for OCR training loss
                train_targets=[]
                for i in range(train_bs):
                    num_box=len(boxes[i])
                    tmp_dict={}
                    tmp_dict['labels'] = torch.tensor([0]*num_box).cuda()  # 0 for text
                    tmp_dict['boxes'] = torch.tensor(boxes[i]).cuda()
                    tmp_dict['texts'] = text_encs[i]
                    train_targets.append(tmp_dict)


                # OCR model forward pass
                ocr_loss_dict, _ = models['testr_detector'](extracted_feats, train_targets)


                # OCR loss
                ocr_loss = sum(ocr_loss_dict.values())
                ocr_enc_cls = ocr_loss_dict['loss_ce_enc']
                ocr_enc_box = ocr_loss_dict['loss_bbox_enc']
                ocr_enc_giou = ocr_loss_dict['loss_giou_enc']

                # print(f'loss_ocr_enc_cls: {ocr_enc_cls.item():.3f}')
                # print(f'loss_ocr_enc_box: {ocr_enc_box.item():.3f}')
                # print(f'loss_ocr_enc_giou: {ocr_enc_giou.item():.3f}')
                # print('- - '*5)
                # print(f'tot_loss_ocr_enc: {ocr_loss.item():.3f}')
                # print(f'loss_diff: {diff_loss.item():.3f}')
                # print('='*50)


            # TOTAL LOSS FUNCTION
            total_loss = diff_loss + ocr_loss


            # calculate gradient and update model
            opt.zero_grad()
            accelerator.backward(total_loss)
            opt.step()
            accelerator.wait_for_everyone()
            global_step += 1


            # gather losses for logging
            diffusion_loss.append(diff_loss.item())
            ocr_enc_cls_loss.append(ocr_enc_cls.item())
            ocr_enc_box_loss.append(ocr_enc_box.item())
            ocr_enc_giou_loss.append(ocr_enc_giou.item())
            globalstep_total_loss.append(total_loss.item())
            epoch_total_loss.append(total_loss.item())


            # set terminal logging visualization
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Diff_Loss: {diff_loss.item():.6f}")


            # Log gathered training losses
            if global_step % cfg.train.log_loss_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_diffusion_loss = (accelerator.gather(torch.tensor(diffusion_loss, device=device).unsqueeze(0)).mean().item())
                avg_ocr_enc_cls_loss = (accelerator.gather(torch.tensor(ocr_enc_cls_loss, device=device).unsqueeze(0)).mean().item())
                avg_ocr_enc_box_loss = (accelerator.gather(torch.tensor(ocr_enc_box_loss, device=device).unsqueeze(0)).mean().item())
                avg_ocr_enc_giou_loss = (accelerator.gather(torch.tensor(ocr_enc_giou_loss, device=device).unsqueeze(0)).mean().item())
                avg_globalstep_total_loss = (accelerator.gather(torch.tensor(globalstep_total_loss, device=device).unsqueeze(0)).mean().item())
                # clear list
                diffusion_loss.clear()
                ocr_enc_cls_loss.clear()
                ocr_enc_box_loss.clear()
                ocr_enc_giou_loss.clear()
                globalstep_total_loss.clear()
                # log to wandb
                if accelerator.is_main_process:
                    if cfg.log_args.log_tool == 'wandb':
                        wandb.log({"train_loss/diffusion_loss": avg_diffusion_loss})
                        wandb.log({"train_loss/ocr_enc_cls_loss": avg_ocr_enc_cls_loss})
                        wandb.log({"train_loss/ocr_enc_box_loss": avg_ocr_enc_box_loss})
                        wandb.log({"train_loss/ocr_enc_giou_loss": avg_ocr_enc_giou_loss})
                        wandb.log({"train_loss/total_globalstep_loss": avg_globalstep_total_loss})



            # ======================== SAVE MODEL ========================
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    # save only controlnet 
                    if cfg.exp_args.finetuning_method == 'only_ctrlnet':
                        ckpt = pure_cldm.controlnet.state_dict()
                    elif cfg.exp_args.finetuning_method == 'asdf':
                        # ckpt = other_models 
                        pass
                    ckpt_path = f"{ckpt_dir}/{cfg.exp_args.finetuning_method}_{global_step:07d}.pt"
                    torch.save(ckpt, ckpt_path)
            # =============================================================


            # Diffusion sampling(inference) with training data
            if global_step % cfg.train.log_image_every == 0 or global_step == 1:


                # set number of training images to log
                N = cfg.train.log_num_train_img
                log_clean = clean[:N]                                       # b 3 512 512
                log_cond = {k: v[:N] for k, v in cond.items()}              
                log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]                             # b  3 512 512
                log_prompt = train_prompt[:N]


                # put models on evaluation for sampling
                for model in models.values():
                    if isinstance(model, nn.Module):
                        model.eval()


                # sampling 
                with torch.no_grad():
                    z, train_sampled_unet_feats = sampler.sample(     # b 4 64 64
                        model=models['cldm'],
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                        cfg=cfg 
                    )


                    if cfg.exp_args.model_name == 'diffbir_onlybox':

                        # evaluate diffusion features for different timesteps
                        for sampled_iter, sampled_timestep, unet_feats in train_sampled_unet_feats:

                            # OCR model forward pass
                            train_ocr_loss_dict, train_enc_box_points = models['testr_detector'](unet_feats, train_targets)

                            # train OCR loss 
                            train_ocr_enc_cls = train_ocr_loss_dict['loss_ce_enc']
                            train_ocr_enc_box = train_ocr_loss_dict['loss_bbox_enc']
                            train_ocr_enc_giou = train_ocr_loss_dict['loss_giou_enc']
                            train_ocr_loss = sum(train_ocr_loss_dict.values())

                            # log sampling train loss and box to wandb
                            if accelerator.is_main_process:
                                if cfg.log_args.log_tool == 'wandb':

                                    # log OCR loss 
                                    wandb.log({f"sampling_train_iter{sampled_iter}_timestep{sampled_timestep}/train_ocr_enc_cls_loss": train_ocr_enc_cls.item()})
                                    wandb.log({f"sampling_train_iter{sampled_iter}_timestep{sampled_timestep}/train_ocr_enc_box_loss": train_ocr_enc_box.item()})
                                    wandb.log({f"sampling_train_iter{sampled_iter}_timestep{sampled_timestep}/train_ocr_enc_giou_loss": train_ocr_enc_giou.item()})
                                    wandb.log({f"sampling_train_iter{sampled_iter}_timestep{sampled_timestep}/train_total_enc_loss": train_ocr_loss.item()})


                                    # log OCR bbox prediction
                                    vis_train_box=[]
                                    for i in range(N):
                                        vis_train_gt = gt[i]                                # 3 512 512 [-1,1]
                                        vis_train_gt = (vis_train_gt + 1)/2 * 255.0         # 3 512 512 [0,255]
                                        vis_train_gt = vis_train_gt.permute(1,2,0).detach().cpu().numpy().astype(np.uint8).copy()  # 512 512 3
                                        # convert cxcywh -> xyxy format for visualization
                                        for cx, cy, w, h in train_enc_box_points[i]:
                                            x1 = int((cx - w / 2) * 512)
                                            y1 = int((cy - h / 2) * 512)
                                            x2 = int((cx + w / 2) * 512)
                                            y2 = int((cy + h / 2) * 512)
                                            cv2.rectangle(vis_train_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        vis_train_gt = torch.tensor(vis_train_gt)
                                        vis_train_gt = vis_train_gt.cuda().permute(2,0,1).float() / 255.0
                                        vis_train_box.append(vis_train_gt)
                                    vis_train_box = torch.stack(vis_train_box)        # b 3 512 512
                                    wandb.log({f'sampling_train_iter{sampled_iter}_timestep{sampled_timestep}/train_vis_box{i}': wandb.Image(vis_train_box, caption=f'draw sampled training pred box on gt')})


                    ## I dont think we have to log total psnr, ssim, lpips for training..
                    # train_psnr += torch.mean(metric_psnr(torch.clamp((pure_cldm.vae_decode(z) + 1) / 2, min=0, max=1),torch.clamp((log_gt + 1) / 2, min=0, max=1))).item()
                    # train_ssim += torch.mean(metric_ssim(torch.clamp((pure_cldm.vae_decode(z) + 1) / 2, min=0, max=1),torch.clamp((log_gt + 1) / 2, min=0, max=1))).item()
                    # train_lpips += torch.mean(metric_lpips(torch.clamp((pure_cldm.vae_decode(z) + 1) / 2, min=0, max=1),torch.clamp((log_gt + 1) / 2, min=0, max=1))).item()
                    

                    # log sampling training metric results
                    if accelerator.is_main_process:
                        if cfg.log_args.log_tool == 'wandb':

                            # log sampling train metrics 
                            wandb.log({f'sampling_train_metric/train_psnr': torch.mean(metric_psnr(
                                                                                                torch.clamp((pure_cldm.vae_decode(z) + 1) / 2, min=0, max=1),
                                                                                                torch.clamp((log_gt + 1) / 2, min=0, max=1)) ).item(),
                                       f'sampling_train_metric/train_ssim': torch.mean(metric_ssim(
                                                                                                torch.clamp((pure_cldm.vae_decode(z) + 1) / 2, min=0, max=1),
                                                                                                torch.clamp((log_gt + 1) / 2, min=0, max=1))).item(),
                                       f'sampling_train_metric/train_lpips': torch.mean(metric_lpips(
                                                                                                torch.clamp((pure_cldm.vae_decode(z) + 1) / 2, min=0, max=1),
                                                                                                torch.clamp((log_gt + 1) / 2, min=0, max=1))).item(), 
                            })

                            # log sampling training images
                            wandb.log({ f'sampling_train_vis/train_gt': wandb.Image((log_gt + 1) / 2, caption=f'gt_img'),
                                        f'sampling_train_vis/train_lq': wandb.Image(log_lq, caption=f'lq_img'),
                                        f'sampling_train_vis/train_cleaned': wandb.Image(log_clean, caption=f'cleaned_img'),
                                        f'sampling_train_vis/train_sampled': wandb.Image((pure_cldm.vae_decode(z) + 1) / 2, caption=f'sampled_img'),
                                        f'sampling_train_vis/train_prompt': wandb.Image(log_txt_as_img((256, 256), log_prompt), caption=f'prompt'),
                                       })
                            wandb.log({f'sampling_train_vis/train_all': wandb.Image(torch.concat([log_lq, log_clean, (pure_cldm.vae_decode(z) + 1) / 2, log_gt], dim=2), caption='lq_clean_sample,gt')})


                # put models back to training 
                for model in models.values():
                    if isinstance(model, nn.Module):
                        model.train()


            # log validation images 
            if global_step % cfg.val.log_image_every == 0 or global_step == 1:

                tot_val_psnr=0.0
                tot_val_ssim=0.0
                tot_val_lpips=0.0

                # Validation
                for val_batch in val_loader:

                    # load val data
                    to(val_batch, device)
                    val_batch = batch_transform(val_batch)
                    val_gt, val_lq, _, val_texts, val_boxes, val_text_encs, val_img_name = val_batch 
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float()   # b 3 512 512
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
                    val_bs, _, val_H, val_W = val_gt.shape


                    # use null prompt for validation for now
                    val_prompt=["" for i in range(val_bs)]


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
                        if cfg.exp_args.model_name == 'diffbir_onlybox':

                            # process annotations for OCR val loss 
                            val_targets=[]
                            for i in range(val_bs):
                                num_box=len(val_boxes[i])
                                tmp_dict={}
                                tmp_dict['labels'] = torch.tensor([0]*num_box).cuda()  # 0 for text
                                tmp_dict['boxes'] = torch.tensor(val_boxes[i]).cuda()
                                tmp_dict['texts'] = val_text_encs[i]
                                val_targets.append(tmp_dict)


                            # evaluate diffusion features for different timesteps
                            for sampled_iter, sampled_timestep, unet_feats in val_sampled_unet_feats:

                                # OCR model forward pass
                                val_ocr_loss_dict, val_enc_box_points = models['testr_detector'](unet_feats, val_targets)


                                # val OCR loss
                                val_ocr_enc_cls = val_ocr_loss_dict['loss_ce_enc']
                                val_ocr_enc_box = val_ocr_loss_dict['loss_bbox_enc']
                                val_ocr_enc_giou = val_ocr_loss_dict['loss_giou_enc']
                                val_ocr_loss = sum(val_ocr_loss_dict.values())


                                # log sampling val loss and box to wandb
                                if accelerator.is_main_process:
                                    if cfg.log_args.log_tool == 'wandb':
                                        # log loss values 
                                        wandb.log({f"sampling_val_iter{sampled_iter}_timestep{sampled_timestep}/val_ocr_enc_cls_loss": val_ocr_enc_cls.item()})
                                        wandb.log({f"sampling_val_iter{sampled_iter}_timestep{sampled_timestep}/val_ocr_enc_box_loss": val_ocr_enc_box.item()})
                                        wandb.log({f"sampling_val_iter{sampled_iter}_timestep{sampled_timestep}/val_ocr_enc_giou_loss": val_ocr_enc_giou.item()})
                                        wandb.log({f"sampling_val_iter{sampled_iter}_timestep{sampled_timestep}/val_total_enc_loss": val_ocr_loss.item()})


                                    vis_val_box=[]
                                    for i in range(M):
                                        vis_val_gt = val_gt[i]              # 3 512 512
                                        vis_val_gt = (vis_val_gt + 1) / 2 * 255.0

                                        # only label bbox for gt img
                                        vis_val_gt = vis_val_gt.permute(1,2,0).detach().cpu().numpy().astype(np.uint8).copy()  # 512 512 3

                                        for cx, cy, w, h in val_enc_box_points[i]:
                                            # Convert cx, cy, w, h (normalized) to x1, y1, x2, y2 (absolute coordinates)
                                            x1 = int((cx - w / 2) * 512)
                                            y1 = int((cy - h / 2) * 512)
                                            x2 = int((cx + w / 2) * 512)
                                            y2 = int((cy + h / 2) * 512)

                                            # Draw the rectangle on the image
                                            cv2.rectangle(vis_val_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                                        vis_val_gt = torch.tensor(vis_val_gt)
                                        vis_val_gt = vis_val_gt.cuda().permute(2,0,1).float() / 255.0
                                        vis_val_box.append(vis_val_gt)

                                    vis_val_box = torch.stack(vis_val_box)        # b 3 512 512
                                    wandb.log({f'sampling_val_iter{sampled_iter}_timestep{sampled_timestep}/val_vis_box{i}': wandb.Image(vis_val_box, caption=f'draw sampled val pred box on gt')})


                        # log total psnr, ssim, lpips for val
                        tot_val_psnr += torch.mean(metric_psnr(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item()
                        tot_val_ssim += torch.mean(metric_ssim(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item()
                        tot_val_lpips += torch.mean(metric_lpips(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item()


                        # log sampling val imgs to wandb
                        if accelerator.is_main_process:
                            if cfg.log_args.log_tool == 'wandb':

                                # log sampling val metrics 
                                wandb.log({f'sampling_val_metric/val_psnr': torch.mean(metric_psnr(
                                                                                                torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                                torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                        f'sampling_val_metric/val_ssim': torch.mean(metric_ssim(
                                                                                                torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                                torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                        f'sampling_val_metric/val_lpips': torch.mean(metric_lpips(
                                                                                                torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                                torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                        })
                                
                                # log sampling val images 
                                wandb.log({ f'sampling_val_vis/val_gt': wandb.Image((val_log_gt + 1) / 2, caption=f'gt_img'),
                                            f'sampling_val_vis/val_lq': wandb.Image(val_log_lq, caption=f'lq_img'),
                                            f'sampling_val_vis/val_cleaned': wandb.Image(val_log_clean, caption=f'cleaned_img'),
                                            f'sampling_val_vis/val_sampled': wandb.Image((pure_cldm.vae_decode(val_z) + 1) / 2, caption=f'sampled_img'),
                                            f'sampling_val_vis/val_prompt': wandb.Image(log_txt_as_img((256, 256), val_log_prompt), caption=f'prompt'),
                                        })
                                wandb.log({f'sampling_val_vis/val_all': wandb.Image(torch.concat([val_log_lq, val_log_clean, (pure_cldm.vae_decode(val_z) + 1) / 2, val_log_gt], dim=2), caption='lq_clean_sample,gt')})

                    # put models back to training 
                    for model in models.values():
                        if isinstance(model, nn.Module):
                            model.train()


                # log total val metrics 
                if accelerator.is_main_process:
                    if cfg.log_args.log_tool == 'wandb':
                        wandb.log({
                            f'sampling_val_metric/total_val_psnr': tot_val_psnr,
                            f'sampling_val_metric/total_val_ssim': tot_val_ssim,
                            f'sampling_val_metric/total_val_lpips': tot_val_lpips,
                        })
                

            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break


        pbar.close()
        epoch += 1
        avg_epoch_total_loss = (accelerator.gather(torch.tensor(epoch_total_loss, device=device).unsqueeze(0)).mean().item())
        epoch_total_loss.clear()
        if accelerator.is_main_process:
            if cfg.log_args.log_tool == 'wandb':
                wandb.log({"epoch_loss/epoch_diffusion_loss": avg_epoch_total_loss})


    # print end of experiment
    if accelerator.is_main_process:
        print("FINISH !!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
