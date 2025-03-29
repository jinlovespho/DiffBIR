from argparse import ArgumentParser
import copy
from omegaconf import OmegaConf
import torch
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

def main(args) -> None:

    # set accelerator, seed, device, config
    accelerator = Accelerator(split_batches=True)
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    

    # load logging tools and ckpt directory
    exp_dir, ckpt_dir, writer = initialize.load_experiment_settings(accelerator, cfg)


    # load data
    train_ds, val_ds, train_loader, val_loader = initialize.load_data(accelerator, cfg)
    batch_transform = instantiate_from_config(cfg.batch_transform)


    # load models
    models = initialize.load_model(accelerator, device, cfg)
    

    # set training params
    train_params = initialize.set_training_params(accelerator, models, cfg)


    # setup optimizer
    opt = torch.optim.AdamW(train_params, lr=cfg.train.learning_rate)


    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)
    

    # setup accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}
    opt, train_loader, val_loader = accelerator.prepare(opt, train_loader, val_loader)


    # etc
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])
    noise_aug_timestep = cfg.train.noise_aug_timestep


    # print Training Info
    if accelerator.is_main_process:
        print('='*50)
        print(f"Training steps: {cfg.train.train_steps}")
        print(f"Experiment directory: {exp_dir}")
        print(f"Num train_dataset: {len(train_ds):,}")
        print(f"Num val_dataset: {len(val_ds):,}")
        print(f'Loaded models: {list(models.keys())}')
        print(f'Finetuning Method: {cfg.exp_args.finetuning_method}')
        print('='*50)


    # setup variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []

    # Training Loop
    while global_step < max_steps:

        pbar = tqdm( iterable=None, disable=not accelerator.is_main_process, unit="batch", total=len(train_loader), )
        for batch in train_loader:

            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, prompt, texts, boxes, text_encs, img_name = batch

            # # JLP - VIS BATCH IMG, BOX, TEXT
            # for i in range(cfg.train.batch_size):
            #     vis_gt = gt[i]
            #     vis_lq = lq[i]
            #     vis_gt = (vis_gt-vis_gt.min()) / (vis_gt.max()-vis_gt.min()) * 255.0
            #     vis_lq = (vis_lq-vis_lq.min()) / (vis_lq.max()-vis_lq.min()) * 255.0 
            #     vis_gt = vis_gt.detach().cpu().numpy().copy()
            #     vis_lq = vis_lq.detach().cpu().numpy().copy()
            #     # draw box and text
            #     for box_coord, txt in zip(boxes[i], texts[i]):
            #         x,y,w,h = box_coord 
            #         cv2.rectangle(vis_gt, (x,y), (x+w,y+h), (0,255,0), 2)
            #         cv2.putText(vis_gt, txt, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            #     cv2.imwrite(f'./vis/textocr_img_{i}_gt.jpg', vis_gt[...,::-1])
            #     cv2.imwrite(f'./vis/textocr_img_{i}_lq.jpg', vis_lq[:,:,::-1])

            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()   # b 3 512 512
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()   # b 3 512 512

            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = models['swinir'](lq)
                cond = pure_cldm.prepare_condition(clean, prompt)
                # noise augmentation
                cond_aug = copy.deepcopy(cond)
                if noise_aug_timestep > 0:
                    cond_aug["c_img"] = diffusion.q_sample(
                        x_start=cond_aug["c_img"],
                        t=torch.randint(
                            0, noise_aug_timestep, (z_0.shape[0],), device=device
                        ),
                        noise=torch.randn_like(cond_aug["c_img"]),
                    )
            t = torch.randint(
                0, diffusion.num_timesteps, (z_0.shape[0],), device=device
            )

            loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 8
                log_clean = clean[:N]
                log_cond = {k: v[:N] for k, v in cond.items()}
                log_cond_aug = {k: v[:N] for k, v in cond_aug.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm,
                        device=device,
                        steps=50,
                        x_size=(len(log_gt), *z_0.shape[1:]),
                        cond=log_cond,
                        uncond=None,
                        cfg_scale=1.0,
                        progress=accelerator.is_main_process,
                    )
                    if accelerator.is_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
                            ("image/condition", log_clean),
                            (
                                "image/condition_decoded",
                                (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/condition_aug_decoded",
                                (pure_cldm.vae_decode(log_cond_aug["c_img"]) + 1) / 2,
                            ),
                            (
                                "image/prompt",
                                (log_txt_as_img((512, 512), log_prompt) + 1) / 2,
                            ),
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
