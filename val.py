from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.model import ControlLDM, Diffusion
from diffbir.sampler import SpacedSampler
import initialize
from accelerate.utils import DistributedDataParallelKwargs
import numpy as np


def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)


    # load models
    models, resume_ckpt_path = initialize.load_model(accelerator, device, args, cfg)
    

    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # Set dummy input
    val_bs=1
    val_lq = torch.rand(val_bs,3,512,512).to(device)    # input image
    val_prompt=["" for _ in range(val_bs)]              # null prompt


    # put models on evaluation for sampling
    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()


    # prepare vae, condition
    with torch.no_grad():
        val_clean = models['swinir'](val_lq)
        val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

        # Diffusion sampling
        val_z, val_sampled_unet_feats = sampler.sample(     # 6 4 56 56
            model=models['cldm'],
            device=device,
            steps=50,
            x_size=(val_bs, 4, int(512/8), int(512/8)),   # manual shape adjustment
            cond=val_cond,
            uncond=None,
            cfg_scale=1.0,
            progress=accelerator.is_main_process,
            cfg=cfg
        )


        # ------------------------------- OCR -------------------------------


        # Warm-up
        for i in range(5):
            for _, _, unet_feats in val_sampled_unet_feats:
                _ = models['testr'](unet_feats)
        torch.cuda.synchronize()

        
        # set recording for OCR forward pass
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


        inference_time=[]
        # evaluate diffusion features for different timesteps
        for sampled_iter, sampled_timestep, unet_feats in val_sampled_unet_feats:
            

            # start recording
            starter.record()


            # forward pass (the actual forward pass is implemented inside ./testr/adet/modeling/transformer_detector.py)
            _ = models['testr'](unet_feats)


            # end recording
            ender.record()
            torch.cuda.synchronize()

            
            # calcualate inference time
            inf_time = starter.elapsed_time(ender)
            inference_time.append(inf_time)


        print('OCR inference time for each sampling steps: ', inference_time)           # mili-second
        print('OCR Avg inference time: ', sum(inference_time)/len(inference_time))




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
