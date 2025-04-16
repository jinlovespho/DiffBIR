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


def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(231, device_specific=True)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)


    # load data
    _, val_ds, _, val_loader = initialize.load_data(accelerator, cfg)


    # load models
    models, resume_ckpt_path = initialize.load_model(accelerator, device, args, cfg)
    

    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}
    val_loader = accelerator.prepare(val_loader)


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # Validation
    for val_batch in val_loader:

        # load val data
        to(val_batch, device)
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



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
