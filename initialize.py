import os 
import wandb 
import argparse
from omegaconf import OmegaConf
from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.dataset.pho_codeformer import collate_fn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch 

def load_experiment_settings(accelerator, cfg):


    # setup an experiment folder
    exp_dir = cfg.train.exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)


    # setup logging tool
    if cfg.log_args.log_tool == 'wandb':
        wandb.login(key=cfg.log_args.wandb_key)
        wandb.init(project=cfg.log_args.wandb_proj_name, 
                name=cfg.log_args.wandb_exp_name, 
                config=argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))
        )
        return exp_dir, ckpt_dir, None
    
    elif cfg.log_args.log_tool == 'tensorboard':
        writer = SummaryWriter(exp_dir)
        return exp_dir, ckpt_dir, writer

def load_data(accelerator, cfg):

    # set dataset 
    train_ds = instantiate_from_config(cfg.dataset.train)
    val_ds = instantiate_from_config(cfg.dataset.val)

    # set data loader 
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_ds, val_ds, train_loader, val_loader 



def load_model(accelerator, device, cfg):

    loaded_models={}

    if cfg.exp_args.model_name == 'diffbir_baseline':

        cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
        sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
        unused, missing = cldm.load_pretrained_sd(sd)
        if accelerator.is_main_process:
            print(
                f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
                f"unused weights: {unused}\n"
                f"missing weights: {missing}"
            )

        if cfg.train.resume:
            cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
            if accelerator.is_main_process:
                print(
                    f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
                )
        else:
            init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
            if accelerator.is_main_process:
                print(
                    f"strictly load controlnet weight from pretrained SD\n"
                    f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                    f"weights initialized from scratch: {init_with_scratch}"
                )

        swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
        sd = torch.load(cfg.train.swinir_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        swinir.load_state_dict(sd, strict=True)
        for p in swinir.parameters():
            p.requires_grad = False
        if accelerator.is_main_process:
            print(f"load SwinIR from {cfg.train.swinir_path}")

        # set mode and cuda
        loaded_models['cldm'] = cldm.train().to(device)
        loaded_models['swinir'] = swinir.eval().to(device)
    
    elif cfg.exp_args.model_name == '':
        pass

    return loaded_models


def set_training_params(accelerator, models, cfg):

    train_params=[]

    for model_name, model in models.items():

        for name, param in model.named_parameters():

            if cfg.exp_args.finetuning_method == 'full_finetuning':
                param.requires_grad = True 
                train_params.append(param)

            elif cfg.exp_args.finetuning_method == 'only_ctrlnet':
                if 'controlnet' in name:
                    param.requires_grad = True
                    train_params.append(param)
                else: 
                    param.requires_grad = False

            elif cfg.exp_args.finetuning_method == 'only_unet':
                if 'unet' in name:
                    param.requires_grad = True
                    train_params.append(param)
                else: 
                    param.requires_grad = False
        
    return train_params



