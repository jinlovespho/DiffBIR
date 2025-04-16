import os 
import sys
import wandb 
import argparse
from omegaconf import OmegaConf
from diffbir.model import ControlLDM, SwinIR, Diffusion
from diffbir.utils.common import instantiate_from_config, to, log_txt_as_img
from diffbir.dataset.pho_codeformer import collate_fn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch 



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



def load_model(accelerator, device, args, cfg):

    loaded_models={}

    # default: load cldm, swinir
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    # sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    # unused, missing = cldm.load_pretrained_sd(sd)
    # if accelerator.is_main_process:
    #     print(
    #         f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
    #         f"unused weights: {unused}\n"
    #         f"missing weights: {missing}"
    #     )

    # if cfg.train.resume:
    #     cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
    #     if accelerator.is_main_process:
    #         print(
    #             f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
    #         )
    # else:
    #     init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
    #     if accelerator.is_main_process:
    #         print(
    #             f"strictly load controlnet weight from pretrained SD\n"
    #             f"weights initialized with newly added zeros: {init_with_new_zero}\n"
    #             f"weights initialized from scratch: {init_with_scratch}"
    #         )


    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    # sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    # if "state_dict" in sd:
    #     sd = sd["state_dict"]
    # sd = {
    #     (k[len("module.") :] if k.startswith("module.") else k): v
    #     for k, v in sd.items()
    # }
    # swinir.load_state_dict(sd, strict=True)
    # for p in swinir.parameters():
    #     p.requires_grad = False
    # if accelerator.is_main_process:
    #     print(f"load SwinIR from {cfg.train.swinir_path}")

    # set mode and cuda
    loaded_models['cldm'] = cldm.train().to(device)
    loaded_models['swinir'] = swinir.eval().to(device)
    

    # ------------------------ ADD MODELS -------------------------------

    # training ocr detection with diffbir features
    if cfg.exp_args.model_name == 'diffbir_onlybox' or cfg.exp_args.model_name == 'diffbir_testr':
        sys.path.append(f'{os.getcwd()}/testr')
        from testr.adet.modeling.transformer_detector import TransformerDetector
        from testr.adet.config import get_cfg

        # get testr config
        config_testr = get_cfg()
        config_testr.merge_from_file(args.config_testr)
        config_testr.freeze()

        # load testr model
        detector = TransformerDetector(config_testr)

        # load testr pretrained weights
        if cfg.exp_args.testr_ckpt_dir is not None:
            ckpt = torch.load(cfg.exp_args.testr_ckpt_dir, map_location="cpu")
            load_result = detector.load_state_dict(ckpt['model'], strict=False)
            
            if accelerator.is_main_process:
                print("Loaded TESTR checkpoint keys:")
                print(" - Missing keys:", load_result.missing_keys)
                # print(" - Unexpected keys:", load_result.unexpected_keys)


        loaded_models['testr'] = detector.train().to(device)
    
    # add other models
    elif cfg.exp_args.model_name == '':
        pass



    # -------------------------------- RESUME TRAINING ---------------------------------------
    if cfg.exp_args['resume_ckpt_dir'] is not None:

        # set ckpt path
        ckpt_dir = f"{cfg.exp_args['resume_ckpt_dir']}"
        # ckpts = sorted(os.listdir(ckpt_dir))
        # ckpt_path = f"{ckpt_dir}/{ckpts[-1]}"        
        ckpt=torch.load(ckpt_dir, map_location="cpu")

        # Efficient weight loading with missing key handling
        for model_name, model in loaded_models.items():
            if model_name in ckpt:
                missing, unexpected = model.load_state_dict(ckpt[model_name], strict=False)
                print(f"Loaded {model_name} | Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
            else:
                print(f"⚠️ Warning: No checkpoint found for {model_name}")

        # Move models to the correct device (if needed)
        for model in loaded_models.values():
            model.to(device)
        
        return loaded_models, ckpt_dir


    return loaded_models, None


def set_training_params(accelerator, models, cfg):

    train_params=[]
    all_model_names=[]
    train_model_names=[]

    for model_name, model in models.items():

        for name, param in model.named_parameters():
            all_model_names.append(name)


            if cfg.exp_args.finetuning_method == 'full_finetuning':
                param.requires_grad = True 
                train_model_names.append(name)
                train_params.append(param)


            elif cfg.exp_args.finetuning_method == 'ctrlnet':
                if 'controlnet' in name:
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else: 
                    param.requires_grad = False


            elif cfg.exp_args.finetuning_method == 'unet':
                if 'unet' in name:
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else: 
                    param.requires_grad = False
            
            
            elif cfg.exp_args.finetuning_method == 'testr_detector':
                if 'testr.transformer.encoder' in name or 'testr.transformer.level_embed' in name:
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False


            elif cfg.exp_args.finetuning_method == 'ctrlnet_and_testr_detector':
                if 'controlnet' in name or 'testr.transformer.encoder' in name or 'testr.transformer.level_embed' in name:
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False
            
            # train all components of testr
            elif cfg.exp_args.finetuning_method == 'testr':
                if 'testr' in name:
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False

            # train ctrlnet and all components of testr
            elif cfg.exp_args.finetuning_method == 'ctrlnet_and_testr':
                if 'testr' in name or 'controlnet' in name:
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False
                    
            elif cfg.exp_args.finetuning_method == 'ctrlnet_and_unetAttn':
                if 'controlnet' in name or ('unet' in name and 'attn' in name):
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False
            
            elif cfg.exp_args.finetuning_method == 'ctrlnet_and_unetAttn_and_testr':
                if 'controlnet' in name or ('unet' in name and 'attn' in name) or ('testr' in name):
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False



    # print modules to be trained
    if accelerator.is_main_process:
        print('================================================================= MODELS TO BE TRAINED =================================================================')
        chunk_size = 10  # Adjust based on readability
        for i in range(0, len(train_model_names), chunk_size):
            print(train_model_names[i:i+chunk_size])  # Print in smaller chunks
    
    
    return train_params, train_model_names



