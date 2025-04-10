
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=2 accelerate launch train_only_unet_jaewon_check.py        --config configs/train/train_diffbir_sam_only_unet_jaewon_check.yaml \
                                                                                --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml
