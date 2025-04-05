
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=1 accelerate launch train_baseline.py      --config configs/train/train_diffbir_sam_try1_baseline.yaml \
                                                                --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml
