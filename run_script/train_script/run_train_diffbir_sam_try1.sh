
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=2 accelerate launch train.py   --config configs/train/train_diffbir_sam_try1.yaml \
                                                    --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml
