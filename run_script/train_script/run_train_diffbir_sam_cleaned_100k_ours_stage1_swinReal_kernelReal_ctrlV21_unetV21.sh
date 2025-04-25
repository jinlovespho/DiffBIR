
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=0 accelerate launch train.py       --config configs/train/train_diffbir_sam_cleaned_100k_ours_stage1_swinReal_kernelReal_ctrlV21_unetV21.yaml \
                                                        --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml \
