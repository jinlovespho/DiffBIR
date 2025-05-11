
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=0 accelerate launch val.py         --config configs/val/SAMTEXT_LV1_val_diffbir_sam_cleaned_100k_ours_stage3_swinReal_kernelReal_ctrlV21_unetV21.yaml \
                                                        --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml \
