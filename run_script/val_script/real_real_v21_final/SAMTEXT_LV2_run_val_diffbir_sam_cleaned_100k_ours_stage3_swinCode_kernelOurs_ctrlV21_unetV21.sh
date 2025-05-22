
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=0 accelerate launch val.py         --config configs/val/real_real_v21_final/SAMTEXT_LV2_val_diffbir_sam_cleaned_100k_ours_stage3_swinCode_kernelOurs_ctrlV21_unetV21.yaml \
                                                        --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml \
