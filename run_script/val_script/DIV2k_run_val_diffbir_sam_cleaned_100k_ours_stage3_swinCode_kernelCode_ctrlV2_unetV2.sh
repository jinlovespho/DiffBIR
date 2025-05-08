
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=4 accelerate launch val.py         --config configs/val/DIV2k_val_diffbir_sam_cleaned_100k_ours_stage3_swinCode_kernelCode_ctrlV2_unetV2.yaml \
                                                        --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml \
