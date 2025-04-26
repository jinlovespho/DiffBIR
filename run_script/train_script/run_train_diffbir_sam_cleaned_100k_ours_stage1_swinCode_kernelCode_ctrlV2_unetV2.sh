
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py       --config configs/train/train_diffbir_sam_cleaned_100k_ours_stage1_swinCode_kernelCode_ctrlV2_unetV2.yaml \
                                                        --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml
