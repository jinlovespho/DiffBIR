
# try1: finetune testr or testr_ctrlnet, using null prompt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train.py       --config configs/rebuttal_train/train_diffbir_sam_cleaned_100k_ours_stage3_swinReal_kernelReal_ctrlV21_unetV21.yaml \
                                                        --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml \
