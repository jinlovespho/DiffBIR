CUDA_VISIBLE_DEVICES=3 accelerate launch train.py   --config configs/train/train_diffbir_sam_try1.yaml \
                                                    --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml
