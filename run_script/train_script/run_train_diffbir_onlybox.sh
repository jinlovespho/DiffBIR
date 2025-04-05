CUDA_VISIBLE_DEVICES=1 accelerate launch train.py   --config configs/train/train_diffbir_onlybox.yaml \
                                                    --config_testr testr/configs/TESTR/Pretrain/TESTR_R_50_Polygon.yaml
