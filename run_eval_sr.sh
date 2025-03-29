# DiffBIR v2 (ECCV paper version)
CUDA_VISIBLE_DEVICES=2 python -u inference.py \
    --task sr \
    --upscale 4 \
    --version v2 \
    --sampler spaced \
    --steps 50 \
    --captioner none \
    --pos_prompt '' \
    --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
    --cfg_scale 4 \
    --input inputs/demo/bsr \
    --output results/v2_demo_bsr \
    --device cuda --precision fp32

# # DiffBIR v2.1
# python -u inference.py \
# --task sr \
# --upscale 4 \
# --version v2.1 \
# --captioner llava \
# --cfg_scale 8 \
# --noise_aug 0 \
# --input inputs/demo/bsr \
# --output results/v2.1_demo_bsr