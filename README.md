<table align="center">
  <tr>
    <td>
      <img src="assets/diffbir_logo.png" width="300" style="border-radius: 8px; margin-right: 20px;">
    </td>
    <td>
      <h1 style="margin: 0;">Text-Aware Image Restoration with Diffusion Models</h1>
      <p>
        <a href="https://arxiv.org/abs/2506.09993"><img src="https://img.shields.io/badge/arXiv-2506.09993-B31B1B"></a>
        <a href="https://cvlab-kaist.github.io/TAIR/"><img src="https://img.shields.io/badge/Project%20Page-online-1E90FF"></a>
        <a href="https://huggingface.co/datasets/Min-Jaewon/SA-Text"><img src="https://img.shields.io/badge/HuggingFace-SA--Text-yellow?logo=huggingface&logoColor=yellow"></a>
        <a href="https://huggingface.co/datasets/Min-Jaewon/Real-Text"><img src="https://img.shields.io/badge/HuggingFace-Real--Text-yellow?logo=huggingface&logoColor=yellow"></a>
      </p>
    </td>
  </tr>
</table>

<div align="left">

[Jaewon&nbsp;Min<sup>1*</sup>](https://github.com/Min-Jaewon/) · 
[Jin&nbsp;Hyeon&nbsp;Kim<sup>2*</sup>](https://github.com/jinlovespho) · 
Paul&nbsp;Hyunbin&nbsp;Cho<sup>1</sup> · 
[Jaeeun&nbsp;Lee<sup>3</sup>](https://github.com/babywhale03) · 
Jihye&nbsp;Park<sup>4</sup> · 
Minkyu&nbsp;Park<sup>4</sup> · 
Sangpil&nbsp;Kim<sup>2†</sup> · 
Hyunhee&nbsp;Park<sup>4†</sup> · 
[Seungryong&nbsp;Kim<sup>1†</sup>](https://cvlab.kaist.ac.kr/)

<sup>*</sup> Equal contribution
<sup>1</sup> KAIST&nbsp;AI ·
<sup>2</sup> Korea&nbsp;University ·
<sup>3</sup> Yonsei&nbsp;University ·
<sup>4</sup> Samsung&nbsp;Electronics

<!-- <p align="center">
    <img src="assets/terediff_teaser.png" width="1000">
</p> -->

<!-- <sub><sup>*</sup> Equal&nbsp;contribution  <sup>†</sup> Corresponding&nbsp;authors</sub> -->

<!-- ### [Paper&nbsp;(Coming&nbsp;soon)](#) | [Project&nbsp;Page](https://cvlab-kaist.github.io/TAIR) -->

</div>

## 📢 News
- 🤗 **2025.06.19** — **SA-Text** and **Real-Text** datasets are released along with the [dataset pipeline](https://github.com/paulcho98/text_restoration_dataset/tree/main)!
- 📄 **2025.06.12** — Arxiv paper is released! 
- 🚀 **2025.06.01** — Official launch of the repository and project page!


## 💾 SA-Text Dataset
**SA-Text** is a newly proposed dataset for **Text-Aware Image Restoration (TAIR)** task. It is built from  **SA-1B** dataset using our [dataset pipeline](https://github.com/paulcho98/text_restoration_dataset/tree/main) and  consists of **100K** image-text instance pairs with detailed scene-level annotations.

**Real-Text** is an evaluation dataset for real-world scenarios. It is constructed from [RealSR](https://github.com/csjcai/RealSR) and [DrealSR](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution) using same pipeline as above.

---

### Dataset Download

| Split             | Hugging Face 🤗 | Google Drive 📁 |
|------------------|:---------------:|:---------------:|
| **SA-Text**       | <div align="center">[Link](https://huggingface.co/datasets/Min-Jaewon/SA-Text)</div> | <div align="center">[Link](https://drive.google.com/file/d/1wnGBwrRNJ-hegPtvt8s4y-iXgdED16L4/view?usp=sharing)</div> |
| **Real-Text**     | <div align="center">[Link](https://huggingface.co/datasets/Min-Jaewon/Real-Text)</div> | <div align="center">[Link](https://drive.google.com/file/d/1sIjeFe0Rq6IvYEC-pkz6aQ4ubuIge4xi/view?usp=sharing)</div> |


### Structure of dataset from 'Google Drive'
```
SA-Text/
├── images/                        # 100K hiqh-quality scene images with text instances
└── restoration_dataset.json       # Annotations

Real-Text/
├── HQ/                            # High-quality images
├── LQ/                            # Low-quality degraded inputs
└── real_benchmark_dataset.json    # Annotations
```
---

### Notes

- Each image is paired with one or more text instances with polygon-level annotations.
- The dataset follows a consistent annotation format, detailed in the [dataset pipeline](https://github.com/paulcho98/text_restoration_dataset/tree/main).
- We recommend using the dataset from Google Drive for testing our code.



## ⚒️ Dependency and Environment Setup

#### 1. Clone repo
```
git clone https://github.com/jinlovespho/DiffBIR.git -b pho
cd DiffBIR 
```

#### 2. Setup conda environment
```
conda create -n terediff python=3.10 -y
conda activate terediff
```

#### 3. Install libraries
```
# torch installation
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# other requirements
pip install -r requirements.txt

# install detectron2 if detectron2 is not in the folder
# python -m pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
cd detectron2 
pip install -e .

# install testr
cd testr 
pip install -e .
```

#### 4. Download pretrained weights
```
bash download_weights.sh
```

#### 5. Download dataset


## 🔧 Training 
Our text-aware restoration model, **TeReDiff**, comprises two main modules: an image restoration module and a text spotting module. 
Training is conducted in three stages:
- **Stage 1**: Train only the image restoration module.
- **Stage 2**: Train only the text spotting module.
- **Stage 3**: Jointly train both modules.
---
<!-- The training configuration file is located in **configs/train**.  -->
### Stage1 Training Script
Stage1 training configuration file can be found [here](configs/train/train_diffbir_sam_cleaned_100k_ours_stage1_swinReal_kernelReal_ctrlV21_unetV21.yaml)

```
bash run_script/train_script/run_train_diffbir_sam_cleaned_100k_ours_stage1_swinReal_kernelReal_ctrlV21_unetV21.sh
```

### Stage2 Training Script
Stage2 training configuration file can be found [here](configs/train/train_diffbir_sam_cleaned_100k_ours_stage2_swinReal_kernelReal_ctrlV21_unetV21.yaml)

```
bash run_script/train_script/run_train_diffbir_sam_cleaned_100k_ours_stage2_swinReal_kernelReal_ctrlV21_unetV21.sh
```

### Stage3 Training Script
Stage3 training configuration file can be found [here](configs/train/train_diffbir_sam_cleaned_100k_ours_stage3_swinReal_kernelReal_ctrlV21_unetV21.yaml)

```
bash run_script/train_script/run_train_diffbir_sam_cleaned_100k_ours_stage3_swinReal_kernelReal_ctrlV21_unetV21.sh
```


## 🚀 Inference 

---
### Evaluation on SA-Text (Lv1) 
```

```

### Evaluation on SA-Text (Lv2)
```

```

### Evaluation on SA-Text (Lv3)
```

```

### Evaluation on Real-Text
```

```


## Citation

If you find our work useful for your research, please consider citing it :)

```
@article{min2025text,
  title={Text-Aware Image Restoration with Diffusion Models},
  author={Min, Jaewon and Kim, Jin Hyeon and Cho, Paul Hyunbin and Lee, Jaeeun and Park, Jihye and Park, Minkyu and Kim, Sangpil and Park, Hyunhee and Kim, Seungryong},
  journal={arXiv preprint arXiv:2506.09993},
  year={2025}
}
```
