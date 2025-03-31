from typing import List, Dict
import random
import math
import os

import numpy as np
from PIL import Image
import cv2
import polars as pl
import torch
from torch.nn import functional as F

from .diffjpeg import DiffJPEG

import json 
import string

valid_voc = list(string.printable[:-6])
invalid_voc=['□', '∫', 'æ', '⬏', 'Σ', '■', 'Å', 'Ḏ', '£', 'ń', '⌀', 'Ù', '│', 'Ⅶ', 'Â', 'ς', 'Ⅻ', '⁴', 'ъ', '∁', 'Æ', 'α', 'Ç', 'ˣ', '・', '⤤', 'Đ', 'ı', '≡', '⋄', 'Å', 'ᴴ', 'ᵗ', 'Ȃ', 'δ', 'Ì', 'Ρ', '⟷', 'ï', '«', 'ȯ', 'Ǒ', '⇩', 'ζ', '✰', '⁹', 'м', 'Ộ', '❘', '₄', '²', 'φ', '⌴', '⇨', 'ƌ', 'σ', 'Ⅸ', '∞', 'ţ', 'ů', '◁', '½', '¾', 'ᴾ', '�', 'ê', 'Ⅵ', 'ˢ', '°', 'ɮ', '⇪', 'ᵈ', 'Ė', 'Ǐ', '⊲', '·', 'û', '˅', '⊤', '↰', 'Ī', 'ȍ', '×', '⊝', '‟', '√', '➀', 'î', '↹', '➞', '↑', 'ü', '⋏', '℃', 'Û', 'Ȅ', '›', '⟶', '○', 'Ⓡ', 'Ȋ', '➜', 'ᴺ', 'å', '►', '˂', 'ι', 'ā', 'Ś', '∇', '•', '¥', '★', '⋅', 'ₖ', 'ũ', '⁼', 'İ', '∓', '⊂', '➯', '₅', 'Ồ', '»', 'Ž', 'ì', 'Ⅴ', '„', 'Ň', 'ú', '‑', 'Ä', '⊣', '˄', '˙', 'Ó', '±', '╳', 'ⁿ', 'ū', 'ş', 'л', 'Ṡ', 'ᴵ', 'Ȏ', 'ñ', 'λ', '✓', 'ø', '✞', '≤', 'Õ', '⎯', '⬌', 'ʳ', 'Š', '◉', '➨', 'ᶜ', 'ź', 'ġ', 'ÿ', '◦', 'ḻ', '➮', 'ᴸ', 'Ú', '─', '⇧', '⤶', 'ð', 'ë', 'Ξ', 'ȑ', '⇦', '↻', 'ă', 'Ě', 'Ω', 'Á', '₃', 'к', 'Ⅰ', '▬', '—', '∈', 'Ạ', '☐', '⁸', 'Ŕ', 'ù', 'â', 'п', 'ᴭ', '÷', '↲', '‘', 'Ȇ', 'ᵀ', '¿', 'Ț', '▎', 'ě', 'ⱽ', 'Λ', '∷', '△', 'ç', 'ǫ', 'Ầ', '➩', 'и', 'Ū', 'ý', '―', '⇵', 'Í', 'ꝋ', '↓', '©', '³', 'Ɔ', 'è', '🠈', 'ğ', 'Ⓐ', 'я', 'Φ', 'Ấ', 'ᵖ', '︽', '˚', 'œ', '∥', 'β', 'й', 'Ⓒ', '⬍', '∨', '℮', '¼', 'ć', '␣', 'Ã', '🡨', 'Ą', 'ǵ', '™', 'Ế', 'ᵐ', '◄', 'Ń', '✱', 'ô', '¢', '₁', 'Ⅱ', '¹', 'π', 'µ', 'Ĺ', '⍙', 'р', 'Ï', 'ε', '⟵', '∆', 'ы', '⧫', 'ã', 'ė', '⁰', '⬉', '−', '⬋', '◯', 'о', 'À', 'ρ', '☰', 'τ', 'ŗ', '⸬', 'Ö', 'é', 'ə', 'Ǫ', 'Ē', '⎵', '𝔀', 'ⓒ', 'ȏ', '“', 'Č', 'č', 'Î', '∙', 'ṣ', '\u200b', '✚', 'ō', '”', 'ö', 'ᴹ', '▢', 'ν', '⌣', '：', '︾', '﹘', 'а', '∖', '⌄', 'в', '︿', 'ᵃ', 'ớ', '↺', '▲', '▽', '…', 'Ë', '⌫', '⤷', '€', '⊘', 'Ŏ', '₂', '⤺', '⁵', 'Ȧ', '∧', 'ω', '卐', 'Ⅳ', '⁻', '↵', 'ĩ', 'Ⅲ', 'Ă', '⬸', 'ʃ', 'ȇ', '←', '⅓', '⮌', '⇥', 'η', '➦', 'Ô', '⬊', '℉', '⊥', 'á', 'ŉ', '⊚', '–', 'Ā', '∅', 'Ć', '∎', '⤸', '⦁', 'ē', 'ί', 'õ', 'ᴱ', 'υ', 'ß', '◡', 'È', '∣', 'Δ', 'ᴙ', 'ò', '⊢', 'κ', '☓', 'Ề', 'Θ', 'ä', '﹀', '☆', 'Ò', '˃', 'à', 'Ê', 'ʰ', 'Ğ', '’', '→', '®', '●', '⁺', 'Ţ', 'Ż', '̓', '▼', 'Ể', 'ᵒ', 'Ý', 'б', '➔', 'г', '∴', '⅔', '⬈', 'Ō', '∊', 'Π', 'Ⅷ', 'Ñ', '➝', 'É', 'Ł', 'ó', '∉', 'Ø', 'Ü', '⋮', 'ĺ', '≣', '∼', '↱', 'í', 'Ⅹ', 'ę', '⋯', 'с', '╎', '⤦', '⊼', 'ȧ', '∝', '⤻', 'ξ', 'š', '▾', 'γ', '¡', '⊳', 'д', '⁷', 'ж', '➧', 'ᴰ', '‧', '∘', 'ž', 'Ȯ', 'Ⅺ']

def decode(voc_idx):
    s = ''
    for idx in voc_idx:
        if idx<94:
            s += valid_voc[idx]
        else:
            return s
    return s

def encode(word):
    word = word.replace(' ','')
    s = []
    max_word_len = 25
    for i in range(max_word_len):
        if i < len(word):
            voc_idx = valid_voc.index(word[i])
            s.append(voc_idx)
        elif i == len(word):
            s.append(94)
        else:
            s.append(95)
    return s

def load_file_list(file_list_path: str, data_args=None):

    mode = data_args['mode']
    datasets = data_args['datasets']
    ann_path = data_args['ann_path']

    files = []
    for dataset in datasets:

        if dataset == 'TextOCR':

            # load json
            json_path = f'{mode}_dataset_modified_filtered.json'
            with open(f'{ann_path}/{json_path}', 'r') as f:
                json_data = json.load(f)

            # load annotations
            for img_path in os.listdir(f'{file_list_path}/{dataset}/{mode}'):
                gt_path = os.path.join(file_list_path, dataset, mode, img_path)
                
                img_name = img_path.split('/')[-1].split('.')[0]
                img_id = img_name.split('/')[-1].split('_')[0]
                crop_id = img_name.split('/')[-1].split('_')[-1].split('.')[0]
                ann = json_data[img_id][crop_id]
                
                boxes=[]
                texts=[]
                text_encs=[]
                prompts=[]
                for i in range(len(ann)):

                    # text preprocess
                    text=ann[i]['text']
                    txt_len = len(text)
                    
                    if len(text) == 1 and text == '.':
                        continue
                    count=0
                    for char in text:
                        if char in valid_voc:
                            count+=1

                    if count == txt_len:
                        texts.append(text)
                        text_encs.append(encode(text))

                        # box preprocess
                        box=ann[i]['bbox']
                        box = list(map(lambda x: int(x),box))
                        boxes.append(box)

                        # prompt preprocess
                        # prompt=f'A high-quality photo containing the word "{text}"'
                        prompt=""
                        prompts.append(prompt)
                
                assert len(boxes) == len(texts), f"Error: {len(boxes)} != {len(texts)}"

                # if the filetered image has no bbox and texts, skip it
                if len(boxes) == 0:
                    continue

                # # JLP - check if words are properly encoded and decoded
                # tmp1=[decode(word) for word in text_encs]
                # print(tmp1)
                # print(texts)
                # breakpoint()
                
                # # JLP - vis image
                # img=cv2.imread(gt_path)
                # cv2.imwrite('./tmp_img.jpg', img)
                # for i in range(len(boxes)):
                #     x,y,w,h = boxes[i]
                #     text = texts[i] 
                #     cv2.rectangle(img, (x,y), (x+w,y+h) , color= (0,255,0), thickness=2)
                #     cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                # cv2.imwrite('./tmp_img_bbox.jpg', img)

                files.append({"image_path": gt_path, 
                              "prompt": prompts, 
                              "text": texts, 
                              "bbox": boxes,
                              'text_enc': text_encs, 
                              "img_name": img_name})

    if mode=='val':
        files = random.sample(files, 24)
    return files


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        # img: torch.Tensor
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img
