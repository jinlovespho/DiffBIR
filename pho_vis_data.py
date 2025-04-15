import os 
import json 
import numpy as np 
import cv2 


with open('./combined_full_dataset.json', 'r') as f:
    full_data = json.load(f)        # 63558

with open('./combined_restoration_dataset.json', 'r') as f:
    clean_data = json.load(f)             # 17233


# VIS CLEAN DATA 
vocabs=[]
for img_name, img_ann in clean_data.items():
    img_path = f'./images/{img_name}.jpg'
    img = cv2.imread(img_path)                      # 512 512 3
    img_ann = img_ann['0']['text_instances']

    img_box = img.copy()
    img_poly = img.copy()
    for ann in img_ann:
        box = ann['bbox']
        x1,y1,x2,y2 = box 
        poly = ann['polygon']
        text = ann['text']
        vocabs.append(text)

    #     cv2.rectangle(img_box, (x1,y1), (x2,y2), (0, 255, 0), 2)
    #     cv2.putText(img_box, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #     cv2.polylines(img_poly, [np.array(poly).astype(np.int32)], True, (0, 255, 0), 2)
    #     cv2.putText(img_poly, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # cv2.imwrite(f'./vis/clean/{img_name}_box.jpg', img_box)
    # cv2.imwrite(f'./vis/clean/{img_name}_poly.jpg', img_poly)


# VIS VOCAB
with open('./clean_vocabs.txt', 'w') as f:
    f.write('\n'.join(vocabs))

breakpoint()