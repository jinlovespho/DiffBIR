import os
import json 
import string

char_table = [chr(i) for i in range(32,127)]

valid_voc = list(string.printable[:-6])
invalid_voc=['□', '∫', 'æ', '⬏', 'Σ', '■', 'Å', 'Ḏ', '£', 'ń', '⌀', 'Ù', '│', 'Ⅶ', 'Â', 'ς', 'Ⅻ', '⁴', 'ъ', '∁', 'Æ', 'α', 'Ç', 'ˣ', '・', '⤤', 'Đ', 'ı', '≡', '⋄', 'Å', 'ᴴ', 'ᵗ', 'Ȃ', 'δ', 'Ì', 'Ρ', '⟷', 'ï', '«', 'ȯ', 'Ǒ', '⇩', 'ζ', '✰', '⁹', 'м', 'Ộ', '❘', '₄', '²', 'φ', '⌴', '⇨', 'ƌ', 'σ', 'Ⅸ', '∞', 'ţ', 'ů', '◁', '½', '¾', 'ᴾ', '�', 'ê', 'Ⅵ', 'ˢ', '°', 'ɮ', '⇪', 'ᵈ', 'Ė', 'Ǐ', '⊲', '·', 'û', '˅', '⊤', '↰', 'Ī', 'ȍ', '×', '⊝', '‟', '√', '➀', 'î', '↹', '➞', '↑', 'ü', '⋏', '℃', 'Û', 'Ȅ', '›', '⟶', '○', 'Ⓡ', 'Ȋ', '➜', 'ᴺ', 'å', '►', '˂', 'ι', 'ā', 'Ś', '∇', '•', '¥', '★', '⋅', 'ₖ', 'ũ', '⁼', 'İ', '∓', '⊂', '➯', '₅', 'Ồ', '»', 'Ž', 'ì', 'Ⅴ', '„', 'Ň', 'ú', '‑', 'Ä', '⊣', '˄', '˙', 'Ó', '±', '╳', 'ⁿ', 'ū', 'ş', 'л', 'Ṡ', 'ᴵ', 'Ȏ', 'ñ', 'λ', '✓', 'ø', '✞', '≤', 'Õ', '⎯', '⬌', 'ʳ', 'Š', '◉', '➨', 'ᶜ', 'ź', 'ġ', 'ÿ', '◦', 'ḻ', '➮', 'ᴸ', 'Ú', '─', '⇧', '⤶', 'ð', 'ë', 'Ξ', 'ȑ', '⇦', '↻', 'ă', 'Ě', 'Ω', 'Á', '₃', 'к', 'Ⅰ', '▬', '—', '∈', 'Ạ', '☐', '⁸', 'Ŕ', 'ù', 'â', 'п', 'ᴭ', '÷', '↲', '‘', 'Ȇ', 'ᵀ', '¿', 'Ț', '▎', 'ě', 'ⱽ', 'Λ', '∷', '△', 'ç', 'ǫ', 'Ầ', '➩', 'и', 'Ū', 'ý', '―', '⇵', 'Í', 'ꝋ', '↓', '©', '³', 'Ɔ', 'è', '🠈', 'ğ', 'Ⓐ', 'я', 'Φ', 'Ấ', 'ᵖ', '︽', '˚', 'œ', '∥', 'β', 'й', 'Ⓒ', '⬍', '∨', '℮', '¼', 'ć', '␣', 'Ã', '🡨', 'Ą', 'ǵ', '™', 'Ế', 'ᵐ', '◄', 'Ń', '✱', 'ô', '¢', '₁', 'Ⅱ', '¹', 'π', 'µ', 'Ĺ', '⍙', 'р', 'Ï', 'ε', '⟵', '∆', 'ы', '⧫', 'ã', 'ė', '⁰', '⬉', '−', '⬋', '◯', 'о', 'À', 'ρ', '☰', 'τ', 'ŗ', '⸬', 'Ö', 'é', 'ə', 'Ǫ', 'Ē', '⎵', '𝔀', 'ⓒ', 'ȏ', '“', 'Č', 'č', 'Î', '∙', 'ṣ', '\u200b', '✚', 'ō', '”', 'ö', 'ᴹ', '▢', 'ν', '⌣', '：', '︾', '﹘', 'а', '∖', '⌄', 'в', '︿', 'ᵃ', 'ớ', '↺', '▲', '▽', '…', 'Ë', '⌫', '⤷', '€', '⊘', 'Ŏ', '₂', '⤺', '⁵', 'Ȧ', '∧', 'ω', '卐', 'Ⅳ', '⁻', '↵', 'ĩ', 'Ⅲ', 'Ă', '⬸', 'ʃ', 'ȇ', '←', '⅓', '⮌', '⇥', 'η', '➦', 'Ô', '⬊', '℉', '⊥', 'á', 'ŉ', '⊚', '–', 'Ā', '∅', 'Ć', '∎', '⤸', '⦁', 'ē', 'ί', 'õ', 'ᴱ', 'υ', 'ß', '◡', 'È', '∣', 'Δ', 'ᴙ', 'ò', '⊢', 'κ', '☓', 'Ề', 'Θ', 'ä', '﹀', '☆', 'Ò', '˃', 'à', 'Ê', 'ʰ', 'Ğ', '’', '→', '®', '●', '⁺', 'Ţ', 'Ż', '̓', '▼', 'Ể', 'ᵒ', 'Ý', 'б', '➔', 'г', '∴', '⅔', '⬈', 'Ō', '∊', 'Π', 'Ⅷ', 'Ñ', '➝', 'É', 'Ł', 'ó', '∉', 'Ø', 'Ü', '⋮', 'ĺ', '≣', '∼', '↱', 'í', 'Ⅹ', 'ę', '⋯', 'с', '╎', '⤦', '⊼', 'ȧ', '∝', '⤻', 'ξ', 'š', '▾', 'γ', '¡', '⊳', 'д', '⁷', 'ж', '➧', 'ᴰ', '‧', '∘', 'ž', 'Ȯ', 'Ⅺ']
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']


def decode(idxs):
    s = ''
    for idx in idxs:
        if idx < len(CTLABELS):
            s += CTLABELS[idx]
        else:
            return s
    return s


def encode(word):
    s = []
    max_word_len = 25
    for i in range(max_word_len):
        if i < len(word):
            char=word[i]
            idx = CTLABELS.index(char)
            s.append(idx)
        else:
            s.append(96)
    return s


dataset = 'SAM'
ann_path = '/media/dataset1/jinlovespho/NIPS2025/data/generated_data/ocr'
file_list_path = '/media/dataset1/jinlovespho/NIPS2025/data/generated_data'

json_path = f'transformed_dataset.json'
with open(f'{ann_path}/{json_path}', 'r') as f:
    json_data = json.load(f)

texts=[]
text_encs=[]
imgs = sorted(os.listdir(f'{file_list_path}/{dataset}/clean_crops'))
for img in imgs:
    gt_path = f"{file_list_path}/{dataset}/clean_crops/{img}"

    # (example) img: sa_8558_crop_1_crop_0.jpg
    img_id = '_'.join(img.split('_')[:4])   # sa_8558_crop_1 까지가 id
    img_ann = json_data[img_id]

    model_H, model_W = (512,512)

    # boxes=[]
    # texts=[]
    # text_encs=[]
    # prompts=[]
    # polys=[]

    # JLP vis
    # img0 = cv2.imread(gt_path)  # 512 512 3
    
    for crop_id, crop_ann in img_ann.items():

        for ann in crop_ann:
                
            # process text
            text = ann['text']

            count=0
            for char in text:
                # OCR english: range(32,127)
                if 32 <= ord(char) and ord(char) < 127:
                    count+=1
                    # print(char, ord(char))
            if count == len(text) and count < 26:
                texts.append(text)
                text_encs.append(encode(text))
                try:
                    assert text == decode(encode(text)), 'check text encoding !'
                except:
                    breakpoint()
            # txt_len = len(text)
            # if len(text) == 1 and text == '.':
            #     continue
            # count=0
            # for char in text:
            #     if char in valid_voc:
            #         count+=1
            # if count == txt_len:
            #     texts.append(text)
            #     text_encs.append(encode(text))

breakpoint()