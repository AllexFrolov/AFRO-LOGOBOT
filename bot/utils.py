import re
from typing import Tuple, Any, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from tqdm import notebook
import torch

lemmatizer = WordNetLemmatizer()


def text_to_pic(text: str) -> np.array:
    """
    Convert text to picture
    :param text: (str) text to convert
    the text string in the image
    :return: (numpy.array) array of shape (100, 200, 3)
    """
    # create canvas
    canvas = np.full((500, 500, 3), 0, dtype=np.uint8)
    # choice random color
    color = tuple(np.random.choice(255) for _ in range(3))
    # choice random font
    font = np.random.choice(8)
    fontScale = 3
    thickness = 3
    x, y = get_center(text, canvas, font, fontScale)

    canvas = cv2.putText(canvas, text, (x, y), font, fontScale, color, thickness)
    return canvas


def get_center(text: str, canvas: np.array, font: int, fontScale: int) -> Tuple[int, int]:
    """
    get center coordinate fot text
    :param text: (str)
    :param canvas: (np.array)
    :param font: (int) cv HersheyFonts
    :param fontScale: (int)
    :return: (Tuple[int, int]) center coordinate
    """
    text_size = cv2.getTextSize(text, font, fontScale, 2)[0]

    # get coords based on boundary
    text_x = (canvas.shape[1] - text_size[0]) // 2
    text_y = (canvas.shape[0] + text_size[1]) // 2
    return text_x, text_y


class Transformer(object):
    def __init__(self):
        """
        init albumentation augmentations
        """
        self.transform = A.Compose([
            A.OneOf([
                A.GridDistortion(7, 1., cv2.INTER_LINEAR, cv2.BORDER_REFLECT, p=.8),
                A.ElasticTransform(1., alpha_affine=50, interpolation=cv2.INTER_CUBIC,
                                   border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=.8)
            ], p=.5),
            A.OneOf([
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_REFLECT_101),
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_REPLICATE),
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_WRAP)
            ], p=0.5),
            A.OneOf([
                A.MotionBlur((15, 30)),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.Resize(128, 128, cv2.INTER_CUBIC)
        ])

    def __call__(self, img: Any) -> Any:
        """
        transform image
        :param img: (np.array or pil.Image)
        :return: (np.array or pil.Image)
        """
        return self.transform(image=img)['image']


def text_to_pic_transform(text: str) -> np.array:
    """
    Convert text to picture and added augmentations
    :param text: (str) Text
    :return: (np.array)
    """
    transformer = Transformer()
    return transformer(text_to_pic(text))


def add_text_to_img(text: str, icon_im: np.array) -> np.array:
    """
    add text image to icon image.
    :param text: (str) text
    :param icon_im: (np.array) icon image
    :return: (np.array) joined image
    """
    # transform text to image
    text_img = text_to_pic_transform(text)
    # get mask
    img2gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    mask = img2gray != 255
    # crop extra space
    height = mask.any(axis=1)
    width = mask.any(axis=0)
    ind = np.arange(len(mask))
    w = ind[width][[0, -1]]
    h = ind[height][[0, -1]]
    cut_text_img = text_img[h[0]:h[1], w[0]:w[1]]
    res_text = cv2.resize(cut_text_img, dsize=(128, 30), interpolation=cv2.INTER_AREA)

    join_img = icon_im.copy()
    join_img = cv2.resize(join_img, dsize=(128, 98), interpolation=cv2.INTER_AREA)
    if np.random.randint(2) == 1:
        join_img = np.concatenate((join_img, res_text / 255), axis=0)
    else:
        join_img = np.concatenate((res_text / 255, join_img), axis=0)
    return join_img


def lemmatize_and_clearing(text: str) -> str:
    """
    lemmatize text and save symbols only
    :param text: (str) text
    :return: (str) lemmatized text
    """
    clear_list = ' '.join(re.sub(r'\\n|\W|\d', ' ', text).split()).lower()
    lemm_list = lemmatizer.lemmatize(clear_list)
    return ''.join(lemm_list)


def tokenize(text: str, tokenizer: Any) -> str:
    """Splits a string into substrings of no more than 510 length and tokenizes
    :param text: (str) text
    :param tokenizer: (func) tokenizer
    :return: (str) tokenized text
    """
    if len(text) > 510:
        space_index = text.strip().rfind(' ', 0, 510)
        if space_index == -1:
            space_index = 510
        return tokenizer.encode(text[:space_index])[1:-1] + tokenize(text[space_index:], tokenizer)
    else:
        return tokenizer.encode(text)[1:-1]


def find_file(file_name: str) -> Optional[str]:
    """
    Read and return first line in file.
    :param file_name: (str) full path to fiile
    :return: ([str]) file
    """
    try:
        with open(file_name[:-3] + 'txt', 'r') as f:
            any_data = f.readline()
        return any_data
    except:
        return None


def embed_and_write_file(loader: Any, model: Any, device: torch.device, file_name: str):
    """
    convert vec to embedding and save to file
    :param loader: (Any) DataLoader
    :param model: (Any) Embed_model
    :param device: (torch.device)
    :param file_name: (str) path to save the file
    :return: NoneType
    """
    if device.type == 'cuda':
        from torch.cuda import LongTensor
    else:
        from torch import LongTensor

    model.eval()
    with notebook.tqdm(total=len(loader)) as progress_bar:
        for batch in loader:
            batch_mask = np.where(np.array(batch) != 0, 1, 0)
            batch_tensor = batch.to(device)
            batch_mask_tensor = LongTensor(batch_mask, device=device)
            with torch.no_grad():
                embed = model(batch_tensor, attention_mask=batch_mask_tensor).last_hidden_state
                embed_cpu = pd.DataFrame(embed.cpu().numpy()[:,0])
                embed_cpu.to_csv(file_name, index=False, header=None, mode='a')
            progress_bar.update()
