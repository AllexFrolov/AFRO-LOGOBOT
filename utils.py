import re
from typing import Tuple, Any, Optional, List, Union

import albumentations as A
import cv2
from PIL import Image
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
    # choice random color
    color = tuple(np.random.choice(255) for _ in range(3))
    # choice random font
    font = 3
    fontScale = 3
    thickness = 3
    text_size = cv2.getTextSize(text, font, fontScale, 2)[0]
    canvas = np.full((text_size[1] + 1024, text_size[0] + 1024, 3), 255, dtype=np.uint8)

    x, y = get_center(text_size, canvas)
    canvas = cv2.putText(canvas, text, (x, y), font, fontScale, color, thickness)
    return canvas


def get_center(text_size: Tuple[int, int], canvas: np.array,) -> Tuple[int, int]:
    """
    get center coordinate fot text
    :param text_size: (Tuplt[int, int])
    :param canvas: (np.array)
    :return: (Tuple[int, int]) center coordinate
    """
    # get coords based on boundary
    text_x = (canvas.shape[1] - text_size[0]) // 2
    text_y = (canvas.shape[0] + text_size[1]) // 2
    return text_x, text_y


class Transformer(object):
    def __init__(self):
        """
        init albumentation augmentations
        """
        border_color = (255, 255, 255)
        self.transform = A.Compose([
            A.OneOf([
                A.GridDistortion(7, 1., cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, value=border_color, p=.8),
                A.ElasticTransform(1., alpha_affine=50, interpolation=cv2.INTER_CUBIC,
                                   border_mode=cv2.BORDER_CONSTANT, value=border_color, p=.8)
            ], p=1.),
            A.OneOf([
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, value=border_color,),
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, value=border_color,),
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, value=border_color,)
            ], p=0.5),
            A.OneOf([
                A.MotionBlur((15, 30)),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ], p=0.5),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # A.Resize(128, 128, cv2.INTER_CUBIC)
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
    :return: (np.array) joined image. (h, w, c) type uint8
    """
    # transform text to image
    text_img = text_to_pic_transform(text)
    # get mask
    img2gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    mask = img2gray != 255
    h_size, w_size = mask.shape
    # crop extra space
    height_mask = mask.any(axis=1)
    width_mask = mask.any(axis=0)
    ind_h = np.arange(h_size)
    ind_w = np.arange(w_size)
    w = ind_w[width_mask][[0, -1]]
    h = ind_h[height_mask][[0, -1]]
    cut_text_img = text_img[h[0]:h[1], w[0]:w[1]]
    res_text = cv2.resize(cut_text_img, dsize=(128, 30), interpolation=cv2.INTER_AREA)

    img2gray = cv2.cvtColor(icon_im, cv2.COLOR_BGR2GRAY)
    mask_icon = img2gray < 240
    canvas = np.full(icon_im.shape, 255, dtype=np.uint8)
    canvas[mask_icon] = icon_im[mask_icon]
    join_img = canvas.copy()
    if join_img.shape[1] != 128:
        join_img = cv2.resize(join_img, dsize=(128, 128), interpolation=cv2.INTER_AREA)

    res_text_gray = cv2.cvtColor(res_text, cv2.COLOR_BGR2GRAY)
    res_mask = res_text_gray != 255
    if np.random.randint(2) == 1:
        join_img[-30:][res_mask] += res_text[res_mask]
    else:
        join_img[:30][res_mask] += res_text[res_mask]
    return join_img


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))


def add_logo_to_pic(logo: np.array, pic: Union[np.array, str], coord: List[int],
                    angle: Optional[int]=None) -> np.array:
    """
    Added logo to picture
    :param logo: (uint8 array) Logo
    :param pic: (union[uint8 array, str]) array or path to image
    :param coord: (list) top left point to paste logo in pic
    :param angle: ([int]) Angle to rotate the logo
    :return: (np.array) joined image. (h, w, c) type uint8
    """
    logo_ = logo.copy()
    h, w, _ = logo.shape
    if angle is not None:
        logo_ = rotate_bound(logo_, -10)
    img2gray = logo_.mean(axis=-1)
    mask = img2gray < 255

    if isinstance(pic, str):
        joined_img = np.array(Image.open(pic))
    else:
        joined_img = pic.copy()
    joined_img[coord[0]:coord[0] + h,
    coord[1]:coord[1] + w][mask] = logo_[mask]
    return joined_img


def get_examples(logo: np.array, examples: Union[List[str], str]) -> np.array:
    """

    :param logo: (np.array)
    :param examples: (Union[List[str], str])
    :return:
    """
    examp_preset = {
        'man': {
                'pic': 'img/man.jpg',
                'coord': [195, 235],
                'angle': -10
                }
    }

    return add_logo_to_pic(logo, **examp_preset[examples])





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
