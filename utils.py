from typing import Tuple, Any, Optional, List, Union

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import random


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


def get_center(text_size: Tuple[int, int], canvas: np.array, ) -> Tuple[int, int]:
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
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, value=border_color, ),
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, value=border_color, ),
                A.Rotate(10, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, value=border_color, )
            ], p=0.5),
            
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
                    angle: Optional[int] = None) -> np.array:
    """
    Added logo to picture
    :param logo: (uint8 array) Logo
    :param pic: (union[uint8 array, str]) array or path to image
    :param coord: (list) top left point to paste logo in pic
    :param angle: ([int]) Angle to rotate the logo
    :return: (np.array) joined image. (h, w, c) type uint8
    """
    logo_ = logo.copy()
    if angle is not None:
        logo_ = rotate_bound(logo_, angle)
    img2gray = logo_.mean(axis=-1)
    logo_mask = img2gray < 255
    h, w, _ = logo_.shape
    if isinstance(pic, str):
        joined_img = np.array(Image.open(pic))
    else:
        joined_img = pic.copy()
    joined_img[coord[0]:coord[0] + h,
    coord[1]:coord[1] + w][logo_mask] = logo_[logo_mask]
    return joined_img


def get_examples(logo: np.array) -> np.array:
    """
    return random choice meme and return logo into meme
    :param logo: (np.array)
    :return: (np.array (h, v, c) uint8)
    """
    examp_preset = {
        'man': {
            'pic': 'img/man.jpg',
            'coord': [205, 245],
            'angle': -4
        },
        'bad_guy': {
            'pic': 'img/bad_guy.jpg',
            'coord': [130, 205],
        },
        'svetlacov': {
        'pic': 'img/Svetlakov.jpg',
        'coord': [115, 220],
        }
    }
    exp = random.choice(list(examp_preset))
    return add_logo_to_pic(logo, **examp_preset[exp])

