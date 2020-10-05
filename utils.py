from typing import Tuple, Any

import albumentations as A
import cv2
import numpy as np
from nltk.stem import WordNetLemmatizer

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


class Transformer:
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


def join_images(text_im: np.array, icon_im: np.array) -> np.array:
    brows, bcols = icon_im.shape[:2]
    rows, cols, channels = text_im.shape

    roi = icon_im[int(brows / 2) - int(rows / 2):int(brows / 2) + int(rows / 2),
          int(bcols / 2) - int(cols / 2):int(bcols / 2) + int(cols / 2)]

    img2gray = cv2.cvtColor(text_im, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # mask = cv2.adaptiveThreshold(img2gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    img2_fg = cv2.bitwise_and(text_im, text_im, mask=mask)

    dst = cv2.add(img1_bg, img2_fg.astype(float) / 255.)
    icon_im[int(brows / 2) - int(rows / 2):int(brows / 2) + int(rows / 2),
    int(bcols / 2) - int(cols / 2):int(bcols / 2) + int(cols / 2)] = dst

    return icon_im


def lemmatize_and_clearing(text: str) -> str:
    """
    lemmatize text and save symbols only
    :param text: (str) text
    :return: (str) lemmatized text
    """
    clear_list = ' '.join(re.sub(r'\\n|\W|\d', ' ', text).split()).lower()
    lemm_list = lemmatizer.lemmatize(clear_list)
    return ''.join(lemm_list)
