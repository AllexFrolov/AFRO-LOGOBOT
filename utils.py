from typing import Tuple, Any

import albumentations as A
import cv2
import numpy as np


def text_to_pic(text: str) -> np.array:
    """
    Convert text to picture
    :param text: (str) text to convert
    the text string in the image
    :return: (numpy.array) array of shape (100, 200, 3)
    """
    # create canvas
    canvas = np.full((500, 500, 3), 255, dtype=np.uint8)
    # choice random color
    color = tuple(np.random.choice(255) for _ in range(3))
    # choice random font
    font = np.random.choice(8)
    fontScale = 2
    thickness = 2
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
                                   border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=.8)
            ]),
            A.OneOf([
                A.Rotate(30, cv2.INTER_NEAREST, cv2.BORDER_REFLECT_101),
                A.Rotate(30, cv2.INTER_NEAREST, cv2.BORDER_REPLICATE),
                A.Rotate(30, cv2.INTER_NEAREST, cv2.BORDER_WRAP)
            ], p=1.),
            A.OneOf([
                A.MotionBlur((15, 30)),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        ])

    def __call__(self, img: Any) -> Any:
        """
        transform image
        :param img: (np.array or pil.Image)
        :return: (np.array or pil.Image)
        """
        return self.transform(image=img)['image']


transformer = Transformer()


def convert_to_pic(text: str) -> np.array:
    """
    Convert text to picture and added augmentations
    :param text: (str) Text
    :return: (np.array)
    """
    return transformer(text_to_pic(text))
