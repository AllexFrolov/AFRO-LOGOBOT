import numpy as np

# !pip install tensorflow==2.2.0
# !pip install ISR
from ISR.models import RDN
from PIL import Image, ImageFilter

if __name__ == '__main__':
    print('Hallo, Afro super resolutor is here!')

def superresolute(lr_img, scale=2):
    """
    Takes an image with low resolution and increases it
    Input: PIL Image, Output: PIL Image
    """
    assert type(lr_img) == np.ndarray
    rdn = RDN(weights='psnr-small')
    for _ in range(scale):
      lr_img = rdn.predict(lr_img, by_patch_of_size=50)

    return lr_img

def imgfilter(img, radius=3):
    """
    Takes the image and removes the noise with little affect to resolution
    Input: PIL Image, Output: PIL Image
    """
    assert type(img) == np.ndarray
    img = Image.fromarray(img)
    lol = img.filter(ImageFilter.GaussianBlur(radius=radius))
    lol2 = lol.filter(ImageFilter.EDGE_ENHANCE_MORE)
    final_img = np.array(lol2)
    return final_img

