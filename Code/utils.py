import numpy as np
from PIL import Image
import tensorflow as tf
from skimage import io,data,color
import math

def load_images(file):
    im = Image.open(file)
    img = np.array(im, dtype="float32") / 255.0
    img_max = np.max(img)
    img_min = np.min(img)
    img_norm = np.float32((img - img_min) / np.maximum((img_max - img_min), 0.001))
    return img_norm
    
def save_images(filepath, result_1, result_2 = None, result_3 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    result_3 = np.squeeze(result_3)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)
    if not result_3.any():
        cat_image = cat_image
    else:
        cat_image = np.concatenate([cat_image, result_3], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')



