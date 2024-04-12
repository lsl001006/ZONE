from PIL import Image
import numpy as np
import os
from tqdm import tqdm


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def load_512_batch(image_paths):
    for img_path in tqdm(image_paths):
        image = load_512(img_path)
        basename = os.path.basename(img_path)
        image = Image.fromarray(image)
        image.save(f'/path/{basename}')

if __name__ == "__main__":

    path = ""
    paths = [os.path.join(path, im) for im in os.listdir(path)]
    load_512_batch(paths)