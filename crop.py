import os

from PIL import Image
import numpy as np


def recurse(l, img, path, id, List):
    # if img[:target-3,target+3] ==
    # print(img[:,target-3:target+3])
    r = l + 350
    index = path.find(".")
    save_path = "./data/cropped_image/" + path[:index] + f"_{id}" + path[index:]

    if r > img.shape[1] > l:
        crop_img = Image.fromarray(img[:, l:img.shape[1]])
        crop_img.save(save_path)
        List.append(save_path)
        return

    for i in range(r, -1, -1):
        part = img[:, i - 6:i + 6]
        if np.all(part == 255):
            crop_img = Image.fromarray(img[:, l:i])
            crop_img.save(save_path)
            List.append(save_path)
            recurse(i + 1, img, path, id + 1, List)
            break


def crop(path):
    List = []
    head_tail = os.path.split(path)
    img = np.array(Image.open(path))
    recurse(0, img, head_tail[1], 0, List)
    return List
