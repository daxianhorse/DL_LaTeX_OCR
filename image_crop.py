from os.path import split

from PIL import Image

List = []
interval = 0
path = ""
img = None


def recurse(l, id):
    r = l + interval
    index = path.find(".")
    save_path = "./data/cropped_image/" + path[:index] + f"_{id}" + path[index:]
    w, h = img.size

    if r > w > l:
        crop_img = img.crop((l, 0, w, h))
        crop_img.save(save_path)
        List.append(save_path)
        return

    for i in range(r, -1, -1):
        part = img.crop((i - 6, 0, i + 6, h))
        if part.getextrema() == (255, 255):
            crop_img = img.crop((l, 0, i, h))
            crop_img.save(save_path)
            List.append(save_path)
            recurse(i + 1, id + 1)
            break


def crop(image_path, crop_interval=500):
    global List, interval, path, img
    List = []
    interval = crop_interval
    img = Image.open(image_path)
    head_tail = split(image_path)
    path = head_tail[1]
    recurse(0, 0)
    return List


# print(crop("C:/Users/isudfv/Documents/Tencent Files/1476088347/FileRecv/math_formula_images_grey/123_0.png"))
