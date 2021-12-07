import pathlib
from PIL import Image

def get_match_dict(images_folder, formulas_folder, w = 530, h = 130):
    img_path = pathlib.Path(images_folder)
    formula_path = pathlib.Path(formulas_folder)

    img_list = []

    for x in img_path.iterdir():
        img = Image.open(x)

        width, height = img.size

        if width <= w and height <= h:
            img_list.append(x)

    img_to_formula = {
        x.as_posix(): (formula_path / (x.stem + '.txt')).as_posix()
        for x in img_list
    }

    return img_to_formula