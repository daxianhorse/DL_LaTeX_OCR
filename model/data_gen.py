# %%
import pathlib
import tensorflow as tf


def image_process(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3)
    img = 255 - img
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    return img


def load_data(images_folder, formulas_folder):
    img_path = pathlib.Path(images_folder)
    formula_path = pathlib.Path(formulas_folder)

    img_path = [x for x in img_path.iterdir()]
    img_to_formula = {
        x.as_posix(): (formula_path / (x.stem + '.txt')).as_posix()
        for x in img_path
    }

    valid_num = int(0.2 * len(img_to_formula))
    valid_ds = (list(img_to_formula.keys())[:valid_num],
                list(img_to_formula.values())[:valid_num])
    train_ds = (list(img_to_formula.keys())[valid_num:],
                list(img_to_formula.values())[valid_num:])

    return train_ds, valid_ds
