import tensorflow as tf
from utils.images import image_process

_vectorization = None
height = 130
width = 530


def get_train_valid_ds(match_dict,
                       vectorization_formula,
                       batch_size=16,
                       valid_rate=0.2):

    global _vectorization

    valid_num = int(len(match_dict) * valid_rate)

    valid_list = (list(match_dict.keys())[:valid_num],
                  list(match_dict.values())[:valid_num])

    train_list = (list(match_dict.keys())[valid_num:],
                  list(match_dict.values())[valid_num:])

    _vectorization = vectorization_formula

    train_ds = make_dataset(train_list, batch_size)
    val_ds = make_dataset(valid_list, batch_size)

    return train_ds, val_ds


def format_dataset(img, formula):
    img = image_process(img, width, height)
    formula = tf.io.read_file(formula)
    formula = _vectorization('<start> ' + formula + ' <end>')
    return ({
        "img": img,
        "formula": formula[:-1],
    }, formula[1:])


def make_dataset(pairs, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(pairs)
    dataset.shuffle(len(pairs[0]))
    dataset = dataset.map(format_dataset, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(16)