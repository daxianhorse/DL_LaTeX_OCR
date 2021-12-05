# %%
import math
import pathlib
import tensorflow as tf
from PIL import Image


def image_process(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3)
    img = 255 - img
    img = tf.image.resize_with_crop_or_pad(img, 150, 350)
    return img


def load_data(images_folder, formulas_folder):
    img_path = pathlib.Path(images_folder)
    formula_path = pathlib.Path(formulas_folder)

    img_list = []

    for x in img_path.iterdir():
        img = Image.open(x)

        width, height = img.size

        if width <= 350 and height <= 150:
            img_list.append(x)

    img_to_formula = {
        x.as_posix(): (formula_path / (x.stem + '.txt')).as_posix()
        for x in img_list
    }

    valid_num = int(0.2 * len(img_to_formula))
    valid_ds = (list(img_to_formula.keys())[:valid_num],
                list(img_to_formula.values())[:valid_num])
    train_ds = (list(img_to_formula.keys())[valid_num:],
                list(img_to_formula.values())[valid_num:])

    return train_ds, valid_ds


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  experessed in terms of b, sin(a) and cos(a).
  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image
  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
    tf.to_float = lambda x: tf.cast(x, tf.float32)

    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in range(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in range(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x
