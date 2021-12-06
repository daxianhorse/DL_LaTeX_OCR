# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers import embeddings
from tensorflow.keras.layers import TextVectorization
import matplotlib.pyplot as plt

from model.data_gen import *

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

sequence_length = 50
batch_size = 16
AUTOTUNE = tf.data.AUTOTUNE


def formula_vertorization(vocab_path):
    vectorization = TextVectorization(
        standardize=None,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorization.set_vocabulary(vocab_path)
    print(len(vectorization.get_vocabulary()))

    return vectorization


vectorization = formula_vertorization('data/vocab.txt')

vocab_size = len(vectorization.get_vocabulary())


def format_dataset(img, formula):
    img = image_process(img)
    formula = tf.io.read_file(formula)
    formula = vectorization('<start> ' + formula + ' <end>')
    return ({
        "img": img,
        "formula": formula[:-1],
    }, formula[1:])


def make_dataset(pairs):
    dataset = tf.data.Dataset.from_tensor_slices(pairs)
    dataset.shuffle(len(pairs[0]))
    dataset = dataset.map(format_dataset, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(16)


train_ds, val_ds = load_data('data/maths/images', 'data/maths/formulas')

train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)

from keras.applications import efficientnet


class CNNBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_model = efficientnet.EfficientNetB0(
            # input_shape=(150, 500, 3),
            include_top=False,
            weights='imagenet',
        )
        self.base_model.trainable = True
        self.reshape = layers.Reshape((-1, self.base_model.output.shape[-1]))

    def call(self, inputs):
        x = efficientnet.preprocess_input(inputs)
        x = self.base_model(inputs)
        x = add_timing_signal_nd(x)
        x = self.reshape(x)
        return x


embed_dim = 256
latent_dim = 512

encoder_inputs = keras.Input(shape=(None, None, 3), name="img")
x = CNNBlock()(encoder_inputs)
x = layers.Dense(embed_dim, activation='relu')(x)
encoded_source = layers.Bidirectional(layers.GRU(latent_dim),
                                      merge_mode="sum")(x)

# """**GRU-based decoder and the end-to-end model**"""

past_target = keras.Input(shape=(None, ), dtype="int32", name="formula")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
seq2seq_rnn = keras.Model([encoder_inputs, past_target], target_next_step)

# """**Training our recurrent sequence-to-sequence model**"""


class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )

    def get_config(self):
        config = {
            "post_warmup_learning_rate": self.post_warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        return config


epochs = 15

num_train_steps = len(train_ds) * epochs
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4,
                         warmup_steps=num_warmup_steps)

seq2seq_rnn.compile(optimizer=keras.optimizers.Adam(lr_schedule),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

seq2seq_rnn.summary()

# seq2seq_rnn.fit(train_ds, epochs=epochs, validation_data=val_ds)
# seq2seq_rnn.save_weights('rnn_test.h5')

seq2seq_rnn.load_weights('rnn_test.h5')

import numpy as np

spa_vocab = vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = sequence_length - 1

from PIL import Image


def decode_sequence(img_path):
    img = Image.open(img_path).convert('L')
    img = keras.preprocessing.image.img_to_array(img)
    img = 255 - img
    img = tf.image.resize_with_crop_or_pad(img, 150, 500)
    img = tf.image.grayscale_to_rgb(img)

    plt.imshow(img)

    img = tf.expand_dims(img, 0)
    decoded_sentence = "<start>"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = vectorization([decoded_sentence])
        next_token_predictions = seq2seq_rnn.predict(
            [img, tokenized_target_sentence])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "<end>":
            break
    return decoded_sentence


print(decode_sequence('data/biology/images/2_0.png'))
# %%
