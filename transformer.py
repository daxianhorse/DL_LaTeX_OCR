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
batch_size = 32
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


train_ds, val_ds = load_data('data/biology/images', 'data/biology/formulas')

train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)

from keras.applications import efficientnet

image_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomContrast(0.3),
])


class CNNBlock(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_model = efficientnet.EfficientNetB0(
            input_shape=(224, 224, 3),
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


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)

        proj_input = self.layernorm_2(inputs + attention_output)
        return proj_input

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            layers.Dense(dense_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([
            tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32)
        ],
                         axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        attention_output_1 = self.attention_1(query=inputs,
                                              value=inputs,
                                              key=inputs,
                                              attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(attention_output_1 +
                                              attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)


"""#### Putting it all together: A Transformer for machine translation

**PositionalEmbedding layer**
"""


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim,
                                                 output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length,
                                                    output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config


"""**End-to-end Transformer**"""

embed_dim = 256
dense_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(224, 224, 3), name="img")
x = image_augmentation(encoder_inputs)
x = CNNBlock()(x)
x = layers.Dropout(0.5)(x)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = keras.Input(shape=(None, ), dtype="int32", name="formula")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
"""**Training the sequence-to-sequence Transformer**"""


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


epochs = 20

num_train_steps = len(train_ds) * epochs
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4,
                         warmup_steps=num_warmup_steps)

transformer.compile(optimizer=keras.optimizers.Adam(lr_schedule),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

transformer.summary()

transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)
transformer.save_weights('tran_test.h5')

transformer.load_weights('tran_test.h5')

import numpy as np

spa_vocab = vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = sequence_length - 1

from PIL import Image


def decode_sequence(img_path):
    # img = tf.io.read_file(img_path)
    # img = tf.io.decode_png(img, channels=3)

    img = Image.open(img_path).convert('L')
    img = keras.preprocessing.image.img_to_array(img)
    img = 255 - img
    img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    img = tf.image.grayscale_to_rgb(img)

    plt.imshow(img)

    img = tf.expand_dims(img, 0)

    decoded_sentence = "<start>"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([img, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "<end>":
            break
    return decoded_sentence


# print(decode_sequence('test_1.png'))
# # %%
# print(decode_sequence('test_2.png'))
# %%
print(decode_sequence('data/biology/images/2_0.png'))
# %%