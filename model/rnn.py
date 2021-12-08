# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.images import add_timing_signal_nd






def get_rnn_model(CNNBlock, embed_dim=256, latent_dim=512, vocab_size=417):
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

    return seq2seq_rnn
