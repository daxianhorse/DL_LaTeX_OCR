# 预测结构时禁用cuda，可以缩短初始化的时间
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from model.cnn_block import CnnEfficient
from model.transformer import get_transformer_model
from utils.match_dict import get_match_dict
from utils.vectorization import formula_vertorization
from model.build_dataset import get_train_valid_ds

# 词向量
vectorization = formula_vertorization('data/vocab.txt')
vocab_size = len(vectorization.get_vocabulary())
sequence_length = 50

# 载入数据集
match_dict = get_match_dict('data/biology/images', 'data/biology/formulas')
train_ds, val_ds = get_train_valid_ds(match_dict,
                                      vectorization,
                                      batch_size=16,
                                      valid_rate=0.2)

# 载入模型和权重(可选)
model = get_transformer_model(CnnEfficient,
                              vocab_size=vocab_size,
                              sequence_length=sequence_length)
model.load_weights('weights/transformer_math.h5')

# 以下为预测部分

# 生成对应词典
latex_vocab = vectorization.get_vocabulary()
latex_index_lookup = dict(zip(range(len(latex_vocab)), latex_vocab))
max_decoded_sentence_length = sequence_length - 1

import numpy as np
from utils.images import image_process


def decode_sequence(img_path):
    img = image_process(img_path)

    img = tf.expand_dims(img, 0)

    decoded_sentence = "<start>"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = vectorization([decoded_sentence])[:, :-1]
        predictions = model.predict([img, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = latex_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "<end>":
            break
    return decoded_sentence[8:-6]


from utils.image_crop import *
def get_latex(path):
    image_list = crop(path, crop_interval=330)
    latex = ""
    for i in image_list:
        print(i)
        latex += decode_sequence(i)
    return latex
