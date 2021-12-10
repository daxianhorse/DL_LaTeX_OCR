import distance
import numpy as np
import tensorflow as tf
from model.cnn_block import CnnEfficient
from model.transformer import get_transformer_model
from utils.match_dict import get_match_dict
from utils.vectorization import formula_vertorization
from model.build_dataset import get_train_valid_ds
from model.rnn_attention import get_rnn_attention_model

# 词向量
vectorization = formula_vertorization('data/vocab.txt')
vocab_size = len(vectorization.get_vocabulary())
sequence_length = 50

# 生成对应词典
latex_vocab = vectorization.get_vocabulary()
latex_index_lookup = dict(zip(range(len(latex_vocab)), latex_vocab))
max_decoded_sentence_length = sequence_length - 1

# 定义评测指标
# 参考自https://github.com/LinXueyuanStdio/LaTeX_OCR_PRO/blob/master/model/evaluation/text.py
# 全部都改写成了keras metrics即插即用的形式


def exact_match_score(references, hypotheses):
    """Computes exact match scores.
    Args:
        references: list of list of tokens (one ref)
        hypotheses: list of list of tokens (one hypothesis)
    Returns:
        exact_match: (float) 1 is perfect
    """

    exact_match = 0

    for ref, hypo in zip(references.numpy(), hypotheses.numpy()):
        pre = []
        for i in range(sequence_length - 1):
            sampled_token_index = np.argmax(hypo[i, :])

            if sampled_token_index == 3:
                hypo = np.array(pre, dtype='int32')
                break
            else:
                pre.append(sampled_token_index)

        for i in range(sequence_length - 1):
            if ref[i] == 3:
                ref = ref[:i].astype('int32')
                break

        if np.array_equal(ref, hypo):
            exact_match += 1

    return exact_match / float(max(len(hypotheses), 1))


def edit_distance(references, hypotheses):
    """Computes Levenshtein distance between two sequences.
    Args:
        references: list of list of token (one hypothesis)
        hypotheses: list of list of token (one hypothesis)
    Returns:
        1 - levenshtein distance: (higher is better, 1 is perfect)
    """

    d_leven, len_tot = 0, 0

    for ref, hypo in zip(references.numpy(), hypotheses.numpy()):
        pre = ""
        for i in range(sequence_length - 1):
            sampled_token_index = np.argmax(hypo[i, :])

            if sampled_token_index == 3:
                break
            else:
                pre += latex_index_lookup[sampled_token_index]

        src = ""
        for i in range(sequence_length - 1):
            if ref[i] == 3:
                ref = ref[:i].astype('int32')
                break
            else:
                src += latex_index_lookup[ref[i]]

        d_leven += distance.levenshtein(pre, src)
        len_tot += float(max(len(pre), len(src)))

    return 1. - d_leven / len_tot


# 载入数据集
match_dict = get_match_dict('data/maths/images', 'data/maths/formulas')
train_ds, val_ds = get_train_valid_ds(match_dict,
                                      vectorization,
                                      batch_size=16,
                                      valid_rate=0.3)

# 载入模型和权重
model = get_transformer_model(CnnEfficient,
                              vocab_size=vocab_size,
                              sequence_length=sequence_length)
model.load_weights('weights/transformer_math.h5')
# model = get_rnn_attention_model(CnnEfficient)
# model.load_weights('weights/rnn_attention_math.h5')

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=[edit_distance, exact_match_score],
              run_eagerly=True)

model.evaluate(val_ds)
