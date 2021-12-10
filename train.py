import tensorflow as tf
from model.transformer import *
from model.rnn import *
from model.cnn_block import *
from utils.match_dict import get_match_dict
from utils.vectorization import formula_vertorization
from model.build_dataset import get_train_valid_ds
from model.lrschedule import LRSchedule
from model.rnn_attention import get_rnn_attention_model

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

# # 载入模型(可选)
# model = get_transformer_model(CNNBlock=CnnResNet,
#                               vocab_size=vocab_size,
#                               sequence_length=sequence_length)
# model = get_transformer_model(CNNBlock=CnnEfficient, vocab_size=vocab_size,
#                               sequence_length=sequence_length)
# model.load_weights('weights/transformer_math.h5')
# model = get_rnn_model(CNNBlock=CnnEfficient)
# model = get_rnn_model(CNNBlock=CnnMobileNet)
model = get_rnn_attention_model(CnnEfficient)
model.load_weights('weights/rnn_attention_math.h5')

# 设置epochs大小
epochs = 15

# 生成学习率调节器
num_train_steps = len(train_ds) * epochs
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4,
                         warmup_steps=num_warmup_steps)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 模型概览
model.summary()

# 训练模型
model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# 保存权重
# 参数为保存路径
model.save_weights('tran_test.h5')