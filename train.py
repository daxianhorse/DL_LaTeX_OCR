import tensorflow as tf
from model.transformer import get_transformer_model
from utils.match_dict import get_match_dict
from utils.vectorization import formula_vertorization
from model.build_dataset import get_train_valid_ds

# 预测结构时禁用cuda，可以缩短初始化的时间
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 载入模型和权重
transformer = get_transformer_model()
transformer.load_weights('weights/transformer_math.h5')

# 词向量
vectorization = formula_vertorization('data/vocab.txt')
vocab_size = len(vectorization.get_vocabulary())
sequence_length = 50

# 载入数据集
match_dict = get_match_dict('data/maths/images', 'data/maths/formulas')
train_ds, val_ds = get_train_valid_ds(match_dict, vectorization)


# 学习率调节模块
class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
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

# 设置epochs大小
epochs = 15

# 生成学习率调节器
num_train_steps = len(train_ds) * epochs
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4,
                         warmup_steps=num_warmup_steps)

# 编译模型
transformer.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])

# 训练模型
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)

# 保存权重
# 参数为保存路径
transformer.save_weights('tran_test.h5')