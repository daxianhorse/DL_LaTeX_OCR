import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications import efficientnet
from keras.applications import resnet_v2
from keras.applications import mobilenet_v2
from utils.images import add_timing_signal_nd


class CnnMobileNet(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_model = mobilenet_v2.MobileNetV2(
            include_top=False,
            weights='imagenet',
        )

        self.reshape = layers.Reshape((-1, self.base_model.output.shape[-1]))

    def call(self, inputs):
        x = mobilenet_v2.preprocess_input(inputs)
        x = self.base_model(inputs)
        x = add_timing_signal_nd(x)
        x = self.reshape(x)
        return x


class CnnEfficient(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_model = efficientnet.EfficientNetB0(
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


class CnnResNet(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_model = resnet_v2.ResNet50V2(
            include_top=False,
            weights='imagenet',
        )

        self.reshape = layers.Reshape((-1, self.base_model.output.shape[-1]))

    def call(self, inputs):
        x = resnet_v2.preprocess_input(inputs)
        x = self.base_model(inputs)
        x = add_timing_signal_nd(x)
        x = self.reshape(x)
        return x