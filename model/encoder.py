# import the library
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

CNN_Encoder = models.Sequential([
    layers.Rescaling(scale=1. / 127.5, offset=-1),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPool2D(2, 2, 'same'),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPool2D(2, 2),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPool2D((2, 1), (2, 1)),
    layers.Conv2D(512, 3, padding='same', activation='relu'),
    layers.MaxPool2D((1, 2), (1, 2)),
    layers.Conv2D(512, 3, activation='relu'),
    # layers.Flatten(),
])

