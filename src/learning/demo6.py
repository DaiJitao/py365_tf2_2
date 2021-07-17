# encoding=utf8
import sys

import tensorflow as tf
from tensorflow.keras.layers import Softmax, ReLU
import numpy as np
np.random.seed(1)
data = np.random.randint(10, size=[1,3])
print(data)
data = tf.constant(data)
print(data)
res = Softmax(data, axis=-1)

print(res)