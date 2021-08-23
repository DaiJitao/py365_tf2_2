# encoding=utf8
import sys

import tensorflow as tf

import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Embedding, Concatenate, Dense,Input
from tensorflow.keras.models import Model

a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
a1 = tf.tile(a, [2, 3])
print(a1)
print(a1.shape)


a = tf.constant([[1,2,3,3],
                 [1, 2, 3, 4]])
print(a)
res = tf.cumsum(a, axis=0)
print(res)
y = tf.constant([[2]], dtype=tf.float32)

res = Embedding(input_dim=20, output_dim=1)(y)
print('------------>:\n\n')
print(res)

res1 = Embedding(input_dim=20, output_dim=2, input_length=1)(y)
res2 = Embedding(input_dim=20, output_dim=2, input_length=1)(y)
print(res1)
print(res2)
input = Input(shape=[1])
res3 = Dense(1, activation=None, use_bias=True)(input)
d = Model(input, res3)
print('---res3',res)
d.get_layer()
