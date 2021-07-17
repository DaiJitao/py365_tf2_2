# encoding=utf8
import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, AveragePooling2D, AveragePooling1D, Reshape, Lambda, Concatenate, Average
import numpy as np
from tensorflow.keras import backend as K

np.random.seed(1)
x = np.random.randint(1,10,size=[2, 1, 9])
print(np.reshape(x, newshape=(2,9)))

x=tf.Variable(x, dtype=tf.float32)

# channels_last对应于具有形状(batch, length, channels)的输入，而channels_first对应于具有形状(batch, channels, length)的输入。
avg = AveragePooling1D(pool_size=3, strides=2, padding='same', data_format='channels_first')(x)
print(avg)
res = Reshape([-1])(avg)
print(res)

def absDiff(inputs):
    a_vec, b_vec = inputs[0], inputs[1]
    return K.abs(a_vec - b_vec)

diff = Lambda(absDiff, name='abs-diff')([res, res+1])
all_vec = Concatenate(axis=1)([diff, diff, diff])
print(all_vec)
outputs=Dense(3, activation=tf.nn.softmax)(all_vec)
print(outputs)

print(tf.nn.sigmoid(tf.constant([0], dtype=tf.float32)))
