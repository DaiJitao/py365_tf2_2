# encoding=utf8
import sys

import tensorflow.keras.backend as k
from tensorflow.keras.activations import relu
from tensorflow.keras.backend import dot
from tensorflow.keras.layers import Input, Dense, dot, Dot, Add, add, Multiply, multiply, Average, average, subtract, Maximum
from tensorflow.keras import layers,constraints,activations,initializers, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Concatenate
import numpy as np
import tensorflow as tf
if __name__ == '__main__':
    t = tf.cast(tf.range(0,20), tf.float32)
    # d1 = tf.norm(t, ord=2)
    # t = tf.reshape(t, shape=(1, 10))
    # d2 = tf.nn.l2_normalize(t, axis=1)
    # d3 = k.l2_normalize(t, axis=1)
    # y = tf.constant([0,0,1,1,1])
    # loss = tf.constant([1])

    # res = k.sum(loss * y) / k.sum(y)
    # res = k.mean(y)
    # print(res)
    #print(k.reshape(t, shape=(2,-1)))
    y_pred=tf.reshape(t, shape=(5, 4))
    print(y_pred[None, 1])
    print(y_pred[:, None])
    print()

    a = tf.constant([[2],[3]], dtype=tf.float32)
    b = tf.constant([[1],[1]], dtype=tf.float32)
    print('------------------------------>')
    print(k.shape(a))
    print('------------------------------>')
    print(type(k.int_shape(a)))
