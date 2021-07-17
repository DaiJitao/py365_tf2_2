# encoding=utf8
import sys

import tensorflow as tf

from tensorflow.keras import Sequential, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Embedding, Masking, Dot, Lambda
import tensorflow.keras.backend as k
import numpy as np


class MyMean(Layer):
    def __init__(self, **kwargs):
        super(MyMean, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        data, emb = inputs
        all_ = tf.reduce_sum(data, axis=1)
        return k.dot(data, emb)/all_

    def compute_output_shape(self, input_shape):
        return (1,)



# def myMean(inputs):
#     inputdata, emb = inputs
#     b = tf.reduce_sum(inputdata, axis=1)
#     return k.dot(inputdata, emb) / b


# 初始化两个输入形参

in_interest = Input(shape=[3], name="interest")  # None*3, 最长长度3

emb = Embedding(input_dim=11, output_dim=1)(in_interest)
summ = MyMean()([in_interest, emb])

d = Model(inputs=[in_interest], outputs=[summ])
d.summary()

x_data = {"interest": np.array([[1, 0, 1], [1, 0, 1]])}

pred = d.predict(x_data)
print(pred)
print(pred.shape)
