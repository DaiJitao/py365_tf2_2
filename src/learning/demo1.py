# encoding=utf8
import sys

import tensorflow as tf
from tensorflow.keras.layers import Layer

class MyDense(Layer):

    def __init__(self, inp_dim, oup_dim):
        super(MyDense, self).__init__()
        self.kernel = self.add_variable('w', [inp_dim, oup_dim], trainable=True)

    def call(self, inputs, **kwargs):
        out = inputs * self.kernel
        out = tf.nn.relu(out)
        return out

if __name__ == '__main__':
    dense = MyDense(100, 200)
    print(dense)

