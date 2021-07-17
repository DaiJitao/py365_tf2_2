# encoding=utf8
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import ModelCheckpoint

batch_size = 128
epochers = 20
n_classes = 10
lr = 0.1
width = 28
height = 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(y_test)
label = y_test[:10]
print(label)
print(tf.one_hot(label, depth=10))
class LogisticRegression(tf.keras.Model):
    def __init__(self, num_classes):
        super(LogisticRegression, self).__init__()
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=None, mask=None):
        output = self.dense(inputs)
        with tf.device('cpu:0'):
            output = tf.nn.softmax(output)

        return output
print('keras:')
test = tf.keras.utils.to_categorical(label,10)
print(test)



