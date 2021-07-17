# encoding=utf8
import sys

import tensorflow as tf

from tensorflow.keras import Sequential, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Embedding, Masking, Multiply, Dot
import tensorflow.keras.backend as K
import numpy as np


class MyMean(Layer):

    def __init__(self, **kwargs):
        super(MyMean, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        return input_shape[0]





# 四个ID列表型样本，经过padding
ID_seq0 = [1, 0, 1, 0]
ID_seq1 = [1, 1, 0, 0]
ID_seq2 = [1, 1, 1, 0]
ID_seq3 = [1, 1, 1, 1]
data = np.array([ID_seq0])#, ID_seq1, ID_seq2, ID_seq3])

model = Sequential()
in_ = Input(shape=[3])
# model.add(Masking())
model.add(Embedding(input_dim=6, output_dim=1, input_length=4))

model.save('test_model', save_format='tf')
result = model.predict(data)
print(result)
print(result.shape)
print(type(result))
print('------------------------------------------------')
result = np.array([
       [[1,2,3],
        [2,3,4]]
])
result = tf.convert_to_tensor(result,dtype=tf.float32)
f = tf.reduce_mean(result, axis=2)
print(f)
print(f.shape)
