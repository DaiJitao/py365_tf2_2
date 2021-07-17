# encoding=utf8
import sys
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling2D, Conv2D
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
(x_train, y_train), (x_test, y_test) = mnist.load_data()

a = [1, 2, 8, 4]
b = [5, 6, 7, 8]
x = tf.constant(a)
y = tf.constant(b)
dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 返回DataSet, (x,y)相当于zip(x,y)
for d in dataset:
    print(d)

# 返回一个可迭代的对象
iter_res = iter(dataset)
print('-->iter_res:', next(iter_res))

# 数据集对象常用的预处理api
y_true = tf.constant([[1., 0.], [1., 1.], [2, 2]])
y_pred = tf.constant([[0., 1.], [1., 1.], [-2, 2]])
cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
res = cosine_loss(y_true=y_true, y_pred=y_pred)
print(type(res))


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=1)
    res = x * y
    res = K.sum(res, axis=-1)
    return K.expand_dims(res)
    # return K.mean(res, axis=1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

distance = Lambda(cosine_distance)([y_true, y_pred])
print('distance:', distance)
print('\n\n\n')
print(y_true)
print('keepdims false:', K.mean(y_true, axis=1))
print('keepdims true:', K.mean(y_true, axis=1, keepdims=True))
print('keepdims true:', K.mean(y_true, axis=0, keepdims=True))
# print(tf.reduce_mean(y_true, axis=0))
# a = Input(shape=(2,))
# b = Input(shape=(2,))
# fenzi = Lambda(lambda x: sim_dot)([a, b])
# model = Model(inputs=[a, b], outputs=[fenzi])
# model.summary()
