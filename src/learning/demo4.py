# encoding=utf8
import sys
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as k

a = np.array(range(10))
print(a)
b = np.array([1,1,1,1,1,1,1,1,0,0])
print(b)
c = tf.boolean_mask(tensor=a, mask=b)
c= tf.reshape(c, [2,2,2])
print(c)
print(tf.size(c))

b = tf.reshape(a, shape=(2,5))
c = tf.reshape(a, shape=(5,2))
res = k.dot(b, c)
print(res)

