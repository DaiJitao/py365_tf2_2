# encoding=utf8
import sys

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, BatchNormalization, Activation,Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences

input_len = 32
x_in = Input(shape=(input_len,), name='Input-Token-MLP')
s_in = Input(shape=(input_len,), name='Input-Segment-MLP')
concated_vec = Concatenate(axis=1, name='Token-Segment-Concate')([x_in, s_in]) # None,64
emb = Embedding(input_dim=23000, output_dim=256, name='embedding')(concated_vec) # None,64, 256

qt_vec = Dense(units=768, name='qt-dense-1')(emb)
qt_vec = BatchNormalization(name='norm-1')(qt_vec)
qt_vec = Activation('relu')(qt_vec)

qt_vec = Dense(units=768, name='qt-dense-2')(qt_vec)
qt_vec = BatchNormalization(name='norm-2')(qt_vec)
qt_vec = Activation('relu')(qt_vec)

qt_vec = Dense(units=512, activation='relu', name='qt-dense-3')(qt_vec)
qt_vec = BatchNormalization(name='norm-3')(qt_vec)
qt_vec = Activation('relu')(qt_vec)

qt_vec = Dense(units=128, activation='relu', name='qt-dense-4')(qt_vec) #None,64,128

data = list(range(1,10))
data = tf.constant(data)
res =  Embedding(input_dim=23000, output_dim=256, name='embedding')(data)
print(res)
print(res.shape)
res = tf.keras.layers.Reshape([1,-1])(res)
print(res.shape)

