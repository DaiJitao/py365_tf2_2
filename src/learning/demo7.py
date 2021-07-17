# encoding=utf8
import sys
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAvgPool1D, Lambda
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

np.random.seed(1)
data = np.random.randint(0,6.0, size=[2,2,3])
data = data.astype(np.float)
print(data)

res = GlobalAveragePooling1D(data_format='channels_last')(data)
print(res)

def avg(inputs):
    return K.mean(inputs, axis=1)

res = Lambda(avg, name='avg')(data)
print(res)


def get_labels_of_similarity(y_pred):
    idxs = K.arange(0, K.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    labels = K.equal(idxs_1, idxs_2)
    labels = K.cast(labels, K.floatx())
    return labels

idxs = K.arange(0, 10)
res = get_labels_of_similarity(idxs)
print(res)

K.sparse_categorical_crossentropy()