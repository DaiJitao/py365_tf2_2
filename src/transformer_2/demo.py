# encoding=utf8
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Concatenate, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot, text_to_word_sequence
import pandas as pd
import numpy as np
import pandas as pd
from urllib.parse import unquote

test = '%E5%88%AE%E5%88%B0%E5%A2%99'
print(unquote(test))

# df = pd.DataFrame([
#     ['green', 'A'],
#     ['red', 'B'],
#     ['blue', 'A']])
#
# df.columns = ['color', 'class']
# print(df)
# res = pd.get_dummies(df)
# print(res)
#
# import numpy as np
#
# a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,10]])
#
# print(a)
# print(a.shape)
# a = tf.nn.embedding_lookup(a, [[[2,1,1]]])
# print(a)
# print(a.shape)


