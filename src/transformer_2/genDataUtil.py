# encoding=utf8
import sys
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf




diff_max = 5
title_max = 21

def getData(inf):
    result = []
    labels = []
    with open(inf, 'r') as fp:
        for line in fp:
            arr = line.strip().split('\t')
            y = int(arr[0])
            labels.append(y)
            diff_arr = arr[1].split('|')
            diff = [int(i) for i in diff_arr]
            diff = np.reshape(diff, newshape=(1,-1))
            diff = pad_sequences(diff, padding='post', maxlen=6)
            title_arr = arr[2].split('|')
            title = [int(i) for i in title_arr]
            title = np.reshape(title, newshape=(1,-1))
            title = pad_sequences(title, padding='post', maxlen=22)
            res = np.concatenate( (diff, title), axis=1)
            result.append(res[0].tolist())

    data = np.array(result)
    y = np.array(labels)
    return (data, y)



if __name__ == '__main__':
    y_train = np.array([[1,0], [0,1], [0,1], [1,0]])
    # y_train = pd.get_dummies(y_train)
    # y_train = tf.constant(y_train)
    y_train = tf.argmax(y_train, axis=1)
    print(y_train.numpy())
    print(type(y_train))

if __name__ == '__main__1':
    inf = 'clean_train_data.csv'
    getData(inf)