# encoding=utf8
import sys
'''https://www.yuque.com/chudi/tzqav9/ny150b#UfkYS
'''

import tensorflow as tf
from tensorflow import keras
from utils import *

EPOCH = 10
BATCH_SIZE = 32
VEC_DIM = 10
DNN_LAYERS = [64, 128, 64]
DROPOUT_RATE = 0.5

base, test = loadData()
# 所有的特征各个类别值个数之和
FEAT_CATE_NUM = base.shape[1] - 1
K = tf.keras.backend


class CrossLayer(keras.layers.Layer):
    def __init__(self, feat_num, vec_dim, **kwargs):
        self.feat_num = feat_num
        self.vec_dim = vec_dim
        super(CrossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.V = self.add_weight(name='V', shape=(self.feat_num, self.vec_dim), initializer='uniform', trainable=True)
        super(CrossLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        a = K.pow(K.dot(inputs, self.V), 2)
        b = K.dot(K.pow(inputs, 2), K.pow(self.V, 2))
        return 0.5 * K.mean(a - b, 1, keepdims=True)


def run():
    # 返回id化特征 和 one-hot特征
    val_x_id, val_x_hot, val_y = getAllData(test)
    train_x_id, train_x_hot, train_y = getAllData(base)
    cate_num = val_x_id[0].shape[0]
    hot_num = val_x_hot[0].shape[0]
    sub_emb_arr = []
    product_list = []

    # Deep 部分
    inputs_id = keras.Input((cate_num,))
    emb = keras.layers.Embedding(FEAT_CATE_NUM, VEC_DIM, input_length=cate_num)(inputs_id)
    deep = keras.layers.Flatten()(emb)
    deep = keras.layers.Dropout(DROPOUT_RATE)(deep)
    for units in DNN_LAYERS:
        deep = keras.layers.Dense(units, activation='relu')(deep)
        deep = keras.layers.Dropout(DROPOUT_RATE)(deep)

    # FM 部分
    # 将emb切分成各个field的小emb
    split_arr = tf.split(emb, cate_num, 1)
    for split in split_arr:
        sub_emb_arr.append(keras.layers.Flatten()(split))
    # 内积
    for i in range(0, len(sub_emb_arr)):
        for j in range(i + 1, len(sub_emb_arr)):
            product_list.append(keras.layers.Dot(axes=1)([sub_emb_arr[i], sub_emb_arr[j]]))
    wide = keras.Input((hot_num,))

    deep_fm = keras.layers.concatenate(product_list + [wide] + [deep])
    deep_fm = keras.layers.Dropout(DROPOUT_RATE)(deep_fm)
    outputs = keras.layers.Dense(1, activation='sigmoid',kernel_regularizer=keras.regularizers.l2(0.001))(deep_fm)

    model = keras.Model(inputs=[inputs_id, wide], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer(0.001), metrics=[keras.metrics.AUC()])
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs',
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_grads=True,
                                             write_images=True,
                                             embeddings_freq=0,
                                             embeddings_layer_names=None,
                                             embeddings_metadata=None)

    model.fit([train_x_id, train_x_hot], train_y, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2,
              validation_data=([val_x_id, val_x_hot], val_y),
              callbacks=[tbCallBack])


run()

# Epoch 1/10
# 90562/90562 - 47s - loss: 0.6253 - auc: 0.7076 - val_loss: 0.5935 - val_auc: 0.7444
# Epoch 2/10
# 90562/90562 - 42s - loss: 0.5756 - auc: 0.7710 - val_loss: 0.5817 - val_auc: 0.7539
# Epoch 3/10
# 90562/90562 - 42s - loss: 0.5663 - auc: 0.7776 - val_loss: 0.5832 - val_auc: 0.7534
# Epoch 4/10
# 90562/90562 - 41s - loss: 0.5616 - auc: 0.7811 - val_loss: 0.5777 - val_auc: 0.7575
# Epoch 5/10
# 90562/90562 - 42s - loss: 0.5575 - auc: 0.7848 - val_loss: 0.5807 - val_auc: 0.7592
# Epoch 6/10
# 90562/90562 - 41s - loss: 0.5533 - auc: 0.7885 - val_loss: 0.5759 - val_auc: 0.7625
# Epoch 7/10
# 90562/90562 - 42s - loss: 0.5464 - auc: 0.7948 - val_loss: 0.5700 - val_auc: 0.7664
# Epoch 8/10
# 90562/90562 - 40s - loss: 0.5395 - auc: 0.8016 - val_loss: 0.5678 - val_auc: 0.7705
# Epoch 9/10
# 90562/90562 - 41s - loss: 0.5330 - auc: 0.8080 - val_loss: 0.5682 - val_auc: 0.7716
# Epoch 10/10
# 90562/90562 - 40s - loss: 0.5277 - auc: 0.8124 - val_loss: 0.5669 - val_auc: 0.7735