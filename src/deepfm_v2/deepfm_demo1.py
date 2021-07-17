# encoding=utf8
import sys
import numpy as np

'''https://zhuanlan.zhihu.com/p/154591869
连续特征没有参与到FM部分
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Layer, Add, Activation
from tensorflow.keras.utils import plot_model
features = [('gender', 'sparse'), ('age', 'dense'), ('income', 'dense'), ('vip_grade', 'sparse')]
sparse_feat_len = {'gender': 3, 'vip_grade':10}

def gen_data():
    x, y = {}, []
    while True:
        gender = np.array([1, 2, 1, 2, 1])
        age = np.array([10, 15, 30, 40, 10])
        income = np.array([10, 20, 30, 50, 40])
        vip_grade = np.array([1, 2, 3, 4, 5])
        labels = np.array([1, 1, 1, 0, 0])
        x['gender'], x['age'] = gender, age
        x['income'], x['vip_grade'] = income, vip_grade
        y = labels
        yield (x, y)
        


class FM(Layer):
    def __init__(self, **kwargs):
        super(FM, self).__init__()

    def call(self, inputs, **kwargs):
        sum_square = tf.square(tf.reduce_sum(inputs, axis=1, keepdims=True))
        square_sum = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
        res = sum_square - square_sum
        output = 0.5 * tf.reduce_sum(res, axis=2, keepdims=False)
        return output
    
def common_input(features, sparse_features_len, input_shape=(1,), emb_output_dim=5):
    input_sparse_layers, input_dense_layers = [], []
    embedding = []
    for name, ntype in features:
        if ntype == 'sparse':
            input = Input(shape=input_shape, name=name)
            input_sparse_layers.append(input)
            input_dim = sparse_features_len[name]
            emb = Embedding(input_dim=input_dim + 1, output_dim=emb_output_dim, name=name+'_emb')(input)
            embedding.append(emb)
        if ntype == 'dense':
            input = Input(shape=input_shape, name=name)
            input_dense_layers.append(input)
    
    return input_sparse_layers, embedding, input_dense_layers

input_sparse_layers, embedding, input_dense_layers = common_input(features, sparse_feat_len)

sparse_embedding = Concatenate(axis=-1)(embedding) #(None, 1, K)
dense_input = Concatenate(axis=-1)(input_dense_layers)

# 第一阶特征组合
first_order = Dense(1,activation=None, name='first_order')(Flatten()(sparse_embedding))
# 第二阶特征组合
second_order = FM(name='second_order')(sparse_embedding)

# deep part
sparse_embedding = Flatten()(sparse_embedding)
common_input_concate = Concatenate(axis=-1, name='concate')([sparse_embedding, dense_input])
dense_out = Dense(units=256, activation='relu', name='dense1')(common_input_concate)
dense_out = Dense(units=128, activation='relu', name='dense2')(dense_out)
y_dnn = Dense(units=1, activation='relu', name='dnn_output')(dense_out)

final_logit = Add(name='add_all')([first_order, second_order, y_dnn])
output = Activation('sigmoid')(final_logit)
input = input_sparse_layers + input_dense_layers
deepfm = Model(input, output)
deepfm.summary()
plot_model(deepfm, show_shapes=True, show_layer_names=False, to_file='mymodel.png')
if __name__ == '__main__':
    config = { 'train': True, 'fm_is_contain':True # 连续值是否加入fm部分进行训练
               }
    # deepfm.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'accuracy'], )
    # data = gen_data()
    # deepfm.fit(gen_data(), steps_per_epoch=2, epochs=2, verbose=1)
    # mf = 'deepfm.v01.h5'
    # deepfm.save(mf)