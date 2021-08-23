# encoding=utf8
import sys

''' https://github.com/jc-LeeHub/Recommend-System-tf2.0/blob/master/xDeepFM/model.py '''

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Layer, Input, Concatenate, Reshape
from tensorflow.keras.models import Model

class Linear(Layer):
    
    def __ini__(self):
        super(Linear, self).__init__()
        self.output = Dense(1, activation=None)
        
    def call(self, inputs, **kwargs):
        output = self.output(inputs)
        return output
    
class Dense_layer(Layer):
    
    def __init__(self, hidden_units, out_dim=1, activation='relu', dropout=0.0):
        super(Dense_layer, self).__init__()
        self.hidden_layers = [Dense(i, activation=activation) for i in hidden_units]
        self.out_layer = Dense(out_dim, activation=None)
        self.dropout = Dropout(rate=dropout)
    
    def call(self, inputs, **kwargs):
        # inputs: [None, n*k]
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        
        x = self.dropout(x)
        output = self.out_layer(x)
        return output

class CIN(Layer):
    def __init__(self, cin_size):
        super(CIN, self).__init__()
        self.cin_size = cin_size
        
    def build(self, input_shape):
        # input_shape: [None, n, k]
        self.field_num = [input_shape[1]] + self.cin_size # 每层的矩阵个数(包括第0层)

        self.cin_W = [self.add_weight(
                         name='w'+str(i),
                         shape=(1, self.field_num[0]*self.field_num[i], self.field_num[i+1]),
                         initializer=tf.initializers.glorot_uniform(),
                         regularizer=tf.keras.regularizers.l1_l2(1e-5),
                         trainable=True)
                      for i in range(len(self.field_num)-1)]

    def call(self, inputs, **kwargs):
        # inputs: [None, n, k]
        k = inputs.shape[-1]
        res_list = [inputs]
        X0 = tf.split(inputs, k, axis=-1)           # 最后维切成k份，list: k * [None, field_num[0], 1]
        for i, size in enumerate(self.field_num[1:]):
            Xi = tf.split(res_list[-1], k, axis=-1) # list: k * [None, field_num[i], 1]
            x = tf.matmul(X0, Xi, transpose_b=True) # list: k * [None, field_num[0], field_num[i]]
            x = tf.reshape(x, shape=[k, -1, self.field_num[0]*self.field_num[i]])
                                                    # [k, None, field_num[0]*field_num[i]]
            x = tf.transpose(x, [1, 0, 2])          # [None, k, field_num[0]*field_num[i]]
            x = tf.nn.conv1d(input=x, filters=self.cin_W[i], stride=1, padding='VALID')
                                                    # (None, k, field_num[i+1])
            x = tf.transpose(x, [0, 2, 1])          # (None, field_num[i+1], k)
            res_list.append(x)

        res_list = res_list[1:]   # 去掉X0
        res = tf.concat(res_list, axis=1)  # (None, field_num[1]+...+field_num[n], k)
        output = tf.reduce_sum(res, axis=-1)  # (None, field_num[1]+...+field_num[n])
        return output

class xDeepFM(Model):
    def __init__(self, feature_columns, cin_size, hidden_units, out_dim=1, activation='relu', dropout=0.0):
        super(xDeepFM, self).__init__()
        self.dense_feats, self.sparse_feats, self.sparse_feat_size, self.sparse_out_dim = feature_columns
        self.embed_layers = [Embedding(input_dim=self.sparse_feat_size.get(feat), output_dim=self.sparse_out_dim.get(feat))
                                    for feat in self.sparse_feats]
        self.linear = Linear()
        self.dense_layer = Dense_layer(hidden_units, out_dim, activation, dropout)
        self.cin_layer = CIN(cin_size)
        self.out_layer = Dense(1, activation=None)

    def call(self, inputs, training=None, mask=None):
        dense_inputs, sparse_inputs = inputs[:, :13], inputs[:, 13:]

        # linear
        linear_out = self.linear(inputs)

        emb = [self.embed_layers[i](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])] # [n, None, k]
        emb = tf.transpose(tf.convert_to_tensor(emb), [1, 0, 2]) # [None, n, k]

        # CIN
        cin_out = self.cin_layer(emb)

        # dense
        emb = tf.reshape(emb, shape=(-1, emb.shape[1]*emb.shape[2]))
        emb = tf.concat([dense_inputs, emb], axis=1)
        dense_out = self.dense_layer(emb)

        output = self.out_layer(linear_out + cin_out + dense_out)
        return tf.nn.sigmoid(output)
    

if __name__ == '__main__':
    '''离散特征要进行labelEncoding处理；原理参照：from sklearn.preprocessing import LabelEncoder
    '''
    dense_feats = ['d1', 'd2']
    sparse_feats = ['s1', 's2']
    alias_feat = {}
    sparse_feat_size = {'s1': 10, 's2': 10}
    sparse_out_dim = {'s1': 8, 's2': 8}
    
    test_size = 0.2
    hidden_units = [256, 128, 64]
    dropout = 0.3
    cin_size = [128, 128]
    
    # 输入部分 ------------------------------
    dense_inputs = []
    for f in dense_feats:
        if f in alias_feat:
            f = alias_feat[f]
        _input = Input([1], name=f)
        dense_inputs.append(_input)

    sparse_inputs = []
    for f in sparse_feats:
        if f in alias_feat:
            f = alias_feat[f]
        _input = Input([1], name=f)
        sparse_inputs.append(_input)

    # -------- 线性部分 --------------------
    # 将输入拼接到一起，方便连接 Dense 层
    all_feats_inputs = Concatenate(axis=1, name='dense_concate')(dense_inputs + sparse_inputs)  # ?, 13
    # 然后连上输出为1个单元的全连接层，表示对 dense 变量的加权求和
    linear_part = Dense(1, activation=None, name='linear_part')(all_feats_inputs)  # ?, 1
    
    # ----------- 构造embedding
    sparse_kd_embed = []
    for i, _input in enumerate(sparse_inputs):
        feat_name = sparse_feats[i]
        voc_size = sparse_feat_size.get(feat_name)
        k_size = sparse_out_dim.get(feat_name)
        if voc_size == None:
            voc_size = sparse_feat_size.get('default')
    
        reg = tf.keras.regularizers.l2(0.7)
        _embed = Embedding(voc_size + 5, output_dim=k_size, embeddings_regularizer=reg)(_input) # (?, 1, k)
        sparse_kd_embed.append(_embed)

    dense_kd_embed = []
    for feat, _input in zip(dense_feats, dense_inputs):
        # dense_kd_emb = Dense(k_size, use_bias=True, name=feat + "_dense_emb")(_input)
        dense_reshape = Reshape([1, 1])(_input) # (?, 1, 1)
        dense_kd_embed.append(dense_reshape)
    
    all_feats_inputs = Concatenate(axis=1)(sparse_kd_embed + dense_kd_embed) #(?, 1*n, n*k + dense_num)
    
    # --------- DNN部分------------------------
