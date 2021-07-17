# encoding=utf8
import sys

from tensorflow.keras.layers import Concatenate, Dense, Input, Embedding, Flatten, Add, Lambda, Multiply, Subtract
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import AUC, BinaryAccuracy
from tensorflow.keras.utils import plot_model, multi_gpu_model
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os
import tensorflow as tf

"""
src: https://github.com/zxxwin/tf2_deepfm
连续特征参与fm的一阶部分，而二阶交叉部分只考虑了离散特征，原因是embedding之后，维度不一致；
"""
config = {
    'is_numeric_in_fm': True, # 数值类型是否加入fm的二阶交叉部分，进行交叉运算
    "k_size": 8, # embedding维度
    "first_order_emb": True, # 计算一节特征时，一节特征是否需要embedding
}
first_order_emb = config['first_order_emb']
isUsegpu = False # 是否使用gpu
if isUsegpu:
    use_gpu = '2,3'
    os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
    gpus = 2
else:
    use_gpu = ''
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpus = 0
    
config['use_gpu'], config['gpus'] = use_gpu, gpus


is_numeric_in_fm = config["is_numeric_in_fm"]
k_size = config['k_size']

dense_feats = ["age", "income"]
sparse_feats = ["gender", "vip_grade"]
sparse_feat_size = {'gender': 3, 'vip_grade': 10}
# 构造 dense 特征的输入 ----------------------------------------------
dense_inputs = []
for f in dense_feats:
    _input = Input([1], name=f)
    dense_inputs.append(_input)

sparse_inputs = []
for f in sparse_feats:
    _input = Input([1], name=f)
    sparse_inputs.append(_input)
    
# 将输入拼接到一起，方便连接 Dense 层
concat_dense_inputs = Concatenate(axis=1, name='dense_concate')(dense_inputs)  # ?, 13
# 然后连上输出为1个单元的全连接层，表示对 dense 变量的加权求和
fst_order_dense_layer = Dense(1, name='fst_ord_dense')(concat_dense_inputs)  # ?, 1
# -------------------------------fm first order --------------------------------
if first_order_emb:
    sparse_1d_embed = []
    for i, _input in enumerate(sparse_inputs):
        f = sparse_feats[i]
        voc_size = sparse_feat_size[f]
        # 使用 l2 正则化防止过拟合
        reg = tf.keras.regularizers.l2(0.5)
        _embed = Embedding(voc_size + 1, 1, embeddings_regularizer=reg)(_input)
        # 由于 Embedding 的结果是二维的，
        # 因此如果需要在 Embedding 之后加入 Dense 层，则需要先连接上 Flatten 层
        _embed = Flatten()(_embed)
        sparse_1d_embed.append(_embed)
    # 对每个 embedding lookup 的结果 wi 求和
    fst_order_sparse_layer = Add(name='fst_ord_sparse')(sparse_1d_embed)
else:
    concat_sparse_inputs = Concatenate(axis=1, name='sparse_concate')(sparse_inputs)
    fst_order_sparse_layer = Dense(1, name='fst_ord_sparse', use_bias=True)(concat_sparse_inputs)  # ?, 1
    #fst_order_sparse_layer = Add(name='fst_ord_sparse')(concat_sparse_inputs)

linear_part = Add(name='fst_ord')([fst_order_dense_layer, fst_order_sparse_layer])
# ----------------------------------------sencond order-----------------------------------------------------
# 只考虑sparse的二阶交叉
sparse_kd_embed = []
for i, _input in enumerate(sparse_inputs):
    f = sparse_feats[i]
    voc_size = sparse_feat_size[f]
    reg = tf.keras.regularizers.l2(0.7)
    _embed = Embedding(voc_size + 1, k_size, embeddings_regularizer=reg)(_input)
    sparse_kd_embed.append(_embed)
# 1.将所有sparse的embedding拼接起来，得到 (n, k)的矩阵，其中n为特征数，k为embedding大小
concat_sparse_kd_embed = Concatenate(axis=1)(sparse_kd_embed)  # ?, n, k
# 2.先求和再平方
sum_kd_embed = Lambda(lambda x: K.sum(x, axis=1), name='snd_ord_sum')(concat_sparse_kd_embed)  # ?, k
square_sum_kd_embed = Multiply()([sum_kd_embed, sum_kd_embed])  # ?, k
# 3.先平方再求和
square_kd_embed = Multiply()([concat_sparse_kd_embed, concat_sparse_kd_embed]) # ?, n, k
sum_square_kd_embed = Lambda(lambda x: K.sum(x, axis=1))(square_kd_embed)  # ?, k
# 4.相减除以2
sub = Subtract()([square_sum_kd_embed, sum_square_kd_embed])  # ?, k
sub = Lambda(lambda x: x*0.5)(sub)  # ?, k
snd_order_sparse_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name='snd_ord')(sub)  # ?, 1
# -------------------deep part ---------------------------------------------------------------------------------
ften_sparse_emb = Flatten()(concat_sparse_kd_embed)  # ?, n*k
deep_inputs = Concatenate(axis=1, name='sparse_dense_concate')([ften_sparse_emb, concat_dense_inputs]) # (?, n*k + desne_feat_size)

deep_part = Dense(512, activation='relu', name='deep_part1')(deep_inputs)
deep_part = Dropout(0.5, name='dropout_deep_part1')(deep_part)  # ?, 512
deep_part = Dense(256, activation='relu', name='deep_part2')(deep_part)
deep_part = Dropout(0.3, name='dropout_deep_part2')(deep_part)  # ?, 256
deep_part = Dense(256, activation='relu', name='deep_part3')(deep_part)
deep_part = Dropout(0.1, name='dropout_deep_part3')(deep_part)  # ?, 256
fc_layer_output = Dense(1)(deep_part)  # ?, 1

# -----------------组合 --------------------------------------------------------------------------------
output_layer = Add(name='deepfm')([linear_part, snd_order_sparse_layer, fc_layer_output])
output_layer = Activation("sigmoid")(output_layer)


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
        
devices = []
if use_gpu != '':
    for i in range(gpus):
        devices.append('/gpu:{}'.format(i))
config['devices'] = devices

if __name__ == '__main__':
    train = True
    batch_size = 2
    train_data_length = 10
    epochs = 100
    steps_per_epoch = int(train_data_length / batch_size)
    modelfile = 'best_deepfm.h5'
    config['train'], config['train_data_length'], config['modelfile'] = train, train_data_length, modelfile
    print('--->config:{}\n'.format(config))
    if train:
        train_data = gen_data()
        test_data = gen_data()
        # call back function
        checkpoint = ModelCheckpoint(
            modelfile, monitor='binary_crossentropy', verbose=1, save_best_only=True, mode='min')
        # 当使用ReduceLROnPlateau在训练过程中优化减小learning_rate,
        reduce_lr = ReduceLROnPlateau(
            monitor='binary_crossentropy', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
        earlystopping = EarlyStopping(
            monitor='binary_crossentropy', min_delta=0.0001, patience=8, verbose=1, mode='auto')
        callbacks = [checkpoint, reduce_lr, earlystopping]
        #strategy = tf.distribute.MirroredStrategy(devices=devices)
        #with strategy.scope():
        if True:
            model = Model(dense_inputs + sparse_inputs, output_layer)
            #plot_model(model, show_shapes=True, show_layer_names=True, to_file='deepfm.png')
            #print("--->to png:" + 'deepfm.png')
            if gpus > 1:
                model = multi_gpu_model(model, gpus=gpus)
            
            model.summary()
            metrics = ["binary_crossentropy", AUC(name='auc'), 'acc']
            opt = optimizers.Adam(lr=0.003, decay=0.0001)
            model.compile(optimizer=opt,
                     loss="binary_crossentropy",
                     metrics=metrics)
        
        model.fit(train_data, verbose=1, steps_per_epoch=steps_per_epoch, epochs=epochs,
                  #validation_data=test_data,
                  callbacks=callbacks,
                  shuffle=True)
        # model.save(modelfile, signatures=None)
        # print('--->saved model in {}'.format(modelfile))
    else:
        pass
