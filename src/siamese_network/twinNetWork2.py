# encoding=utf8
import sys

'''
孪生网络，实现网络之间的参数独立和共享
参考网址：https://blog.csdn.net/qq_35826213/article/details/86313469
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import np_utils
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import np_utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# ---------------------函数功能区 参数独立-------------------------
def FeatureNetwork():
    """生成特征提取网络"""
    """这是根据，MNIST数据调整的网络结构，下面注释掉的部分是，原始的Matchnet网络中feature network结构"""
    inp = Input(shape = (28, 28, 1), name='FeatureNet_ImageInput')
    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)

    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    model = Model(inputs=inp, outputs=models)
    return model

def ClassiFilerNet():  # add classifier Net
    """生成度量网络和决策网络，其实maychnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)"""
    input1 = FeatureNetwork()                     # 孪生网络中的一个特征提取
    input2 = FeatureNetwork()                     # 孪生网络中的另一个特征提取
    for layer in input2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
        layer.name = layer.name + str("_2")
    inp1 = input1.input
    inp2 = input2.input
    merge_layers = concatenate([input1.output, input2.output])        # 进行融合，使用的是默认的sum，即简单的相加
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])
    return class_models

# ---------------------主调区-------------------------
matchnet = ClassiFilerNet()
matchnet.summary()  # 打印网络结构
plot_model(matchnet, to_file='test.png')  # 网络结构输出成png图片
# --------------------------------------------------------------------------------------------------------------------
# ------------ 以下实现参数共享 -------------
'''FeatureNetwork()的功能和上面的功能相同，为方便选择，在ClassiFilerNet()函数中加入了判断是否使用共享参数模型功能，令reuse=True，便使用的是共享参数的模型。
关键地方就在，只使用的一次Model，也就是说只创建了一次模型，虽然输入了两个输入，但其实使用的是同一个模型，因此权重共享的。
'''

# ----------------函数功能区-----------------------
def FeatureNetwork():
    """生成特征提取网络"""
    """这是根据，MNIST数据调整的网络结构，下面注释掉的部分是，原始的Matchnet网络中feature network结构"""
    inp = Input(shape = (28, 28, 1), name='FeatureNet_ImageInput')
    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)

    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    # models = Conv2D(64, kernel_size=(3, 3), strides=2, padding='valid')(models)
    # models = Activation('relu')(models)
    # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    model = Model(inputs=inp, outputs=models)
    return model

def ClassiFilerNet(reuse=False):  # add classifier Net
    """生成度量网络和决策网络，其实maychnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)"""

    if reuse:
        inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')
        models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
        models = Activation('relu')(models)
        models = MaxPool2D(pool_size=(3, 3))(models)

        models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
        # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
        models = Activation('relu')(models)

        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        models = Activation('relu')(models)

        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        models = Activation('relu')(models)

        # models = Conv2D(64, kernel_size=(3, 3), strides=2, padding='valid')(models)
        # models = Activation('relu')(models)
        # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
        models = Flatten()(models)
        models = Dense(512)(models)
        models = Activation('relu')(models)
        model = Model(inputs=inp, outputs=models)

        inp1 = Input(shape=(28, 28, 1))  # 创建输入
        inp2 = Input(shape=(28, 28, 1))  # 创建输入2
        model_1 = model(inp1)  # 孪生网络中的一个特征提取分支
        model_2 = model(inp2)  # 孪生网络中的另一个特征提取分支
        merge_layers = concatenate([model_1, model_2])  # 进行融合，使用的是默认的sum，即简单的相加

    else:
        input1 = FeatureNetwork()                     # 孪生网络中的一个特征提取
        input2 = FeatureNetwork()                     # 孪生网络中的另一个特征提取
        for layer in input2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
            layer.name = layer.name + str("_2")
        inp1 = input1.input
        inp2 = input2.input
        merge_layers = concatenate([input1.output, input2.output])        # 进行融合，使用的是默认的sum，即简单的相加
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])
    return class_models
