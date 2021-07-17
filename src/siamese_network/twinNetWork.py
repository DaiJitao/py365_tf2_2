

'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for more details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU

该代码实现孪生网络，两个网络之间实现了参数共享；
'''
from __future__ import absolute_import  # absolute:绝对引入，引入系统的标准模块，就是引入自带的模块
from __future__ import print_function  # future就是把下一个版本的特性引入到当前版本中
import numpy as np  # 引入np用于数值计算

import random  # random可返回随机生成的一个实数，范围是[0,1)
from tensorflow.keras.datasets import mnist  # 引入mnist数据库
from tensorflow.keras.models import Model  # 用于搭建模型、模型实例化和训练等过程中的相关操作
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, \
    Lambda  # 看名字就知道这是模型中的常见操作，为了简化后续程序，在这里引入，也可以直接在搭建中写keras.layers.Dense()等，注意Lambda层比较特殊，这是自定义操作层，你可随意发挥
from tensorflow.keras.optimizers import RMSprop  # 采用RMSprop进行优化，再次将优化算法引入了
from tensorflow.keras import backend as K  # 方便一些简单的数学操作，个人感觉应该是可以更好的处理过程中的张量间的操作
from tensorflow.keras.utils import plot_model
''' 本文件实现了共享参数的孪生网络 '''

num_classes = 10  # 数据库中图像的类别数为10
epochs = 20  # 就是进行20轮数据更新每一轮中输入batch_size个样本


# 下面定义了好多函数，很Keras，任务分解，每个任务都整成小模块，最后拼起来，API思想很深入吧
def euclidean_distance(vects):  # 定义欧式距离函数，输入是Vector，注意输入中包含两个空间点，如何存储的？需要考虑下
    x, y = vects  # 拆开，为什么这样就能将两个点拆开了？
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)  # 平方差求和，这就是孪生网络中两个子网络合并的第一步，求和？
    return K.sqrt(K.maximum(sum_square, K.epsilon()))  # 返回2范数和epsilon中的最大值


def eucl_dist_output_shape(shapes):  # 欧式距离输出形状函数，返回shape1的第一维大小
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):  # 对比损失，输入是图像的真实标签y_true网络的预测标签y_pred(图像对是否是同一类，是同一类的概率)
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)  # 这一块要结合论文细看一下损失函数的定义原理？？？


def create_pairs(x, digit_indices):  # 构造对，交替生成正负样本对，输入x图像数据    及每类的索引标签
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []  # 存储样本对
    labels = []  # 存储样本对对应的标签
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 中括号里面计算每一类的样本数，n是最小的样本数-1
    for d in range(num_classes):  # 对每一类进行操作
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]  # 和同类构成正样本对标签为1
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]  # 和不同的类构成负样本对标签为0
            labels += [1, 0]
    return np.array(pairs), np.array(labels)  # 返回样本对和标签


def create_base_network(input_shape):  # 构造网络共享网络进行特征提取，输入是输入图像的shape
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred):  # 计算准确率函数，输入：y_true真实样本对标签，y_pred预测样本对标签
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5  # 将预测标签中概率小于0.5的元素设置为1其他元素为0 ？？感觉应该是>
    return np.mean(pred == y_true)  # 预测与真实标签对比，统计相同元素的个数,并计算识别率


def accuracy(y_true, y_pred):  # 准确率函数，输入是预测标签和真实标签
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))  # K.cast()数据类型转换，统计相同元素个数并计算识别率


# the data, split between train and test sets 训练和测试数据的划分
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 加载数据集
x_train = x_train.astype('float32')  # 类型转换
x_test = x_test.astype('float32')
x_train /= 255  # 归一化0-1
x_test /= 255
input_shape = x_train.shape[1:]  # 网络输入形状例如图像的shape=[1000,250,250,3]则input_shape = [250,250,3]
# create training+test positive and negative pairs构造训练和测试的正负样本对
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)
digit_indices = [np.where(y_test == i)[0] for i in
                 range(num_classes)]  # np.where()输出满足条件元素的坐标，此处返回的是                                  0-9每个数字图片的索引值
te_pairs, te_y = create_pairs(x_test, digit_indices)  # 构造测试对及标签
# network definition 定义网络
base_network = create_base_network(input_shape)  # 基本网络Model
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)  # 网络的两个输入
# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
print(type(base_network))
processed_a = base_network(input_a)  # 两个共享之路的输出
processed_b = base_network(input_b)
distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])  # 自定义层计算距离，注意此处API
model = Model([input_a, input_b], distance)  # 实例化，输入是a,b输出是distance
# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])  # 编译

model.summary()

plot_model(model, to_file='model.png')