# coding:utf-8
from tensorflow.keras.layers import Input, Concatenate, Reshape, Subtract, Dense, Embedding, Dropout, RepeatVector, Add, Multiply, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import os


class MyFlatten(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MyFlatten, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        if mask == None:
            return mask
        return K.batch_flatten(mask)

    def call(self, inputs, mask=None):
        return K.batch_flatten(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))

    def get_config(self):
        base_config = super(MyFlatten, self).get_config()
        base_config.update({})
        return base_config


class MySumLayer(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MySumLayer, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            if K.ndim(x) != K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0, 2, 1])
            x = x * mask
            if K.ndim(x) == 2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)
        else:
            if K.ndim(x) == 2:
                x = K.expand_dims(x)
            return K.sum(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i != self.axis:
                output_shape.append(input_shape[i])
        if len(output_shape) == 1:
            output_shape.append(1)
        return tuple(output_shape)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super(MySumLayer, self).get_config()
        config.update(base_config)
        return config

class MyMeanPool(Layer):
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(MyMeanPool, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # need not to pass the mask to next layers
        return None

    def call(self, x, mask=None):
        if mask is not None:
            if K.ndim(x)!=K.ndim(mask):
                mask = K.repeat(mask, x.shape[-1])
                mask = tf.transpose(mask, [0,2,1])
            mask = K.cast(mask, K.floatx())
            x = x * mask
            return K.sum(x, axis=self.axis) / K.sum(mask, axis=self.axis)
        else:
            return K.mean(x, axis=self.axis)

    def compute_output_shape(self, input_shape):
        output_shape = []
        for i in range(len(input_shape)):
            if i!=self.axis:
                output_shape.append(input_shape[i])
        return tuple(output_shape)

    def get_config(self):
        base_config = super(MyMeanPool, self).get_config()
        config = {"axis": self.axis}
        base_config.update(config)
        return base_config

# 连续的特征；
numic_feat = ['view_count', 'like_count', 'reply_count', 'quality_score', 'rank_score_a', 'nlp_content_score',
              'nlp_cover_score', 'nlp_effect_score', 'nlp_quality_score', 'nlp_rare_score', 'nlp_time_score',
              'base_weight', 'title_query_sim']

#  连续型
in_view_count = Input(shape=[1], name="view_count")  # None*1
in_like_count = Input(shape=[1], name="like_count")  # None*1
in_reply_count = Input(shape=[1], name="reply_count")  # None*1
in_quality_score = Input(shape=[1], name="quality_score")  # None*1
in_rank_score_a = Input(shape=[1], name="rank_score_a")  # None*1
in_nlp_content_score = Input(shape=[1], name="nlp_content_score")  # None*1
in_nlp_cover_score = Input(shape=[1], name="nlp_cover_score")  # None*1
in_nlp_effect_score = Input(shape=[1], name="nlp_effect_score")  # None*1
in_nlp_quality_score = Input(shape=[1], name="nlp_quality_score")  # None*1
in_nlp_rare_score = Input(shape=[1], name="nlp_rare_score")  # None*1
in_nlp_time_score = Input(shape=[1], name="nlp_time_score")  # None*1
in_base_weight = Input(shape=[1], name="base_weight")  # None*1
in_title_query_sim = Input(shape=[1], name="title_query_sim")  # None*1
numic_feat_inputs = [in_view_count, in_like_count, in_reply_count, in_quality_score, in_rank_score_a, in_nlp_content_score, in_nlp_cover_score, in_nlp_effect_score, in_nlp_quality_score, in_nlp_rare_score, in_nlp_time_score, in_base_weight, in_title_query_sim]

# 单值离散型
in_gender = Input(shape=[1], name="gender")  # None*1
in_age = Input(shape=[1], name="age")  # None*1
# 多值离散型
in_query_keyword = Input(shape=[query_keyword_size], name="query_keyword") # None*size, 最长长度query_keyword.size
in_title = Input(shape=[title_size], name="title") # None*size, 最长长度title.size
in_car_brand_ids = Input(shape=[car_brand_ids_size], name="car_brand_ids") # None*size, 最长长度car_brand_ids.size
in_cms_series_ids = Input(shape=[cms_series_ids_size], name="cms_series_ids") # None*size, 最长长度cms_series_ids.size
in_cms_spec_ids = Input(shape=[cms_spec_ids_size], name="cms_spec_ids") # None*size, 最长长度cms_spec_ids.size
multi_value_inputs = [in_query_keyword, in_title, in_car_brand_ids, in_cms_series_ids, in_cms_spec_ids]
# bert向量
in_title_bert_vec = Input(shape=[3], name="titleBertVec") # 直接输入到deep部分

'''First Order Embeddings'''
# 连续型
firstInputs = numic_feat_inputs
numeric = Concatenate()(firstInputs)  # None*len(firstInputs)
dense_numeric = Dense(1)(numeric)  # None*1
emb_gender_1d = Reshape([1])(Embedding(input_dim=3, output_dim=1)(in_gender))  # None*1, 性别取值3种;
emb_age_1d = Reshape([1])(Embedding(input_dim=10, output_dim=1)(in_age))  # None*1, 年龄取值10种
# 多值离散型
emb_query_keyword_1d = Embedding(11, 1, mask_zero=True)(in_query_keyword) # None*3*1
emb_query_keyword_1d = MyMeanPool(axis=1)(emb_query_keyword_1d) # None*1
multi_value_1d = [].append(emb_query_keyword_1d)

'''compute'''
first_order_inputs = [dense_numeric, emb_gender_1d, emb_age_1d, emb_interest_1d]
y_first_order = Add()(first_order_inputs)  # None*1
# --------------------------- 第一阶完毕  ----------------------------------------------------
latent = 8
'''Second Order Embeddings'''
# 连续
view_count_Kd = RepeatVector(1)(Dense(latent)(in_view_count))  # None * 1 * K
like_count_Kd = RepeatVector(1)(Dense(latent)(in_like_count))  # None * 1 * K
reply_count_Kd = RepeatVector(1)(Dense(latent)(in_reply_count))  # None * 1 * K
quality_score_Kd = RepeatVector(1)(Dense(latent)(in_quality_score))  # None * 1 * K
rank_score_a_Kd = RepeatVector(1)(Dense(latent)(in_rank_score_a))  # None * 1 * K
nlp_content_score_Kd = RepeatVector(1)(Dense(latent)(in_nlp_content_score))  # None * 1 * K
nlp_cover_score_Kd = RepeatVector(1)(Dense(latent)(in_nlp_cover_score))  # None * 1 * K
nlp_effect_score_Kd = RepeatVector(1)(Dense(latent)(in_nlp_effect_score))  # None * 1 * K
nlp_quality_score_Kd = RepeatVector(1)(Dense(latent)(in_nlp_quality_score))  # None * 1 * K
nlp_rare_score_Kd = RepeatVector(1)(Dense(latent)(in_nlp_rare_score))  # None * 1 * K
nlp_time_score_Kd = RepeatVector(1)(Dense(latent)(in_nlp_time_score))  # None * 1 * K
base_weight_Kd = RepeatVector(1)(Dense(latent)(in_base_weight))  # None * 1 * K
title_query_sim_Kd = RepeatVector(1)(Dense(latent)(in_title_query_sim))  # None * 1 * K
numic_feat_second = [view_count_Kd, like_count_Kd, reply_count_Kd, quality_score_Kd, rank_score_a_Kd, nlp_content_score_Kd, nlp_cover_score_Kd, nlp_effect_score_Kd, nlp_quality_score_Kd, nlp_rare_score_Kd, nlp_time_score_Kd, base_weight_Kd, title_query_sim_Kd]
# 单值离散型
emb_gender_Kd = Embedding(3, latent)(in_gender)  # None * 1 * K
emb_age_Kd = Embedding(10, latent)(in_age)  # None * 1 * K
# 多值离散型
emb_interest_Kd = Embedding(11, latent, mask_zero=True)(in_interest) # (None , 3 , K)
emb_interest_Kd = RepeatVector(1)(MyMeanPool(axis=1)(emb_interest_Kd)) #( None , 1 , K)

second_order_inputs = numic_feat_second + [emb_gender_Kd, emb_age_Kd, emb_interest_Kd]
emb = Concatenate(axis=1)(second_order_inputs)  # None * 4 * K

'''compute'''
summed_features_emb = MySumLayer(axis=1)(emb)  # None * K
summed_features_emb_square = Multiply()([summed_features_emb, summed_features_emb])  # None * K

squared_features_emb = Multiply()([emb, emb])  # None * 9 * K
squared_sum_features_emb = MySumLayer(axis=1)(squared_features_emb)  # Non * K

sub = Subtract()([summed_features_emb_square, squared_sum_features_emb])  # None * K
sub = Lambda(lambda x: x * 0.5)(sub)  # None * K

y_second_order = MySumLayer(axis=1)(sub)  # None * 1

'''deep parts'''
y_deep = Flatten()(emb) # MyFlatten()(emb)  # None*(6*K)

y_deep_bert = Concatenate(axis=1, name="ConcatEmbBert")([y_deep, in_title_bert_vec])
print("y_deep_bert",y_deep_bert)
y_deep_bert = Dropout(0.5)(Dense(128, activation='relu', name="Dense1")(y_deep_bert))
y_deep_bert = Dropout(0.5)(Dense(64, activation='relu', name="Dense2")(y_deep_bert))
y_deep_bert = Dropout(0.5)(Dense(32, activation='relu', name="Dense3")(y_deep_bert))
y_deep_bert = Dropout(0.5)(Dense(1, activation='relu', name="Dense4")(y_deep_bert))

'''deepFM'''
y = Concatenate(axis=1)([y_first_order, y_second_order, y_deep_bert])
deepfm_out = Dense(1, activation='sigmoid')(y)


def gen_data_demo():
    scores = np.array([0.2, 0.3, 0.3])
    sales = np.array([0.2, 0.3, 0.9])
    genders = np.array([0, 0, 1])
    ages = np.array([1, 2, 7])
    interest = np.array([[1, 0, 1], [1,1,0],[1,0,0]])
    y_ = np.array([1, 1, 0])

    while True:
        titleBertVec = np.random.random((3, 3))

        x_train = {"score": scores, "sales": sales, "gender": genders, "age": ages, "titleBertVec": titleBertVec,
                   "interest": interest}
        yield (x_train, y_)


# GPU
use_gpu ='1,2'
# os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu
devices = []
if use_gpu == '':
    gpus = 0
    print('--->use gpus:', gpus)
else:
    gpus = len(use_gpu.strip().split(','))
    for i in use_gpu.split(','):
        devices.append('/gpu:{}'.format(i))

print('--->devices:', devices,',gpus:', gpus)

def train_generator(gen_data, isTrain=False, isPredict=False):
    mf = "deepfm.flow.v0.0.1.h5"
    epochs = 5
    all_size = 10
    train_batch_size = 3
    steps_per_epoch = int(all_size / train_batch_size)
    # numic

    inputs = numic_feat_inputs + [in_gender, in_age, in_title_bert_vec, in_interest]
    model = Model(inputs=inputs, outputs=[deepfm_out])
    # strategy = tf.distribute.MirroredStrategy(devices=devices)
    # with strategy.scope():
    if True:
        if gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'accuracy'], )
        if isTrain:
            model.fit_generator(gen_data, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2, )
            model.save(mf, save_format='h5')
            print("-->ok saved model in {}".format(mf))
            # model.save_weights('my_model.h5', save_format='h5')
            # json_model = model.to_yaml()
            # print(json_model)
            # model.save_weights("weights/model")
    if isPredict:
        predict(mf)

def predict(mf):
    cus_objs = {"MyFlatten": MyFlatten, "MySumLayer": MySumLayer, "MyMeanPool": MyMeanPool}
    model = load_model(mf, custom_objects=cus_objs)
    model.summary()

if __name__ == '__main__':
    gen_data = gen_data_demo()
    #train(title_bert_vec=titleBertVec)
    train_generator(gen_data=gen_data, isTrain=True, isPredict=True)
