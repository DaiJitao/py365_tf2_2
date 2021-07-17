# coding:utf-8
from tensorflow.keras.layers import Input, Concatenate, Reshape, Subtract, Dense, Embedding, Dropout, RepeatVector, Add, Multiply, Lambda, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


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

#  连续型
in_score = Input(shape=[1], name="score")  # None*1
in_sales = Input(shape=[1], name="sales")  # None*1
print("in_sales",in_sales)
# 单值离散型
in_gender = Input(shape=[1], name="gender")  # None*1
in_age = Input(shape=[1], name="age")  # None*1
# 多值离散型
in_interest = Input(shape=[20], name="interest") # None*3, 最长长度3

# bert向量
in_title_bert_vec = Input(shape=[3], name="titleBertVec")

'''First Order Embeddings'''
numeric = Concatenate()([in_score, in_sales])  # None*2
dense_numeric = Dense(1)(numeric)  # None*1
emb_gender_1d = Reshape([1])(Embedding(input_dim=3, output_dim=1)(in_gender))  # None*1, 性别取值3种;
emb_age_1d = Reshape([1])(Embedding(input_dim=10, output_dim=1)(in_age))  # None*1, 年龄取值10种

emb_interest_1d = Embedding(11, 1, mask_zero=True)(in_interest) # None*3*1
emb_interest_1d = MyMeanPool(axis=1)(emb_interest_1d) # None*1

'''compute'''
first_order_inputs = [dense_numeric, emb_gender_1d, emb_age_1d, emb_interest_1d]
y_first_order = Add()(first_order_inputs)  # None*1

latent = 8
'''Second Order Embeddings'''
emb_score_Kd = RepeatVector(1)(Dense(latent)(in_score))  # None * 1 * K
emb_sales_Kd = RepeatVector(1)(Dense(latent)(in_sales))  # None * 1 * K
emb_gender_Kd = Embedding(3, latent)(in_gender)  # None * 1 * K
emb_age_Kd = Embedding(10, latent)(in_age)  # None * 1 * K

emb_interest_Kd = Embedding(11, latent, mask_zero=True)(in_interest) # (None , 3 , K)
emb_interest_Kd = RepeatVector(1)(MyMeanPool(axis=1)(emb_interest_Kd)) #( None , 1 , K)

second_order_inputs = [emb_score_Kd, emb_sales_Kd, emb_gender_Kd, emb_age_Kd, emb_interest_Kd]
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
y = Dense(1, activation='sigmoid')(y)

def train(title_bert_vec, train=False):
    model = Model(inputs=
                  [in_score, in_sales, in_gender, in_age, in_title_bert_vec, in_interest],
                  outputs=[y])
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'accuracy'], )
    # model.summary()
    scores = np.array([0.2, 0.3, 0.3])
    sales = np.array([0.2, 0.3, 0.9])
    genders = np.array([0, 0, 1])
    ages = np.array([1, 2, 7])
    interest = np.array([1,0,1])

    y_ = np.array([1, 1, 0])

    x_train = {"score": scores, "sales": sales, "gender": genders, "age": ages, "titleBertVec": title_bert_vec,
               "interest":interest}
    if train:
        model.fit(x=x_train, y=y_, epochs=10, verbose=2)
        # model.save_weights('my_model.h5', save_format='h5')
        json_model = model.to_yaml()
        # print(json_model)
        #model.save_weights("weights/model")
        model.save("test.demo.h5")

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


def train_generator(gen_data, isTrain=False, isPredict=False):
    mf = "deefm.flow.v0.0.1.h5"
    epochs = 5
    all_size = 10
    train_batch_size = 3
    steps_per_epoch = int(all_size / train_batch_size)
    model = Model(inputs=
                  [in_score, in_sales, in_gender, in_age, in_title_bert_vec, in_interest],
                  outputs=[y])
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'accuracy'], )
    # model.summary()

    if isTrain:
        model.fit_generator(gen_data, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=2,)
        # model.save_weights('my_model.h5', save_format='h5')
        json_model = model.to_yaml()
        # print(json_model)
        #model.save_weights("weights/model")
        model.save(mf)
        print("-->ok saved model in {}".format(mf))

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
