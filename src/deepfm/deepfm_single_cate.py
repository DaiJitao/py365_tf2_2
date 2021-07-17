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



# numeric fields; 连续型
in_score = Input(shape=[1], name="score")  # None*1 # Tensor("score:0", shape=(None, 1), dtype=float32)
in_sales = Input(shape=[1], name="sales")  # None*1
print("in_sales",in_sales)
# single value categorical fields
in_gender = Input(shape=[1], name="gender")  # None*1
in_age = Input(shape=[1], name="age")  # None*1

# bert向量
in_title_bert_vec = Input(shape=[3], name="titleBertVec")
print(in_title_bert_vec)

'''First Order Embeddings'''
numeric = Concatenate()([in_score, in_sales])  # None*2
print("numeric",numeric)
dense_numeric = Dense(1)(numeric)  # None*1
emb_gender_1d = Reshape([1])(Embedding(input_dim=3, output_dim=1)(in_gender))  # None*1, 性别取值3种;
emb_age_1d = Reshape([1])(Embedding(input_dim=10, output_dim=1)(in_age))  # None*1, 年龄取值10种

'''compute'''
y_first_order = Add()([dense_numeric, emb_gender_1d, emb_age_1d])  # None*1

latent = 8
'''Second Order Embeddings'''
emb_score_Kd = RepeatVector(1)(Dense(latent)(in_score))  # None * 1 * K
emb_sales_Kd = RepeatVector(1)(Dense(latent)(in_sales))  # None * 1 * K
emb_gender_Kd = Embedding(3, latent)(in_gender)  # None * 1 * K
emb_age_Kd = Embedding(10, latent)(in_age)  # None * 1 * K

emb = Concatenate(axis=1)([emb_score_Kd, emb_sales_Kd, emb_gender_Kd, emb_age_Kd])  # None * 4 * K
print("emb",emb)
'''compute'''
summed_features_emb = MySumLayer(axis=1)(emb)  # None * K
summed_features_emb_square = Multiply()([summed_features_emb, summed_features_emb])  # None * K

squared_features_emb = Multiply()([emb, emb])  # None * 9 * K
squared_sum_features_emb = MySumLayer(axis=1)(squared_features_emb)  # Non * K

sub = Subtract()([summed_features_emb_square, squared_sum_features_emb])  # None * K
sub = Lambda(lambda x: x * 0.5)(sub)  # None * K

y_second_order = MySumLayer(axis=1)(sub)  # None * 1

'''deep parts'''
print("emb", emb)

print(in_title_bert_vec)
y_deep = Flatten()(emb) # MyFlatten()(emb)  # None*(6*K)
print("y_deep",y_deep)
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
    model = Model(inputs=[in_score, in_sales, in_gender, in_age, in_title_bert_vec], outputs=[y])
    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'accuracy'], )
    model.summary()
    scores = np.array([0.2, 0.3, 0.3])
    sales = np.array([0.2, 0.3, 0.9])
    genders = np.array([0, 0, 1])
    ages = np.array([1, 2, 7])
    y_ = np.array([1, 1, 0])

    x_train = {"score": scores, "sales": sales, "gender": genders, "age": ages, "titleBertVec": title_bert_vec}
    if train:
        model.fit(x=x_train, y=y_, epochs=10, verbose=2)
        model.save_weights('my_model.h5', save_format='h5')
        json_model = model.to_yaml()
        print(json_model)
        #model.save_weights("weights/model")
        #model.save("test.demo.h5")


def predict():
    cus_objs = {"MyFlatten": MyFlatten, "MySumLayer": MySumLayer}
    model = load_model("test.demo.h5", custom_objects={"MyFlatten": MyFlatten, "MySumLayer": MySumLayer})
    model.summary()


import numpy as np

if __name__ == '__main__':
    titleBertVec = np.random.random((3, 3))
    #train(title_bert_vec=titleBertVec)
    train(titleBertVec,True)

    # train()
    # predict()
    # train()

# plot_model(model, 'model.png', show_shapes=True)
