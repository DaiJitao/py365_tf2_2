# 文本分类实验

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dropout, Dense
from src.transformer_2.Encoder import *
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.utils import plot_model
from genDataUtil import getData
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import os
from genDataUtil import diff_max, title_max

os.environ["CUDA_VISIBLE_DEVICES"]=''

# 1. 数据信息
max_features = 219990  # 20000
maxlen = diff_max + title_max
batch_size = 2048
epochs = 20
train = True
predict = False

print('Loading data...')
print("maxlen:", maxlen)
# (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=max_features)
train_data = "data/train_data_template.csv"
print("input data:"+train_data)
test_data = "data/test_data_template.csv"
modelfile = 'transformer.qt.h5'
(x_train, y_true) = getData(train_data)
# (x_train, y_true) = getData(test_data)

y_train = pd.get_dummies(y_true)
print(len(x_train), 'train sequences')
print('x_train shape:', x_train.shape)

# 2. 构造模型，及训练模型
inputs_diff = Input(shape=(maxlen,), dtype='int32', name='diff')
embeddings = Embedding(max_features, 128)(inputs_diff)
print("embeddings:")
print(embeddings)
mask_inputs = padding_mask(inputs_diff)
out_seq = Encoder(n_layers=2, d_model=128, num_heads=4, middle_units=256, max_seq_len=maxlen)([embeddings, mask_inputs])

print("out_seq:")
print(out_seq)

out_seq = GlobalAveragePooling1D()(out_seq)
print("out_seq:")
print(out_seq)

out_seq = Dropout(0.3)(out_seq)
out_seq = Dropout(0.3)(out_seq)
out_seq = Dropout(0.3)(out_seq)
outputs = Dense(2, activation='softmax')(out_seq)

model = Model(inputs=inputs_diff, outputs=outputs)
# print(model.summary())

# plot_model(model, to_file='transformer_model.png')

opt = Adam(lr=0.0002, decay=0.00001)
loss = 'categorical_crossentropy'
model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

if train:
    print('Train...')
    es = EarlyStopping(monitor='binary_crossentropy')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, callbacks=[es])
    model.save(modelfile, save_format='h5')
    print("--->ok saved in {}".format(modelfile))

if predict:
    print("Predict...")
    cus_objs = {"Encoder":Encoder} #,"EncoderLayer":EncoderLayer,"LayerNormalization":LayerNormalization,"MultiHeadAttention":MultiHeadAttention,"PositionalEncoding":PositionalEncoding}
    model = load_model(modelfile, custom_objects=cus_objs)
    # model.summary()
    res = model.predict(x_train)
    res = tf.argmax(res, axis=1).numpy() # tf.argmax(y_train, axis=1)
    if len(res) > 1:
        acc = accuracy_score(y_true, res)
        auc = roc_auc_score(y_true, res)
        print("accuracy:{},auc:{}".format(acc, auc))
    else:
        print("预测结果：{}".format(res))
