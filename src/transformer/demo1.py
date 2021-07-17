# encoding=utf8
import sys

from transformers import pipeline
from tensorflow.keras.callbacks import ModelCheckpoint


classifier = pipeline('ner')
res = classifier('毛泽东是中国共产党的缔造者')
print(res)

ModelCheckpoint(filepath='./', monitor='mae', mode='min', save_best_only=True, period=1)

savedir='model'
filepath="qt_sim_model_{epoch:02d}-{mae:.2f}.h5"
modelfile=os.path.join(savedir, filepath)
checkpoint = ModelCheckpoint(modelfile,monitor='mae',mode='min',verbose=1,save_best_only=True,period=1)

