# encoding=utf8
import sys

#！-*- coding: utf-8 -*-
import sys
import os
import pdb
import time
import datetime
import json
import math
import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import uniout, open
from keras.models import Model

maxlen = 100
batch_size = 100

model_dir = './chinese_simbert_L-12_H-768_A-12'
raw_data_file = sys.argv[1]
train_data_feature_file = sys.argv[2]

# bert配置
config_path = os.path.join(model_dir, 'bert_config.json')
checkpoint_path = os.path.join(model_dir, 'bert_model.ckpt')
dict_path = os.path.join(model_dir, 'vocab.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
size = 0

def trian(batch_data):
  start_time = time.time()
  try:
      a_token_ids, object_uids,long_titles = [], [], []
      for i, d in enumerate(batch_data):
          object_uid = d[0]
          long_title = d[1]
          token_ids = tokenizer.encode(long_title, max_length=maxlen)[0]
          a_token_ids.append(token_ids)
          object_uids.append(object_uid)
          long_titles.append(long_title)
      a_token_ids_arr = sequence_padding(a_token_ids)
      a_vecs = encoder.predict([a_token_ids_arr, np.zeros_like(a_token_ids_arr)],verbose=True)
      a_vecs = a_vecs / (a_vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
      for j, a_vec in enumerate(a_vecs):
          #print("{},{}".format(object_uids[j],long_titles[j]))
          train_data_feature.write(object_uids[j] + '\t' + long_titles[j] + '\t' + json.dumps(a_vec.tolist()) + '\n')
  except Exception as e:
      print('Exception:{}'.format(e))
  end_time = time.time()
  print('end time:{}'.format(end_time - start_time))



print('extract feature')
train_data_feature = open(train_data_feature_file, 'w', encoding='utf-8')

for index, line in enumerate(open(raw_data_file,'r')):
  size += 1
print("总共:{}".format(size))


D = []
with open(raw_data_file, encoding='utf-8') as f:
  for index, l in enumerate(f):
      try:
          l_split = l.strip().split('^')
          if len(l_split) != 3:
              continue
          object_uid = l_split[0]
          long_title = l_split[1]
          if long_title in ['NULL', '']:
              print("skip {}".format(l))
              continue
          D.append((object_uid,long_title))
          if len(D) == batch_size:
              print("处理:{}/{} flush".format(index+1,size))
              trian(D)
              train_data_feature.flush()
              D = []
      except Exception as e:
          print('load data Exception:{}'.format(e))
          continue
  print("处理:{}/{} flush".format(index+1,size))
  trian(D)
train_data_feature.close()
