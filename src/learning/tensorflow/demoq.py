# encoding=utf8
import sys
import tensorflow as tf

#tf.gather

x = tf.constant(3,dtype=tf.float64)

with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
    dy_dx = g.gradient(y, x)
    print(dy_dx)
    
c = tf.nn.embedding_lookup(params=[[1,2,3]], ids=[0,2])
print(c)


# categorical_column_with_identity：把numerical data转乘one hot encoding
#特征数据
features = {
    'birthplace': [[1],[1],[3],[4]]
}
tf.feature_column.categorical_column_with_identity(key='birthplace', num_buckets=4, default_value=0)

pets = {'pets': [2,3,0,1]}  #猫0，狗1，兔子2，猪3

column = tf.feature_column.categorical_column_with_identity(
    key='pets',
    num_buckets=4)

indicator = tf.feature_column.indicator_column(column)


features = {
    'price':[[2.0],[30.0],[5.0],[100.0]]
}
item1_price=tf.feature_column.numeric_column("price")

value_city = '北京'.encode('utf-8')
value_use_day = 7
value_pay = 289.4
value_poi = [b'123', b'345', b'789']

bl_city = tf.train.BytesList(value=[value_city])
il_use_day = tf.train.Int64List(value=[value_use_day])
fl_pay = tf.train.FloatList(value=[value_pay])
bl_poi = tf.train.BytesList(value=value_poi)

'''
下面生成tf.train.Feature
'''
feature_city = tf.train.Feature(bytes_list = bl_city)
feature_use_day = tf.train.Feature(int64_list = il_use_day)
feature_pay = tf.train.Feature(float_list = fl_pay)

feature_poi = tf.train.Feature(bytes_list = bl_poi)
'''
下面定义tf.train.Features
'''
feature_dict = {"city":feature_city,"use_day":feature_use_day,"pay":feature_pay,"poi":feature_poi}
features = tf.train.Features(feature = feature_dict)
'''
下面定义tf.train.example
'''
example = tf.train.Example(features = features)

# 把若干个example组合起来，然后转化为二进制文件，就是tfrecord。

path = "./test.tfrecord"
with tf.io.TFRecordWriter(path) as file_writer:
  file_writer.write(example.SerializeToString())
  
data = tf.data.TFRecordDataset(path)
for tmp in data:
    print(tmp)
    
    
tf.data.experimental.make_csv_dataset()