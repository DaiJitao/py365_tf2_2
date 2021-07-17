# encoding=utf8
import sys

import os
import tensorflow as tf
import shutil
import tensorflow as tf

data_dir = './data'
checkpoint_dir = os.path.join(data_dir, 'checkpoints')

def download_and_read(urls):
    texts = []
    for i, url in enumerate(urls):
        p = tf.keras.utils.get_file("")
