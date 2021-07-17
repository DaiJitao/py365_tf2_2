# encoding=utf8
import sys
import tensorflow as tf
import numpy as np


(x_train,y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(type(x_train))
print(x_train.shape)

x_train = x_train.astype(np.float32) / 255.
x_test = x_test.astype(np.float32) / 255.
print( x_train.shape )

batch_size = 128
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batch_size * 5 ).batch(batch_size)

test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.shuffle(batch_size*5).batch(batch_size)

h_dim = 20
encoder = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(256, activation=tf.nn.relu),
     tf.keras.layers.Dense(128, activation=tf.nn.relu),
     tf.keras.layers.Dense(h_dim)]
)

decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(784)
])

class AE(tf.keras.models.Model):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=None):
        h = self.encoder(inputs)
        x_hat = self.decoder(h)
        return x_hat

model = AE()
model.build(input_shape=(4, 784))
model.summary()
optimizer = tf.optimizers.Adadelta() # tf.optimizers.Adam(lr=0.001)

epochs = 100
for epoch in range(epochs):
    for step, x in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])

        with tf.GradientTape() as tape:
            x_rec_logits = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_mean(rec_loss)
        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 ==0:
            print(epoch, step, float(rec_loss))

tf.nn.softmax_cross_entropy_with_logits()
tf.nn.softmax_cross_entropy_with_logits_v2()

# 99 300 0.26982584595680237
# 99 400 0.2738988995552063
#