from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn import preprocessing
import pandas as pd



# Parameters
training_epochs = 5
display_step = 1
batch_size = 128

x_train = np.load('./data/X_train.npy')

"""creating embeddings for labels"""
data = pd.read_csv('./data/train.csv')
labels = data['Id']
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(labels)

#MODEL
# Architecture

def inference(x):
    phase_train = tf.constant(True)
    x = tf.reshape(x, shape=[-1, 100, 100, 1])

    conv1 = tf.layers.conv2d(inputs=x, filters=32,  kernel_size=[7, 7], padding="same", activation=tf.nn.relu)
    norm1 = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

    conv2a = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv2a, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    norm2 = tf.layers.batch_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

    conv3a = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv3a, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    norm3 = tf.layers.batch_normalization(conv3)

    conv4a = tf.layers.conv2d(inputs=norm3, filters=32, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv4a, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    norm4 = tf.layers.batch_normalization(conv4)
    pool3 = tf.layers.max_pooling2d(inputs=norm4, pool_size=[2, 2], strides=2)


    conv5 = tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

    flat = tf.reshape(pool4, [-1, 6 * 6 * 64])
    fc_1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)
    drop1 = tf.layers.dropout(fc_1)
    fc_2 = tf.layers.dense(inputs=drop1, units=128, activation=tf.nn.relu)
    drop2 = tf.layers.dropout(fc_2)
    embed = tf.layers.dense(inputs=drop2, units=128)

    output = tf.nn.l2_normalize(embed, dim=1, epsilon=1e-12, name=None)

    return output



def loss(output, labels):
    triplet = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, output, margin=1.0)
    loss = tf.reduce_mean(triplet, name='triplet')
    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.9, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

x = tf.placeholder("float", [None, 100, 100, 1], name='placehold_x')
y = tf.placeholder('float', [None], name='placehold_y')

output = inference(x)
tf.identity(output, name="inference")

cost = loss(output, y)
global_step = tf.Variable(0, name='global_step', trainable=False)

# Passing global_step to minimize() will increment it at each step.

train_op = training(cost, global_step)
# eval_op = evaluate(output, y)
#summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

"""Making iterator"""
features_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
labels_placeholder = tf.placeholder(y_train.dtype, y_train.shape)

training_dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
batched_dataset = training_dataset.batch(batch_size)

training_init_op = batched_dataset.make_initializable_iterator()
next_element = training_init_op.get_next()

"""Training"""
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)


with tf.device('/cpu:0'):
    with sess.as_default():

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch_train = int(x_train.shape[0] / batch_size)
            sess.run(training_init_op.initializer, feed_dict={features_placeholder: x_train,
                                                      labels_placeholder: y_train})
            # Loop over all batches
            for i in range(total_batch_train):
                # Fit training using batch data

                batch_x, batch_y = sess.run(next_element)

                sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                #train_precision = sess.run(eval_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})

                if not i % 10:
                    print('Epoch #: ', epoch, 'global step', sess.run(global_step), '  Batch #: ', i, 'loss: ', avg_cost)
                          #'train error: ', (1 - train_precision))
            saver.save(sess, "model_logs/model-checkpoint", global_step=global_step, write_meta_graph=True)
                #summary_str = sess.run(summary_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
                #summary_writer.add_summary(summary_str, sess.run(global_step))

                # Display logs per epoch step
            # if epoch % display_step == 0:
            #
            #     sess.run(validation_init_op)
            #     total_batch_eval = int(x_test.shape[0] / batch_size)
            #     for i in range(total_batch_eval):
            #         batch_x_val, batch_y_val = sess.run(next_element)
            #         error = sess.run(eval_op,
            #                             feed_dict={x: batch_x_val, y: batch_y_val, keep_prob: 1})
            #         error += error
            #     print("Validation Error:", (1 - error/batch_size))
            #
            #     #saver.save(sess, "model_logs/model-checkpoint", global_step=global_step)


        print ("Optimization Finished!")


