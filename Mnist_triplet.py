from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from keras import backend as K


# Parameters
training_epochs = 5
display_step = 1
batch_size = 128
width = 28
height = 28
#number of classes
nClass = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
#y_train = keras.utils.to_categorical(y_train, nClass)[:60000]
#y_test = keras.utils.to_categorical(y_test, nClass)

x_train_real = []
for i in range(int(60000)):
    img = x_train[i].reshape((28, 28))
    img = np.uint8(img * 255)
    img = cv.resize(img, (28, 28))
    img = img / 255
    x_train_real.append(img.reshape((28, 28, 1)))

x_test_real = []
for i in range(x_test.shape[0]):
  img = x_test[i].reshape((28, 28))
  img = np.uint8(img * 255)
  img = img / 255
  x_test_real.append(img.reshape((28, 28, 1)))

x_train_real = np.array(x_train_real)
x_test_real = np.array(x_test_real)

#MODEL

# Architecture

def inference(x):
    phase_train = tf.constant(True)
    x = tf.reshape(x, shape=[-1, height, width, 1])

    conv1 = tf.layers.conv2d(inputs=x, filters=32,  kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    norm1 = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    norm2 = tf.layers.batch_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)



    flat = tf.reshape(conv3, [-1, 7 * 7 * 32])
    fc_1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=fc_1, rate=0.4)

    embed = tf.layers.dense(inputs=dropout, units=128)

    output = tf.nn.l2_normalize(embed, dim=1, epsilon=1e-12, name=None)

    return output




def loss(output, labels):
    triplet = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, output, margin=1.0)
    loss = tf.reduce_mean(triplet, name='triplet')
    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 10, 0.96, staircase=True)
    optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op



def evaluate(output, y):
    logits = tf.nn.softmax(output)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("validation error", (1.0 - accuracy))
    return accuracy

x = tf.placeholder("float", [None, height, width, 1], name='placehold_x')
y = tf.placeholder('float', [None], name='placehold_y')

keep_prob = tf.placeholder(tf.float32, name='keep_prob') # dropout probability

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
training_dataset = tf.data.Dataset.from_tensor_slices((x_train_real, y_train))
validation_dataset = tf.data.Dataset.from_tensor_slices((x_test_real, y_test))
training_dataset = training_dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)


iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

"""Training"""
sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True)
#summary_writer = tf.summary.FileWriter("summary_logs/", graph_def=sess.graph_def)
init_op = tf.global_variables_initializer()
sess.run(init_op)


with tf.device('/cpu:0'):
    with sess.as_default():

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch_train = int(x_train.shape[0] / batch_size)
            sess.run(training_init_op)
            # Loop over all batches
            for i in range(total_batch_train):
                # Fit training using batch data
                batch_x, batch_y = sess.run(next_element)

                sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                # Compute average loss
                avg_cost = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
                #train_precision = sess.run(eval_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})

                if not i % 10:
                    print('Epoch #: ', epoch, 'global step', sess.run(global_step), '  Batch #: ', i, 'loss: ', avg_cost)
                          #'train error: ', (1 - train_precision))
            saver.save(sess, "model_logs/model-checkpoint", global_step=global_step)
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



### Predict
#
# img_addr = '/home/pavelkrolevets/Working/TF_facenet/data/VALIDATION_DIR/dog/dog.155.jpg'
# with tf.gfile.FastGFile(img_addr, 'rb') as f:
#     image_data = f.read()
# image = tf.image.decode_jpeg(image_data, channels=3)
# image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)
# image = tf.reshape(image, [-1, width, height, 3])
# pred_im = sess.run(image)
#
# prediction = sess.run(inference, feed_dict={x: pred_im, keep_prob: 1})
# print(prediction)