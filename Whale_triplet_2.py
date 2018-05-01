from __future__ import print_function
import tensorflow as tf
#from keras.datasets import mnist
import numpy as np
#from keras import backend as K
from tensorflow.python import debug as tf_debug
import argparse
import sys
from sklearn import preprocessing
import pandas as pd

# Parameters
training_epochs = 50
display_step = 1
batch_size = 400
margin = 1.5




def main(_):
#MODEL
    # Architecture
    def inference(x):
        phase_train = tf.constant(True)
        x = tf.reshape(x, shape=[-1, 100, 100, 1])

        conv1 = tf.layers.conv2d(inputs=x, filters=64,  kernel_size=[7, 7], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)
        norm1 = tf.layers.batch_normalization(conv1)
        pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

        conv2a = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[1, 1], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)
        conv2 = tf.layers.conv2d(inputs=conv2a, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)
        norm2 = tf.layers.batch_normalization(conv2)
        pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

        conv3a = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[1, 1], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)
        conv3 = tf.layers.conv2d(inputs=conv3a, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)

        conv4a = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[1, 1], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)
        conv4 = tf.layers.conv2d(inputs=conv4a, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)

        conv5a = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[1, 1], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)
        conv5 = tf.layers.conv2d(inputs=conv5a, filters=384, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)

        pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2)

        conv6a = tf.layers.conv2d(inputs=pool3, filters=384, kernel_size=[1, 1], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)
        conv6 = tf.layers.conv2d(inputs=conv6a, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal)

        pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

        flat = tf.reshape(pool4, [-1, 6 * 6 * 256])

        fc_1 = tf.layers.dense(inputs=flat, units=128)

        embed = tf.layers.dense(inputs=fc_1, units=128)

        output = tf.nn.l2_normalize(embed, axis=1, epsilon=1e-12, name='output')

        return output

    def loss(output, labels):
        triplet = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, output, margin=margin)
        loss = tf.reduce_mean(triplet, name='triplet')
        return loss

    def training(cost, global_step):
        tf.summary.scalar("cost", cost)
        learning_rate = tf.train.exponential_decay(0.001, global_step, 100, 0.9, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step=global_step)
        #gradients, variables = zip(*optimizer.compute_gradients(cost))
        #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        #train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op


    x_train = np.load('./data/X_train.npy')

    """creating embeddings for labels"""
    data = pd.read_csv('./data/train.csv')
    # dropping "new_whales"
    data = data[data.Id != "new_whale"]

    labels = data['Id']
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(labels)
    classes = len(pd.unique(labels))
    print("Number of classes= ", classes)

    x = tf.placeholder( tf.float32, [None, 100, 100, 1], name='placehold_x')
    y = tf.placeholder( tf.int32, [None], name='placehold_y')

    output = inference(x)
    tf.identity(output, name="inference")

    cost = loss(output, y)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step)
    saver = tf.train.Saver()

    """Making iterator"""
    features_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
    labels_placeholder = tf.placeholder(y_train.dtype, y_train.shape)

    training_dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    batched_dataset = training_dataset.batch(batch_size)

    training_init_op = batched_dataset.make_initializable_iterator()
    next_element = training_init_op.get_next()

    
    """Training"""
    sess = tf.InteractiveSession()

    if FLAGS.debug and FLAGS.tensorboard_debug_address:
        raise ValueError(
            "The --debug and --tensorboard_debug_address flags are mutually "
            "exclusive.")
    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    elif FLAGS.tensorboard_debug_address:
        sess = tf_debug.TensorBoardDebugWrapperSession(
            sess, FLAGS.tensorboard_debug_address)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for d in ['/device:GPU:0', '/device:GPU:1']:
        with tf.device(d):
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
                        #batch_x = np.reshape(batch_x, (-1, 100,100,1))
                        sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
                        # Compute average loss
                        avg_cost = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                        if not i % 10:
                            print('Epoch #: ', epoch, 'global step', sess.run(global_step), '  Batch #: ', i, 'loss: ', avg_cost)
                    
                    if not epoch % 10:           
                        saver.save(sess, "model_logs/model-checkpoint", global_step=global_step, write_meta_graph=True)
                print("Optimization Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training. "
             "Mutually exclusive with the --tensorboard_debug_address flag.")

    parser.add_argument(
        "--tensorboard_debug_address",
        type=str,
        default=None,
        help="Connect to the TensorBoard Debugger Plugin backend specified by "
             "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
             "--debug flag.")

    parser.add_argument(
        "--ui_type",
        type=str,
        default="curses",
        help="Command-line user interface type (curses | readline)")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
