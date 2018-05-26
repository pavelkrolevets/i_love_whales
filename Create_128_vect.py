from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pandas as pd
import os

x_train = np.load('./data/X_train.npy')
x_test = np.load('./data/X_test.npy')

checkpoint_directory = './model_logs/'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_directory)

graph=tf.Graph()
with tf.Session(graph=graph) as sess:

    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
    #keep_prob = graph.get_tensor_by_name('keep_prob:0')
    x = graph.get_tensor_by_name('placehold_x:0')

    # keep_prob = graph.get_tensor_by_name('keep_prob:0')  # dropout probability
    oper_restore = graph.get_tensor_by_name('inference:0')

    embed_train = []
    for i in range(x_train.shape[0]):
        number = x_train[i,:,:,:]
        #print(number)
        number = np.reshape(number, (1,100,100,1))
        prediction = sess.run(oper_restore, feed_dict={x: number})
        prediction = np.reshape(prediction,(128))
        embed_train.append(prediction)
    embed_train=np.asanyarray(embed_train)

    embed_test = []
    for i in range(x_test.shape[0]):
        number = x_test[i,:,:,:]
        #print(number)
        number = np.reshape(number, (1,100,100,1))
        prediction = sess.run(oper_restore, feed_dict={x: number})
        prediction = np.reshape(prediction,(128))
        embed_test.append(prediction)
    embed_test=np.asanyarray(embed_test)

np.save('./np_embeddings/embeddings_train.npy', embed_train)
np.save('./np_embeddings/embeddings_test.npy', embed_test)

print("Saved")



