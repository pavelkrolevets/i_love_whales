from __future__ import print_function
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import pandas as pd

x_train = np.load('./data/X_train.npy')
# -> TODO: crate test dataset and load it as numpy

"""creating embeddings for labels"""
data = pd.read_csv('./data/train.csv')
# dropping "new_whales"
data = data[data.Id != "new_whale"]
labels = data['Id']
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(labels)

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

    embed = []
    # embed_fire = []
    # -> TODO feed TEST dataset to model and get 128 vectors, change the size accordinly
    # for i in range(x_test.shape[0]):
    #     number = x_test[i,:]
    #     #print(number)
    #     number = np.reshape(number, (1,300,300,1))
    #     prediction = sess.run(oper_restore, feed_dict={x: number})
    #     prediction = np.reshape(prediction,(128))
    #     embed_fire.append(prediction)

    # embed_fire=np.asanyarray(embed_fire)

    for i in range(x_train.shape[0]):
        number = x_train[i,:,:,:]
        #print(number)
        number = np.reshape(number, (1,100,100,1))
        prediction = sess.run(oper_restore, feed_dict={x: number})
        prediction = np.reshape(prediction,(128))
        embed.append(prediction)

    embed=np.asanyarray(embed)



np.save('./np_embeddings/embeddings.npy', embed)
#np.save('./np_embeddings/labels.npy', y_train)
# np.save('./np_embeddings/tets_embed.npy', embed_fire)

print("Saved")



