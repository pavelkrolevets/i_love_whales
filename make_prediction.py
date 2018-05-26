import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
import time

"""Reading train data"""
data_train = pd.read_csv('./data/train.csv')
# dropping "new_whales"
data_train = data_train[data_train.Id != "new_whale"]
data_train = data_train.reset_index(drop=True)
labels = data_train['Id']
file_names = data_train['Image']
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(labels)
#load embeddings
embed_train = np.load('./np_embeddings/embeddings_train.npy')
assert embed_train.shape[0] == len(y_train)

"""Reading test data"""
embed_test = np.load('./np_embeddings/embeddings_test.npy')
data_test = pd.DataFrame()
y_test = []
for path, subdirs, files in os.walk('./data/test'):
    for name in files:
        y_test.append(name)
#load test emeddings
embed_test = np.load('./np_embeddings/embeddings_test.npy')
assert embed_test.shape[0] == len(y_test)

"""Compute distances"""
prediction = []
start_time = time.time()
for i in range(embed_test.shape[0]):
    distances = []
    print('Complited: ', i/embed_test.shape[0])
    for j in range(embed_train.shape[0]):
        dist = np.sqrt(np.sum(np.square(np.subtract(embed_test[i, :], embed_train[j, :]))))
        distances.append(dist)
    val, idx = min((val, idx) for (idx, val) in enumerate(distances))
    prediction.append(y_train[idx])
run_time = time.time() - start_time
print('Run time: ', run_time)

predicted_data = pd.DataFrame()
predicted_data['Image'] = y_test
predicted_data['Id'] = le.inverse_transform(prediction)
predicted_data.to_csv('submission.csv')

print()
