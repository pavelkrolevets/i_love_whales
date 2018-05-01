import numpy as np
import pandas as pd
import cv2 as cv
from sklearn import preprocessing

data = pd.read_csv('./data/train.csv')
# dropping "new_whales"
data = data[data.Id != "new_whale"]
data = data.reset_index(drop=True)
labels = data['Id']
file_names = data['Image']
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(labels)
data['class'] = y_train

embed = np.load('./np_embeddings/embeddings.npy')

# embed_fire = np.load('./np_embeddings/test_embed.npy')
matrix  = np.zeros((embed.shape[0]))
find_whale = data[data.Id == 'w_0a97a25']

whale = embed[find_whale.index[0],:]


"""Compute distances"""
for i in range(embed.shape[0]):

    dist = np.sqrt(np.sum(np.square(np.subtract(embed[i, :], whale))))
    matrix[i] = dist

matrix_pd = pd.DataFrame(matrix, index=labels).sort_values(by=[0])
print(matrix_pd.iloc[0:10])

print()
