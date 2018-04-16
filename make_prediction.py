import numpy as np
import pandas as pd
import cv2 as cv

embed = np.load('./np_embeddings/embeddings.npy')
labels = np.load('./np_embeddings/labels.npy')
embed_fire = np.load('./np_embeddings/test_embed.npy')
matrix  = np.zeros((embed.shape[0]))

number = embed[1,:]
number_fire = embed_fire[0,:]

"""Compute distances"""
for i in range(embed.shape[0]):

    dist = np.sqrt(np.sum(np.square(np.subtract(embed[i, :], number_fire))))
    matrix[i] = dist

matrix_pd = pd.DataFrame(matrix, index=labels).sort_values(by=[0])
print(matrix_pd.iloc[0:10])

print()
