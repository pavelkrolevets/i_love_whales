import pandas as pd
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import h5py
import os

data = pd.read_csv('./data/train.csv')
# dropping "new_whales"
data = data[data.Id != "new_whale"]

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


X_train = []
for i in data['Image'].values:
    addres = './data/train/'+i
    img = cv.imread(addres, 1)
    #ret, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    img = cv.resize(img, (160,160))
    prewhitened = prewhiten(img)
    # plt.imshow(prewhitened)
    # plt.show()
    np_image_data = prewhitened.reshape((160,160,3))
    X_train.append(np_image_data)

X_train = np.asarray(X_train)
np.save('./data/X_train.npy', X_train)

paths_test = []
for path, subdirs, files in os.walk('./data/test'):
    for name in files:
        paths_test.append(os.path.join(path, name))

X_test = []

for path in paths_test:
    img = cv.imread(path, 1)
    # ret, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    img = cv.resize(img, (160, 160))
    prewhitened = prewhiten(img)
    # plt.imshow(prewhitened)
    # plt.show()
    np_image_data = prewhitened.reshape((160, 160, 3))
    X_test.append(np_image_data)

X_test = np.asarray(X_test)
np.save('./data/X_test.npy', X_test)