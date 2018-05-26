import pandas as pd
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import h5py
import os

data = pd.read_csv('./data/train.csv')
# dropping "new_whales"
data = data[data.Id != "new_whale"]

X_train = []
for i in data['Image'].values:
    addres = './data/train/'+i
    img = cv.imread(addres, 0)
    #ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    img = cv.resize(img, (100,100))

    # plt.imshow(img, cmap='gray')
    # plt.show()

    np_image_data = np.asarray(img)
    np_image_data = np_image_data.astype('float32') / 255
    np_image_data = np_image_data.reshape((100,100, 1))
    X_train.append(np_image_data)

X_train = np.asarray(X_train)

np.save('./data/X_train.npy', X_train)

paths_test = []
for path, subdirs, files in os.walk('./data/test'):
    for name in files:
        paths_test.append(os.path.join(path, name))

X_test = []

for path in paths_test:
    img = cv.imread(path, 0)
    # ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    img = cv.resize(img, (100, 100))

    # plt.imshow(img, cmap='gray')
    # plt.show()

    np_image_data = np.asarray(img)
    np_image_data = np_image_data.astype('float32') / 255
    np_image_data = np_image_data.reshape((100, 100, 1))
    X_test.append(np_image_data)

X_test = np.asarray(X_test)
np.save('./data/X_test.npy', X_test)