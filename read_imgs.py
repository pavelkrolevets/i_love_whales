import pandas as pd
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import h5py

data = pd.read_csv('./data/train.csv')

X_train = []

for i in data['Image'].values:
    addres = './data/train/'+i
    img = cv.imread(addres, 0)
    img = cv.resize(img, (100, 100))

    # plt.imshow(img, cmap='gray')
    # plt.show()

    np_image_data = np.asarray(img)
    np_image_data = np_image_data.astype('float32') / 255
    np_image_data = np_image_data.reshape((100,100, 1))
    X_train.append(np_image_data)

X_train = np.asarray(X_train)

np.save('./data/X_train.npy', X_train)
