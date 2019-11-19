from __future__ import print_function, division

from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os

import numpy as np
from scipy import misc
import shutil


def setKey(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]


class DATA_LOADER():
    def __init__(self, directory, grayScale=False, labels=False, img_rows=128, img_cols=128):
        self.img_rows = img_rows
        self.img_cols = img_cols

        self.labels = labels
        if grayScale == True:
            self.channels = 1  # 3 channels RGB
        else:
            self.channels = 3  # 3 channels RGB

        self.num_classes = 0

        self.pathes = {}
        self.classes = []
        self.dataSetLen = 0
        self.minNumberOfImages = 100
        self.sample_list = []

        if (self.labels == True):
            self.Y_train_list = []

        for root, folders, files in os.walk(directory):
            # print ('1 ',len(files))

            if len(files) > self.minNumberOfImages and root != '.':
                # print(root)
                ped = os.path.split(root)[-1]
                dataset_dir = os.path.split(root)[0]

                if (ped != "datasets") or (len(ped) < 1):
                    # print ('2 ',ped)
                    # print ('3 ',len(files))
                    self.classes.append(ped)
                    impath = os.path.join(dataset_dir, ped)
                    self.num_classes += 1
                    # print ('4 ',impath)
                    for file in os.listdir(impath):
                        if file.endswith(".jpg") or file.endswith(".png"):
                            # adding into dictionary image pathes
                            img_rgb = misc.imread(os.path.join(impath, file), mode='RGB')

                            setKey(self.pathes, ped, str(os.path.join(impath, file)))
                            self.sample_list.append(str(os.path.join(impath, file)))
                            self.dataSetLen += 1
                            if (self.labels == True):
                                self.Y_train_list.append(self.num_classes)

        print("Length of dataset is: {}".format(self.dataSetLen))

        print("Number of classes: {}".format(self.num_classes))

        self.training_shape = (self.dataSetLen, self.img_rows, self.img_cols, self.channels)

        print("Dataset shape: {}".format(self.training_shape))

        self.X_samples = np.zeros(shape=self.training_shape, dtype=np.uint8)

    def loadImages(self, train_val_test_split):
        allIndex = 0
        train_part = train_val_test_split[0]
        val_part = train_val_test_split[1]
        test_part = train_val_test_split[2]

        if train_part + val_part + test_part != 1:
            return print('WRONG SPLIT FOR TRAIN, TEST AND VAL')

        # defining ration between positive and negative samples
        # for ped, empty in self.pathes.items():
        for imgPath in self.sample_list:  # self.pathes[ped]:
            # print (imgPath)
            if self.labels == True:
                # print (self.Y_train_list[allIndex])
                self.Y_train[allIndex] = self.Y_train_list[allIndex]
            if self.channels == 1:
                # img_rgb = misc.imread(imgPath)
                img_grey = misc.imread(imgPath, mode='L')
                img_grey = misc.imresize(img_grey, size=(self.img_rows, self.img_cols, self.channels),
                                         interp='bilinear', mode=None)
                # print(img_grey.dtype)
                # print(img_grey.shape)
            else:
                img_rgb = misc.imread(imgPath, mode='RGB')
                img_rgb = misc.imresize(img_rgb, size=(self.img_rows, self.img_cols, self.channels), interp='bilinear',
                                        mode=None)

            # plt.imshow(img_grey, cmap=plt.cm.gray)
            # plt.show()
            if self.channels == 1:
                self.X_samples[allIndex, :, :, 0] = ((img_grey.astype(np.float32) / 255) - 0.5) * 2

            else:
                # self.X_samples[allIndex,:,:,:] = ((img_rgb.astype(np.float32)/ 255) - 0.5) * 2
                # plt.imshow(img_rgb)
                # plt.show()
                self.X_samples[allIndex, :, :, :] = img_rgb.astype(np.float32)

            # print(img.shape)
            allIndex += 1

        X_train, temp = train_test_split(self.X_samples, test_size=test_part + val_part, shuffle=True, random_state=42)
        X_val, X_test = train_test_split(temp, test_size=test_part / (test_part + val_part), shuffle=True,
                                         random_state=42)

        # print("Length of train set is: {}".format(X_train))
        # print("Length of val set is: {}".format(X_val))
        # print("Length of test set is: {}".format(X_test))

        return allIndex, X_train, X_val, X_test

# dataloader = DATA_LOADER(os.path.join(os.getcwd(),'datasets','INRIA_Person_Dataset_Train_128'), grayScale=False)
# num_samples, X_train, X_val, X_test = dataset.loadImages(train_val_test_split=[0.6, 0.2, 0.2])
#
# print('hello')
