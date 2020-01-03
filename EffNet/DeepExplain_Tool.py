from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile, sys, os
# Import DeepExplain
from deepexplain.tensorflow import DeepExplain
import matplotlib.pyplot as plt

# model use some custom objects, so before loading saved model
# import module your network was build with
# e.g. import efficientnet.keras / import efficientnet.tfkeras
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
import tensorflow.keras
import numpy as np
import argparse

import matplotlib
from matplotlib import cm

import sys
import os

from scipy import misc
import shutil


parser = argparse.ArgumentParser(description='My efficient net visualization')
parser.add_argument('--input_dir', default=os.path.join(os.getcwd(),'data','INRIA_Person_Dataset_Test_256'), help='Input image directory pos')
parser.add_argument('--output_dir', default=os.path.join(os.getcwd(),'DeepExplain'), help='Output directory')
parser.add_argument('--model_file', default=os.path.join(os.getcwd(),'logs','Transfer_Learn_EfficientNetB0_epochs_400_lr_0.000020_batch_32_dropout_0','Model_EfficientNetB0_Person.h5'), help='Model file path')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')


args = parser.parse_args()

def deep_explain():
    num_classes = 2

    x_test, y_test = load_data()

    x_test = x_test.astype('float32')

    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    model = load_model(args.model_file)
    #model.summary()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    with DeepExplain(session=K.get_session()) as de:  # <-- init DeepExplain context
        # Need to reconstruct the graph in DeepExplain context, using the same weights.
        # With Keras this is very easy:
        # 1. Get the input tensor to the original model
        input_tensor = model.layers[0].input

        # 2. We now target the output of the last dense layer (pre-softmax)
        # To do so, create a new model sharing the same layers until the last dense (index -2)
        fModel = models.Model(inputs=input_tensor, outputs=model.layers[-2].output)
        target_tensor = fModel(input_tensor)

        xs = x_test[50:60]
        ys = y_test[50:60]

        attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
        # attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
        # attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
        attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
        # attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
        # attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

        # Compare Gradient * Input with approximate Shapley Values
        # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
        # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
        #attributions_sv = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)

    from utils import plot, plt

    n_cols = 6
    n_rows = int(len(attributions_gradin) / 2)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))

    for i, (a1, a2) in enumerate(zip(attributions_gradin, attributions_dl)):
        row, col = divmod(i, 2)
        plot(xs[i], cmap='Greys_r', axis=axes[row, col*3]).set_title('Original')
        plot(a1, xi = xs[i], axis=axes[row,col*3+1]).set_title('Grad*Input')
        plot(a2, xi = xs[i], axis=axes[row,col*3+2]).set_title('Deep Lift')
    plt.show()
    print('Hello')

def load_data():

    dataset = DATA_SET(args.input_dir, grayScale=False)

    X, y, num_samples, num_cl = dataset.loadImages()

    print('Number of samples ',num_samples)

    return X, y

def setKey(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

class DATA_SET():
    def __init__(self, directory, grayScale=True, labels=True, img_rows=224, img_cols=224):
        self.img_rows = img_rows
        self.img_cols = img_cols

        self.labels = labels
        if grayScale == True:
            self.channels = 1  # 3 channels RGB
        else:
            self.channels = 3  # 3 channels RGB

        self.num_classes = 0

        self.pathes = {}
        self.klasses = []
        self.dataSetLen = 0
        self.minNumberOfImages = 1

        if (self.labels == True):
            self.Y_train_list = []

        for root, folders, files in os.walk(directory):
            # print (len(files))
            if len(files) > self.minNumberOfImages and root != '.':
                klasse = os.path.split(root)[-1]
                # creating list of klasses names
                # if klasse != "dataSet_mini" or len(klasse) < 1:
                # print (klasse)
                # print (len(files))
                self.klasses.append(klasse)
                # impath = os.path.join("dataSet_mini", klasse)
                impath = root
                self.num_classes += 1
                # print (impath)
                for file in os.listdir(impath):
                    if file.endswith(".jpg") or file.endswith(".png"):
                        # adding into dictionary image pathes
                        setKey(self.pathes, klasse, str(os.path.join(impath, file)))
                        self.dataSetLen += 1
                        if (self.labels == True):
                            self.Y_train_list.append(self.num_classes-1)

        print("Length of dataset is: {}".format(self.dataSetLen))

        print("Number of classes: {}".format(self.num_classes))

        self.training_shape = (self.dataSetLen, self.img_rows, self.img_cols, self.channels)

        print("Training shape: {}".format(self.training_shape))

        self.X_train_pos = np.zeros(shape=self.training_shape, dtype=np.float32)

        self.Y_train = np.zeros(shape=self.dataSetLen, dtype=np.uint8)

    # np.zeros(shape=(numberOfSatelites, 3), dtype=float)

    def loadImages(self):
        allIndexPos = 0
        allIndexNeg = 0
        # defining ration between positive and negative samples
        for klasse, empty in self.pathes.items():
            for imgPath in self.pathes[klasse]:
                # print (imgPath)
                if self.labels == True:
                    # print (self.Y_train_list[allIndexPos])
                    self.Y_train[allIndexPos] = self.Y_train_list[allIndexPos]
                if self.channels == 1:
                    # img_rgb = misc.imread(imgPath)
                    img_grey = misc.imread(imgPath, mode='L')
                    img_grey = misc.imresize(img_grey, size=(self.img_rows, self.img_cols, self.channels),
                                             interp='bilinear', mode=None)
                    # print(img_grey.dtype)
                    # print(img_grey.shape)
                else:
                    img_rgb = misc.imread(imgPath, mode='RGB')
                    img_rgb = misc.imresize(img_rgb, size=(self.img_rows, self.img_cols, self.channels),
                                            interp='bilinear', mode=None)

                # plt.imshow(img_grey, cmap=plt.cm.gray)
                # plt.show()
                if self.channels == 1:
                    self.X_train_pos[allIndexPos, :, :, 0] = ((img_grey.astype(np.float32) / 255) - 0.5) * 2

                else:
                    # self.X_train_pos[allIndexPos,:,:,:] = ((img_rgb.astype(np.float32)/ 255) - 0.5) * 2
                    # plt.imshow(img_rgb)
                    # plt.show()
                    img_rgb=self.preprocess(img_rgb)
                    self.X_train_pos[allIndexPos, :, :, :] = img_rgb.astype(np.float32)

                # print(img.shape)
                allIndexPos += 1

        return self.X_train_pos, self.Y_train ,allIndexPos, self.num_classes
    def preprocess(self,x):
        x=x/255
        return x

deep_explain()

