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
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.datasets import cifar10
import tensorflow.keras
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='My efficient net visualization')
parser.add_argument('--input_dir_pos', default=os.path.join(os.getcwd(),'data','INRIA_Person','Test','pos'), help='Input image directory pos')
parser.add_argument('--input_dir_neg', default=os.path.join(os.getcwd(),'data','INRIA_Person','Test','neg'), help='Input image directory neg')
parser.add_argument('--output_dir', default=os.path.join(os.getcwd(),'DeepExplain'), help='Output directory')
parser.add_argument('--model_file', default=os.path.join(os.getcwd(),'logs','Transfer_Learn_EfficientNetB0_epochs_13_lr_0.000020_batch_32_dropout_0','Model_EfficientNetB0_Person.h5'), help='Model file path')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')


args = parser.parse_args()


batch_size = 128
num_classes = 10
epochs = 3

# input image dimensions
img_rows, img_cols = 28,28

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = (x_train - 0.5) * 2
x_test = (x_test - 0.5) * 2
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

model = load_model(args.model_file)
model.summary()

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

    xs = x_test[0:10]
    ys = y_test[0:10]

    attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
    # attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
    # attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
    # attributions_dl    = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
    # attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
    # attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

    # Compare Gradient * Input with approximate Shapley Values
    # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
    # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
    attributions_sv = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)

from utils import plot, plt

n_cols = 6
n_rows = int(len(attributions_gradin) / 2)
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3*n_cols, 3*n_rows))

for i, (a1, a2) in enumerate(zip(attributions_gradin, attributions_sv)):
    row, col = divmod(i, 2)
    plot(xs[i].reshape(28, 28), cmap='Greys', axis=axes[row, col*3]).set_title('Original')
    plot(a1.reshape(28,28), xi = xs[i], axis=axes[row,col*3+1]).set_title('Grad*Input')
    plot(a2.reshape(28,28), xi = xs[i], axis=axes[row,col*3+2]).set_title('Shapley Values')