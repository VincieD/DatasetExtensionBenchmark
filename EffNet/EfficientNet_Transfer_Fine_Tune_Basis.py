
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil

import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
import random

batch_size = 48

width = 150
height = 150
epochs = 5
NUM_TRAIN = 500
NUM_TEST = 50
dropout_rate = 0.2
input_shape = (height, width, 3)

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.
from efficientnet import EfficientNetB0 as Net
from efficientnet import center_crop_and_resize, preprocess_input

# loading pretrained conv base model

conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)



model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gmp"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation="softmax", name="fc_out"))

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = os.path.join(os.getcwd(),'data','INRIA_Person','Train')

pos_images = glob.glob(os.path.join(original_dataset_dir, "pos", '*.png'))
neg_images = glob.glob(os.path.join(original_dataset_dir, "neg", '*.png'))

random.shuffle(pos_images)
random.shuffle(neg_images)

print("total pos images: {}\n\rtotal neg images: {}".format(len(pos_images), len(neg_images)))

# The directory where we will
# store our smaller dataset
base_dir = './data/neg_vs_pos_small'
os.makedirs(base_dir, exist_ok=True)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(base_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Directory with our training pos pictures
train_pos_dir = os.path.join(train_dir, 'pos')
os.makedirs(train_pos_dir, exist_ok=True)

# Directory with our training neg pictures
train_neg_dir = os.path.join(train_dir, 'neg')
os.makedirs(train_neg_dir, exist_ok=True)

# Directory with our validation pos pictures
validation_pos_dir = os.path.join(validation_dir, 'pos')
os.makedirs(validation_pos_dir, exist_ok=True)

# Directory with our validation neg pictures
validation_neg_dir = os.path.join(validation_dir, 'neg')
os.makedirs(validation_neg_dir, exist_ok=True)

# Directory with our validation pos pictures
test_pos_dir = os.path.join(test_dir, 'pos')
os.makedirs(test_pos_dir, exist_ok=True)

# Directory with our validation neg pictures
test_neg_dir = os.path.join(test_dir, 'neg')
os.makedirs(test_neg_dir, exist_ok=True)

# Copy first NUM_TRAIN//2 pos images to train_pos_dir
fnames = pos_images[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_pos_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST //2 pos images to validation_pos_dir
fnames = pos_images[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_pos_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2
# Copy next NUM_TRAIN//2 pos images to test_pos_dir
fnames = pos_images[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_pos_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)


# Copy first NUM_TRAIN//2 neg images to train_neg_dir
fnames = neg_images[:NUM_TRAIN//2]
for fname in fnames:
    dst = os.path.join(train_neg_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

offset = NUM_TRAIN//2
# Copy next NUM_TEST // 2 neg images to validation_neg_dir
fnames = neg_images[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(validation_neg_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)
offset = offset + NUM_TEST // 2

# Copy next NUM_TEST // 2 neg images to test_neg_dir
fnames = neg_images[offset:offset + NUM_TEST // 2]
for fname in fnames:
    dst = os.path.join(test_neg_dir, os.path.basename(fname))
    shutil.copyfile(fname, dst)

print('total training pos images:', len(os.listdir(train_pos_dir)))
print('total training neg images:', len(os.listdir(train_neg_dir)))
print('total validation pos images:', len(os.listdir(validation_pos_dir)))
print('total validation neg images:', len(os.listdir(validation_neg_dir)))
print('total test pos images:', len(os.listdir(test_pos_dir)))
print('total test neg images:', len(os.listdir(test_neg_dir)))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255)#,
      # rotation_range=40,
      # width_shift_range=0.2,
      # height_shift_range=0.2,
      # shear_range=0.2,
      # zoom_range=0.2,
      # horizontal_flip=True,
      # fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to target height and width.
        target_size=(height, width),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

# model.add(conv_base)
# model.add(layers.GlobalMaxPooling2D(name="gap"))
# # model.add(layers.Flatten(name="flatten"))
# if dropout_rate > 0:
#     model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# # model.add(layers.Dense(256, activation='relu', name="fc1"))
# model.add(layers.Dense(2, activation='softmax', name="fc_out"))

model.summary()


print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))


model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch= NUM_TRAIN //batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps= NUM_TEST //batch_size,
      verbose=1,
      use_multiprocessing=True,
      workers=4)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

plt.plot(epochs_x, acc, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_x, loss, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()





