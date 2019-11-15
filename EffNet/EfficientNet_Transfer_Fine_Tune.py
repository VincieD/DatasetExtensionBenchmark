from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import csv
import collections
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from IPython.display import Image
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.

from efficientnet import EfficientNetB0 as Net
from efficientnet import center_crop_and_resize, preprocess_input

import argparse

parser = argparse.ArgumentParser(description='My efficient net for fine tuning and transfer learning')
parser.add_argument('--input_dir', default=os.path.join(os.getcwd(),'data','INRIA_Person'), help='Input directory')
parser.add_argument('--output_dir', default='', help='Output directory')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')
parser.add_argument('--num_train', default=1000, type=int, help='train size')
parser.add_argument('--num_test', default=100, type=int, help='test size')
parser.add_argument('--num_val', default=100, type=int, help='val size')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float, help='Batch size')

args = parser.parse_args()


def load_data(args):
    NUM_TRAIN = args.num_train
    NUM_TEST = args.num_test

    # The path to the directory where the original
    # dataset was uncompressed
    original_dataset_dir = args.input_dir #os.path.join(os.getcwd(), 'data', 'INRIA_Person')

    train_path = os.path.join(original_dataset_dir,'Train')
    test_path = os.path.join(original_dataset_dir,'Test')

    pos_train_images = glob.glob(os.path.join(train_path, "pos", '*.png'))
    neg_train_images = glob.glob(os.path.join(train_path, "neg", '*.png'))

    pos_val_test_images = glob.glob(os.path.join(test_path, "pos", '*.png'))
    neg_val_test_images = glob.glob(os.path.join(test_path, "neg", '*.png'))

    random.shuffle(pos_train_images)
    random.shuffle(neg_train_images)
    random.shuffle(pos_val_test_images)
    random.shuffle(neg_val_test_images)

    print("total pos images: {}\n\rtotal neg images: {}".format(len(pos_train_images)+len(pos_val_test_images), len(neg_train_images)+len(neg_val_test_images)))


    # pos_val_images = pos_val_test_images[:args.num_val]
    # pos_test_images = pos_val_test_images[args.num_val:args.num_test]
    # neg_val_images = neg_val_test_images[:args.num_val]
    # neg_test_images = neg_val_test_images[args.num_val:args.num_test]



    # The directory where we will
    # store our smaller dataset
    base_dir = os.path.join(os.getcwd(), 'data', 'neg_vs_pos_small')
    if os.path.isdir(base_dir):

        shutil.rmtree(base_dir)
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
    fnames = pos_train_images[:NUM_TRAIN // 2]
    for fname in fnames:
        dst = os.path.join(train_pos_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    offset = NUM_TRAIN // 2
    # Copy next NUM_TEST //2 pos images to validation_pos_dir
    fnames = pos_val_test_images[:NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(validation_pos_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)
    offset = offset + NUM_TEST // 2
    # Copy next NUM_TRAIN//2 pos images to test_pos_dir
    fnames = pos_val_test_images[NUM_TEST // 2 : NUM_TEST]
    for fname in fnames:
        dst = os.path.join(test_pos_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    # Copy first NUM_TRAIN//2 neg images to train_neg_dir
    fnames = neg_train_images[:NUM_TRAIN // 2]
    for fname in fnames:
        dst = os.path.join(train_neg_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    offset = NUM_TRAIN // 2
    # Copy next NUM_TEST // 2 neg images to validation_neg_dir
    fnames = neg_val_test_images[:NUM_TEST // 2]
    for fname in fnames:
        dst = os.path.join(validation_neg_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)
    offset = offset + NUM_TEST // 2

    # Copy next NUM_TEST // 2 neg images to test_neg_dir
    fnames = neg_val_test_images[NUM_TEST // 2 : NUM_TEST]
    for fname in fnames:
        dst = os.path.join(test_neg_dir, os.path.basename(fname))
        shutil.copyfile(fname, dst)

    print('total training pos images:', len(os.listdir(train_pos_dir)))
    print('total training neg images:', len(os.listdir(train_neg_dir)))
    print('total validation pos images:', len(os.listdir(validation_pos_dir)))
    print('total validation neg images:', len(os.listdir(validation_neg_dir)))
    print('total test pos images:', len(os.listdir(test_pos_dir)))
    print('total test neg images:', len(os.listdir(test_neg_dir)))

    return train_path, test_path, test_path

def train(args, train_dir, validation_dir, test_dir):
    batch_size = args.batch_size
    width = args.width
    height = args.height
    epochs = args.epochs
    NUM_TRAIN = args.num_train
    NUM_TEST = args.num_test
    dropout_rate = args.dropout
    lr =args.lr
    loss ='categorical_crossentropy'
    input_shape = (height, width, 3)

    name = 'Transfer_Learn_EfficientNetB0_epochs_%i_lr_%f_batch_%i_dropout_%i' % (epochs,lr,batch_size,dropout_rate)
    log_dir=os.path.join(os.getcwd(),'logs')
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    filedir = os.path.join(log_dir,name)
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    # else:
    #     shutil.rmtree(filedir)

    # loading pretrained conv base model

    conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)

    # Build model
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalMaxPooling2D(name="gmp"))
    # model.add(layers.Flatten(name="flatten"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_out"))
    # model.add(layers.Dense(256, activation='relu', name="fc1"))
    model.add(layers.Dense(2, activation="softmax", name="fc_out"))

    train_datagen = ImageDataGenerator(
        rescale=1. / 255)  # ,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest')


    # Note that the validation data should not be augmented!
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    # Note that the test data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to target height and width.
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=True,
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

    Savename_log = os.path.join(filedir,'Datalog_'+name+'.csv')
    csv_logger = CSVLogger(Savename_log, append=False, separator=',')

    model.compile(loss=loss,
                  optimizer=optimizers.RMSprop(lr=lr),
                  metrics=['acc'])
    print('train gen len '+str(len(train_generator)))
    print('test gen len '+str(len(test_generator)))
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[csv_logger],
        verbose=1,
        use_multiprocessing=False,
        workers=4)

    # score = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)
    score = model.evaluate_generator(test_generator, max_queue_size=10,
                                     workers=4, use_multiprocessing=False)
    # Save model and hyperparameter

    Savename_model = os.path.join(filedir,'Model_'+name+'.h5')
    model.save(Savename_model)

    para_list = [('Total Train Samples', str(1000)),
                 ('Epochs', epochs),
                 ('Batch Size', batch_size),
                 ('Loss Function', loss),
                 ('Dropout Rate', dropout_rate),
                 ('Optimizer', 'RMSProp'),
                 ('Metrics', 'acc'),
                 ('Test Score/Loss', score[0]),
                 ('Test Accuracy', score[1]),
                 ('Learning Rate', lr),
                 ]
    Parameterlist = collections.OrderedDict(para_list)
    Savename_paralist = os.path.join(filedir,'Parameter_'+name+'.csv')
    with open(Savename_paralist, 'w') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, dialect='excel', delimiter=',')
        for key, value in Parameterlist.items():
            writer.writerow([key, value])
    # Plots
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_x = range(len(acc))

    Plotname_acc = os.path.join(filedir,'Accuracy_'+name+'.png')
    plt.plot(epochs_x, acc, label='Training acc')
    plt.plot(epochs_x, val_acc, label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('Accuracy', fontsize=9)
    plt.xlabel('Epoch', fontsize=9)
    plt.legend()
    plt.savefig(Plotname_acc + '.png', format='png')

    plt.figure()
    Plotname_loss = os.path.join(filedir,'Loss_'+name+'.png')
    plt.plot(epochs_x, loss, label='Training loss')
    plt.plot(epochs_x, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('Loss', fontsize=9)
    plt.xlabel('Epoch', fontsize=9)
    plt.legend()
    plt.savefig(Plotname_loss + '.png', format='png')



def main(args):

    train_dir, validation_dir, test_dir = load_data(args)
    train(args, train_dir, validation_dir, test_dir)


if __name__ == '__main__':
    main(args)
