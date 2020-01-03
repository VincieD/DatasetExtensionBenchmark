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
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
#from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow import keras

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.

# from efficientnet import EfficientNetB0 as Net
# from efficientnet import center_crop_and_resize, preprocess_input

# models can be build with Keras or Tensorflow frameworks
# use keras and tfkeras modules respectively
# efficientnet.keras / efficientnet.tfkeras
import efficientnet.tfkeras as efn
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras import backend




import argparse

parser = argparse.ArgumentParser(description='My efficient net for fine tuning and transfer learning')
parser.add_argument('--input_dir', default=os.path.join(os.getcwd(),'data'), help='Input directory')
parser.add_argument('--output_dir', default='', help='Output directory')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')
parser.add_argument('--num_train', default=1000, type=int, help='train size')
parser.add_argument('--num_test', default=100, type=int, help='test size')
parser.add_argument('--num_val', default=100, type=int, help='val size')
parser.add_argument('--epochs', default=400, type=int, help='number of epochs')
parser.add_argument('--dropout', default=0.2, type=float, help='Dropout rate')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-5, type=float, help='Batch size')

args = parser.parse_args()

def preprocess(x):
    return x/255

def load_data(args):
    NUM_TRAIN = args.num_train
    NUM_TEST = args.num_test

    # The path to the directory where the original
    # dataset was uncompressed
    original_dataset_dir = args.input_dir #os.path.join(os.getcwd(), 'data', 'INRIA_Person')

    train_path = os.path.join(original_dataset_dir,'INRIA_Person_Dataset_Train_256')
    test_path = os.path.join(original_dataset_dir,'INRIA_Person_Dataset_Test_256')

    pos_train_images = glob.glob(os.path.join(train_path, "pos", '*.png'))
    neg_train_images = glob.glob(os.path.join(train_path, "neg", '*.png'))

    pos_val_test_images = glob.glob(os.path.join(test_path, "pos", '*.png'))
    neg_val_test_images = glob.glob(os.path.join(test_path, "neg", '*.png'))

    random.shuffle(pos_train_images)
    random.shuffle(neg_train_images)
    random.shuffle(pos_val_test_images)
    random.shuffle(neg_val_test_images)

    print("total pos images: {}\n\rtotal neg images: {}".format(len(pos_train_images)+len(pos_val_test_images), len(neg_train_images)+len(neg_val_test_images)))

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

    tensorboard = TensorBoard(log_dir=filedir, histogram_freq=50, batch_size=32, write_graph=True,
                                               write_grads=False, write_images=True, embeddings_freq=0,
                                               embeddings_layer_names=None, embeddings_metadata=None,
                                               embeddings_data=None, update_freq='epoch')

    # loading pretrained conv base model
    conv_base = efn.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    # conv_base = Net(weights="imagenet", include_top=False, input_shape=input_shape)

    # Build model
    # model = models.Sequential()
    # model.add(conv_base)
    # model.add(layers.GlobalMaxPooling2D(name="gmp"))
    # # model.add(layers.Flatten(name="flatten"))
    # if dropout_rate > 0:
    #     model.add(layers.Dropout(dropout_rate, name="dropout_out"))
    # # model.add(layers.Dense(256, activation='relu', name="fc1"))
    # model.add(layers.Dense(2, activation="softmax", name="fc_out"))

    x = conv_base.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="dropout_out")(x)
    # and a logistic layer -- let's say we have 200 classes
    x = layers.Dense(2, activation='linear', name="fc_out")(x)
    predictions = layers.Activation('softmax',name='final_act')(x)

    # this is the model we will train
    model = models.Model(inputs=conv_base.input, outputs=predictions)
    model.summary()

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest')


    # Note that the validation data should not be augmented!
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess)
    # Note that the test data should not be augmented!
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess)

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

    #model.summary()

    print('total training images:', str(train_generator.samples))
    print('total validation images:', str(validation_generator.samples))
    print('total test images:', str(test_generator.samples))

    print('This is the number of trainable layers '
          'before freezing the conv base:', len(model.trainable_weights))

    conv_base.trainable = True

    layer_name = 'block7a_se_excite'

    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == layer_name:
            print('Layers are trainable from     ------> '+layer_name+'     onwards')
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    print('This is the number of trainable layers '
          'after freezing some layers:', len(model.trainable_weights))

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
        callbacks=[csv_logger, tensorboard],
        verbose=1,
        use_multiprocessing=False,
        workers=4)

    # score = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)
    score = model.evaluate_generator(test_generator, max_queue_size=10,
                                     workers=4, use_multiprocessing=False)
    # Save model and hyperparameter

    Savename_model = os.path.join(filedir,'Model_EfficientNetB0_Person.h5')
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
                 ('Data', 'INRIA with own data together')
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
