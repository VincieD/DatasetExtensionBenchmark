import os
import matplotlib.pyplot as plt
# Import pandas package
import pandas as pd
import shutil

# model use some custom objects, so before loading saved model
# import module your network was build with
# e.g. import efficientnet.keras / import efficientnet.tfkeras
import efficientnet.tfkeras
from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


import argparse

parser = argparse.ArgumentParser(description='My efficient net visualization')
parser.add_argument('--input_dir', default=os.path.join(os.getcwd(),'data','dataset_tobi_vaclav_campus_256'), help='Input image directory class folders in it')
parser.add_argument('--output_dir', default=os.path.join(os.getcwd(),'logs','wrong_predictions'), help='Output directory')
parser.add_argument('--model_file', default=os.path.join(os.getcwd(),'logs','Transfer_Learn_EfficientNetB0_epochs_350_lr_0.000020_batch_32_dropout_0','Model_EfficientNetB0_Person.h5'), help='Model file path')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')

args = parser.parse_args()


def prediction(args):
    model = load_model(args.model_file)
    model.summary()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    height = args.height
    width = args.width
    batch_size = 32

    # Note that the test data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        args.input_dir,
        target_size=(height, width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

    prediction = model.predict_generator(test_generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False,
                      verbose=1)
    score = model.evaluate_generator(test_generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
    pred = np.argmax(prediction,axis=1)

    pred_class = np.amax(prediction,axis=1)
    print(score)
    image_list=[]
    prob_list=[]
    for i in range(len(test_generator.labels)):
        if test_generator.labels[i]!=pred[i]:
            image_list.append(test_generator.filepaths[i])
            prob_list.append(pred_class[i])

    #print(image_list)

    for i, file in enumerate(image_list):
        prob = 1 - prob_list[i]
        dst =os.path.join(args.output_dir,str(prob)+'.png')
        shutil.copy(file, dst)


    # Define a dictionary containing data
    data = {'Name': test_generator.filenames,
            'class': test_generator.labels,
            'prediction': pred,
            'probability': pred_class}

    # Convert the dictionary into DataFrame
    df = pd.DataFrame(data)

    export_csv = df.to_csv(r'D:\EffNet\pred_results_tobi_vaclav.csv', index=None, header=True)

    print('hello')

# def predict_singe_image(args):
#     model = load_model(args.model_file)
#     model.summary()
#
#
#     '.\data\dataset_tobi_vaclav_campus_256\pos\IMG_20191126_125746.jpg'
#     prediction = model.predict('')

def main():
    prediction(args)

if __name__ == '__main__':
    main()
