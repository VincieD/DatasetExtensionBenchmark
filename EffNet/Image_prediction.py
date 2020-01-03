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
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='My efficient net visualization')
parser.add_argument('--input_dir', default='C:/Users/qxy9300/Desktop/test', help='Input image directory class folders')
parser.add_argument('--output_dir', default=os.path.join(os.getcwd(),'wrong_predictions_all_data'), help='Output directory')
parser.add_argument('--model_file', default=os.path.join(os.getcwd(),'logs','Transfer_Learn_EfficientNetB0_epochs_300_lr_0.000020_batch_32_dropout_0_tf_preprocess','Model_EfficientNetB0_Person.h5'), help='Model file path')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')

args = parser.parse_args()


def prediction(args):
    model = load_model(args.model_file)
    #model.summary()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    height = args.height
    width = args.width
    batch_size = 32

    # Note that the test data should not be augmented!
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
    print(prediction)
    print(score)


    # print(score)
    image_list=[]
    prob_list=[]
    for i in range(len(test_generator.labels)):
        if test_generator.labels[i]!=pred[i]:
            #print('pred[i]',pred[i])
            image_list.append(test_generator.filepaths[i])
            prob_list.append(prediction[i,test_generator.labels[i]])

    print(image_list)

    for i, file in enumerate(image_list):
        prob = prob_list[i]
        dst =os.path.join(args.output_dir,str(prob)+'.png')
        shutil.copy(file, dst)


    # Define a dictionary containing data
    data = {'Name': test_generator.filenames,
            'class': test_generator.labels,
            'prediction': pred}

    # Convert the dictionary into DataFrame
    df = pd.DataFrame(data)

    export_csv = df.to_csv(r'D:\EffNet\pred_results.csv', index=None, header=True)

    print('hello')

def prediction2(args):
    model = load_model(args.model_file)
    # model.summary()
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            print(filename)
            impath = os.path.join(args.input_dir, filename)

            img = image.load_img(impath, target_size=(224, 224))
            y = image.img_to_array(img)
            y = np.expand_dims(y, axis=0)
            im =preprocess_input(y)
            prediction = model.predict(im)
            print(prediction)

def main():
    # prediction(args)
    prediction2(args)

if __name__ == '__main__':
    main()
















