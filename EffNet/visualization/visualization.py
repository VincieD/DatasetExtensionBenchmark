import os

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
import numpy as np
import cv2
import argparse
from viz_occlusion import *
from model import *
from utils import *



parser = argparse.ArgumentParser(description='My efficient net visualization')
parser.add_argument('--input_dir', default='C:/Users/qxy9300/Desktop/test/pos', help='Input image directory')
parser.add_argument('--target_class', default=1, type=int, help='input image class')
parser.add_argument('--output_dir', default=os.path.join(os.getcwd(),'CAM_images'), help='Output directory')
parser.add_argument('--model_file', default=os.path.join(os.getcwd(),'logs','Transfer_Learn_EfficientNetB0_epochs_300_lr_0.000020_batch_32_dropout_0_tf_preprocess','Model_EfficientNetB0_Person.h5'), help='Model file path')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')
parser.add_argument('--cam', default=1, type=int, help='output class activation map')

parser.add_argument('--occ', default=1, type=int, help='if output occluder')
parser.add_argument('--size', type=int, default=224)  # Layer name
parser.add_argument('--occ_size', type=int, default=40)  # Size of occluding window
parser.add_argument('--pixel', type=int, default=0)  # Occluding window - pixel values
parser.add_argument('--stride', type=int, default=5)  # Occlusion Stride
parser.add_argument('--norm', type=int, default=1)  # Normalize probabilities first
parser.add_argument('--percentile', type=int, default=25)  # Regularization percentile for heatmap


args = parser.parse_args()

def visualize_class_activation_map(args,id,filename,model):

    output_path=os.path.join(args.output_dir,(filename.split('.')[0]+'_ID_'+str(id)+'_CAM'+'.png'))
    # The local path to our target image
    img_path = os.path.join(args.input_dir, filename)
    original_img = cv2.imread(img_path, 1)
    width, height, _ = original_img.shape
    #print(width, height)

    resized = cv2.resize(original_img, (args.width, args.height), interpolation=cv2.INTER_AREA)

    # Reshape to the network input shape (3, w, h).
    # img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
    img = np.array([np.float32(resized)])
    img=img/255

    # Get the input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "top_conv")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    target_class = args.target_class
    for i, w in enumerate(class_weights[:, target_class]):
        cam += w * conv_outputs[:, :, i]
    print("predictions  ", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + original_img
    cv2.imwrite(output_path, img)
    print('Saved ',output_path)
    id+=1

def occluder(args,id,filename,model):
    print('/n', args)
    img_path = os.path.join(args.input_dir, filename)
    img_path, img_size = img_path, args.size
    img_name = filename.split('.')[0]#+'_ID_'+str(id)
    occ_size, occ_pixel, occ_stride = args.occ_size, args.pixel, args.stride

    print(filename, img_size)
    # Get original image
    # input_image = cv2.imread(img_path)
    # input_image = cv2.resize(input_image, (img_size, img_size)).astype(np.float32)

    input_image = image.load_img(img_path, target_size=(224, 224))
    input_image = image.img_to_array(input_image)

    # Get probability list and print
    result = pred_prob_list(model, input_image)
    print(result)
    # de_result = decode_predictions(result)[0]
    # print('\nPredicted: ', de_result)
    #
    # # Start occlusion experiment and store predicted probabilities in a file
    # print('Running occlusion iterations (Class:', de_result[0][1], ') ...\n')
    probs = get_occ_imgs(img_path, img_name,model,img_size, occ_size, occ_pixel, occ_stride, result)

    # Get probabilities and apply regularization
    print('\nGetting probability heat-map and regularizing...')
    probs = np.load('occ_exp/probs_' + img_name + '.npy')
    heat = regularize(probs, args.norm, args.percentile,img_path, filename)

    # Project heatmap on original image
    print('\nProject the heat-map to original image...')
    aug = join(heat, img_path, img_size, occ_size, filename)

    print('\n Occlusion Done')

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def main(args):
    # Input pre-trained model
    model = load_trained_model(args.model_file)
    id=0
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            if args.cam==True:
                if not os.path.isdir(args.output_dir):
                    os.makedirs(args.output_dir)
                visualize_class_activation_map(args,id,filename,model)
            if args.occ==True:
                occluder(args,id,filename,model)
            id+=1

if __name__=='__main__':
    main(args)






















