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



parser = argparse.ArgumentParser(description='My efficient net visualization')
parser.add_argument('--input_dir', default=os.path.join(os.getcwd(),'data','INRIA_Person_Dataset_Test_256','pos'), help='Input image directory')
parser.add_argument('--target_class', default=1, type=int, help='input image class')
parser.add_argument('--output_dir', default=os.path.join(os.getcwd(),'CAM_images'), help='Output directory')
parser.add_argument('--model_file', default=os.path.join(os.getcwd(),'logs','Transfer_Learn_EfficientNetB0_epochs_350_lr_0.000020_batch_32_dropout_0','Model_EfficientNetB0_Person.h5'), help='Model file path')
parser.add_argument('--width', default=224, type=int, help='input image width')
parser.add_argument('--height', default=224, type=int, help='input image height')
parser.add_argument('--cam', default=1, type=int, help='output class activation map')


args = parser.parse_args()

# def CAM(args):
#     model = load_model(args.model_file)
#     model.summary()
#
#     if not os.path.isdir(args.output_dir):
#         os.makedirs(args.output_dir)
#
#     for filename in os.listdir(args.input_dir):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#
#             # The local path to our target image
#             img_path = os.path.join(args.input_dir, filename)
#
#             # `img` is a PIL image of size 224x224
#             img = image.load_img(img_path, target_size=(args.width,args.height))
#
#             # `x` is a float32 Numpy array of shape (224, 224, 3)
#             x = image.img_to_array(img)
#
#             # We add a dimension to transform our array into a "batch"
#             # of size (1, 224, 224, 3)
#             x = np.expand_dims(x, axis=0)
#
#             # Finally we preprocess the batch
#             # (this does channel-wise color normalization)
#             x = preprocess_input(x)
#
#             preds = model.predict(x)
#             print('Probability of negative and positive class',preds[0])
#             # print('Predicted:', decode_predictions(preds, top=3)[0])
#
#             class_index = np.argmax(preds[0])
#
#             # This is the "african elephant" entry in the prediction vector
#             class_output = model.output[:, class_index]
#
#             # The is the output feature map of the `block5_conv3` layer,
#             # the last convolutional layer in VGG16
#             last_conv_layer = model.get_layer('top_conv')
#
#
#             # This is the gradient of the "african elephant" class with regard to
#             # the output feature map of `block5_conv3`
#             grads = K.gradients(class_output, last_conv_layer.output)[0]
#
#             # This is a vector of shape (512,), where each entry
#             # is the mean intensity of the gradient over a specific feature map channel
#             pooled_grads = K.mean(grads, axis=(0, 1, 2))
#
#             # This function allows us to access the values of the quantities we just defined:
#             # `pooled_grads` and the output feature map of `block5_conv3`,
#             # given a sample image
#             iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
#
#             # These are the values of these two quantities, as Numpy arrays,
#             # given our sample image of two elephants
#             pooled_grads_value, conv_layer_output_value = iterate([x])
#
#             # We multiply each channel in the feature map array
#             # by "how important this channel is" with regard to the elephant class
#             for i in range(1280):
#                 conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
#
#             # The channel-wise mean of the resulting feature map
#             # is our heatmap of class activation
#             heatmap = np.mean(conv_layer_output_value, axis=-1)
#             heatmap = np.maximum(heatmap, 0)
#             heatmap /= np.max(heatmap)
#             # plt.matshow(heatmap)
#             # plt.show()
#
#             import cv2
#
#             # We use cv2 to load the original image
#             img = cv2.imread(img_path)
#
#             # We resize the heatmap to have the same size as the original image
#             heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#
#             # We convert the heatmap to RGB
#             heatmap = np.uint8(255 * heatmap)
#
#             # We apply the heatmap to the original image
#             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#             # 0.4 here is a heatmap intensity factor
#             superimposed_img = heatmap * 0.4 + img
#
#             # Save the image to disk
#             cv2.imwrite(os.path.join(args.output_dir,'grad_CAM'+filename), superimposed_img)
#         else:
#             continue

def visualize_class_activation_map(args):

    model = load_model(args.model_file)
    model.summary()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            output_path=os.path.join(args.output_dir,(filename.split('.')[0]+'_CAM'+'.png'))
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

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def main(args):
    # CAM(args)
    if args.cam==True:
        visualize_class_activation_map(args)

if __name__=='__main__':
    main(args)






















