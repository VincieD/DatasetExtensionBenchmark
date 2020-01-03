# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import os
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--input", default='C:/Users/qxy9300/Documents/MA/01_Dataset/INRIA_Person_Dataset_Train_256_summer_winter/pos_winter',type=str,
	help="path to the input images")
ap.add_argument("--output", default='generated',type=str,
	help="path to output directory to store augmentation examples")
ap.add_argument("--total", type=int, default=1,
	help="# of training samples to generate per inout image")
args = ap.parse_args()

def augmentation(args):

	image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
	file_names  = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]

	for i, im in enumerate(image_paths):
		# load the input image, convert it to a NumPy array, and then
		# reshape it to have an extra dimension
		print("[INFO] loading example image...")
		image = cv2.imread(im)
		image =horizontal_flip(image)


		out_name=os.path.join(args.output,file_names[i])
		cv2.imwrite(out_name, image)






def horizontal_flip(image_array):
	return cv2.flip(image_array,1)

if __name__ == "__main__":
	augmentation(args)
