# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, default='C:/Users/qxy9300/Documents/MA/01_Dataset/INRIA_Person_Dataset_Train_256_summer_winter/pos_summer',type=str,
	help="path to the input images")
ap.add_argument("--output", required=True, default='generated',type=str,
	help="path to output directory to store augmentation examples")
ap.add_argument("--total", type=int, default=1,
	help="# of training samples to generate per inout image")
args = ap.parse_args()


image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for i in image_paths:
	# load the input image, convert it to a NumPy array, and then
	# reshape it to have an extra dimension
	print("[INFO] loading example image...")
	image = load_img(args.input)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# construct the image generator for data augmentation then
	# initialize the total number of images generated thus far
	aug = ImageDataGenerator(
		horizontal_flip=True,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		fill_mode="nearest")
	total = 0

	# construct the actual Python generator
	print("[INFO] generating images...")
	imageGen = aug.flow(image, batch_size=1, save_to_dir=args.output,
		save_prefix="image", save_format="png")

	# loop over examples from our image data augmentation generator
	for image in imageGen:
		# increment our counter
		total += 1

		# if we have reached the specified number of examples, break
		# from the loop
		if total == args.total:
			break