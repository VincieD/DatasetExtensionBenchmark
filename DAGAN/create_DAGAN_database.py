import numpy as np
import os
from pathlib import Path
from os.path import basename
import matplotlib.image as mpimg

# import curses

"""
This script read all files recursively in given directory, create a class of
each dir, and put the images (normalized numpy arrays) in a npy binary format
in the form: (num_classes, num_exemple, h, w, c)
"""

import argparse

parser = argparse.ArgumentParser(description='My image cropper')
parser.add_argument('--input_dir', default='C:\\Users\\qxy9300\\Documents\\MA\\01_Dataset\\INRIA_Person_Dataset_small',
                    help='Input directory', type=str)
parser.add_argument('--output_file',
                    default='C:\\Users\\qxy9300\\Documents\\MA\\01_Dataset\\INRIA_Person_Dataset_small\\my_ped_database.npy',
                    help='Output file', type=str)

args = parser.parse_args()


def create_db(images_dir_path):
    if not Path(images_dir_path).is_dir():
        print('Error')
        exit()
    k = 0
    l = 0
    max_samples_per_class = 0
    # for d in Path(images_dir_path).glob('*'):  # for files/dir in pathdir
    #     # skipping files
    #     if not d.is_dir():
    #         continue;
    #
    #     # path joining version for other paths
    #     DIR = d
    #     nb_samples = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    #     if nb_samples>max_samples_per_class:
    #         max_samples_per_class = nb_samples

    dataset = np.empty((75,20, 128, 128, 3)) # need (num_classes, num_exemple, h, w, c) first 45 are negatives, the remaining 30 are persons/pedestrians
    for d in Path(images_dir_path).glob('*'):  # for files/dir in pathdir
        # skipping files
        if not d.is_dir():
            continue;

        # path joining version for other paths
        DIR = d
        nb_samples = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        # classDataSet = np.empty((2,nb_samples, 128, 128, 3))
        for i, f in enumerate(Path(d).glob('*.png')):
            # print(f)
            img = mpimg.imread(str(f))
            img = img.astype(np.float)
            img /= 255.0
            img = np.reshape(img, newshape=(img.shape[0], img.shape[1], 3))
            dataset[k, l, :, :, :] = img  # list numpy images
            l+=1
            if l>=20:
                l=0
                k=k+1

    print(dataset.shape)
    return np.array(dataset)


def main(args):
    train_dir = args.input_dir
    data = create_db(train_dir)
    np.save(args.output_file, data)


if __name__ == '__main__':
    main(args)
