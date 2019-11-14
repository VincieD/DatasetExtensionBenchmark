#!/bin/bash
# Importing all necessary modules in order to setup the virtual enviroment and install TF and Keras as FrontEnd
 
#module add python-3.6.2-gcc
module add python36-modules-gcc
module add cuda-9.0
module add cudnn-7.0
 
# activate virtual enviroment
# on KONOS7
cd ~/DatasetExtensionBenchmark
virtualenv ENV_CYCLE_GAN_TF_1_12
source ENV_CYCLE_GAN_TF_1_12/bin/activate


python train.py --X=data/tfrecords/summer.tfrecords --Y=data/tfrecords/winter.tfrecords --batch_size=1 --image_size=256
