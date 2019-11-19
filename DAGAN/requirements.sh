#!/bin/bash
# Importing all necessary modules in order to setup the virtual enviroment and install TF and Keras as FrontEnd
 
 
#module add python-3.6.2-gcc
module add python36-modules-gcc
module add cuda-9.0
module add cudnn-7.0
 
# activate virtual enviroment
# on KONOS7

virtualenv ENV_DAGAN
source ENV_DAGAN/bin/activate
 
# deactivate virtual enviroment
# deactivate
 
# Step #1: install latest tensorflow and keras as backend
# https://www.tensorflow.org/install/pip?lang=python2

# Redirectiong TEMP
TMPDIR=/storage/plzen1/home/vincie/DatasetExtensionBenchmark/ENV_DAGAN
TMP=$TMPDIR
TEMP=$TMPDIR
export TMPDIR TMP TEMP

 
python3 --version
pip3 --version
virtualenv --version
 
pip3 install --upgrade tensorflow-gpu==1.12.0
python3 -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
 
# Step #2: Install Keras
pip3 install numpy scipy
pip3 install scikit-learn
pip3 install pillow
pip3 install h5py
pip3 install matplotlib
pip3 install imageio
pip3 install tqdm==4.11.2
 
# pip3 install keras
#python -c "import keras;"
 
# Step #2: Install Keras Applications
#mkdir ~/software/
#mkdir ~/software/site-packages
 
#TOIN=~/software/site-packages
#export PYTHONPATH=$TOIN/lib:$PYTHONPATH
#export PATH=$TOIN/bin:$PATH
 
#python setup.py install --install-scripts=$TOIN/bin/ --install-purelib=$TOIN/lib --install-lib=$TOIN/lib






