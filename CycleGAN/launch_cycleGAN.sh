#!/bin/bash
#PBS -N v2v-train-comref-cycleGAN_256_winter_summer
#PBS -q gpu
#PBS -l walltime=24:00:00

#PBS -l select=1:ncpus=6:ngpus=1:cluster=konos:scratch_local=0gb:mem=32gb
#PBS -j oe
#PBS -m a

module add python36-modules-gcc
module add cuda-9.0
module add cudnn-7.0
 
# activate virtual enviroment
# on KONOS7
cd ~/DatasetExtensionBenchmark
virtualenv ENV_CYCLE_GAN_TF_1_12
source ENV_CYCLE_GAN_TF_1_12/bin/activate


python train.py --X=data/tfrecords/summer.tfrecords --Y=data/tfrecords/winter.tfrecords --batch_size=1 --image_size=256 --constant_steps=50000 --decay_steps=50000
