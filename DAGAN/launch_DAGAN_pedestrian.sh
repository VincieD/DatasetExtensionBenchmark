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
cd ~/DatasetExtensionBenchmark
source ENV_DAGAN/bin/activate
cd DatasetExtensionBenchmark/DAGAN/


python train_pedestrian_dagan.py --data_dir ./data/INRIA_Person_Dataset_Train_128 --im_size 64 --batch_size 32 \
--generator_inner_layers 3 --discriminator_inner_layers 5 \
--num_generations 64 --experiment_title pedestrian_dagan_128 \
--num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5
