#!/bin/bash

#PBS -N v2v-train-comref_gpu_DAGAN_ped_bs32_im64_epoch5000_G_D_adam
#PBS -q gpu_long
#PBS -l walltime=72:00:00

#PBS -l select=1:ncpus=6:ngpus=1:cluster=konos:scratch_local=0gb:mem=32gb
#PBS -j oe
#PBS -m a
module add python36-modules-gcc
module add cuda-9.0
#module add cudnn-7.0
module add cudnn-7.1.4-cuda90

cd ~/DatasetExtensionBenchmark
source ENV_CYCLE_GAN_TF_1_12/bin/activate
cd DatasetExtensionBenchmark/DAGAN

python3 train_pedestrian_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title 20191221_dagan_pedestrian64 --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5 --epochs 5000 --data_dir data/INRIA_Person_Dataset_Train_256 --im_size 64
