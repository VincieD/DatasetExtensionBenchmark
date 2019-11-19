import subprocess as sp

sp.call('python train_omniglot_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title omniglot_dagan_experiment_default --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5')
