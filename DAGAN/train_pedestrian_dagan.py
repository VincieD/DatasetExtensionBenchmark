import data as dataset
from experiment_builder import ExperimentBuilder
from utils.parser_util import get_args

batch_size, num_gpus, args = get_args()
#set the data provider to use for the experiment
data = dataset.PedestrianDAGANDataset(batch_size=batch_size, last_training_class_index=1200, reverse_channels=False,
                                    num_of_gpus=num_gpus, gen_batches=10)
#init experiment
experiment = ExperimentBuilder(args, data=data)
#run experiment
experiment.run_experiment()
