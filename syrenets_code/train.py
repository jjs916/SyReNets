from neurips_supplementary import train
import argparse
import os
import configparser
import ast

msg = 'train a model in a given experiment modality'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
parser.add_argument('-c', '--config', help='Name of the config file:', default='math1_debug.ini')
# parser.add_argument('experiment', help='experiment type in string form (\'direct\' or \'indirect\')')
parser.add_argument('-ninp', '--number_inputs', help='number of input to the model (q, dq, ddq)', default=6)
# parser.add_argument('-d', '--depth', help='number of symbolic layers in the network', default=3)
# parser.add_argument('-sl', '--number_selection_heads', help='number of selection heads in each symbolic layer',
#                     default=12)
# parser.add_argument('-mb', '--number_mini_match', help='number of mini batches in training', default=1000)
parser.add_argument('-ns', '--number_samples', help='number of samples in each mini match', default=32)
# parser.add_argument('-niter', '--number_iterations', help='number of training iterations', default=100)
# parser.add_argument('-lambda_2', '--lambda_entropy', help='Lambda hyperparamter for the entorpy loss term', default=0.001)
args = parser.parse_args()

directory = os.path.dirname(os.getcwd())
config_name = args.config
config = configparser.ConfigParser()
config.read(os.path.join(directory, 'config', config_name))
# algorithms = ast.literal_eval(config['META']['algorithms'])
experiment_name = config['META']['experiment_name']
results_path = os.path.join(directory, config['META']['results_path'])

number_selection_heads_list = ast.literal_eval(config['PARAMETERS']['selection_heads'])
depth_list = ast.literal_eval(config['PARAMETERS']['depth'])
n_iterations_list = ast.literal_eval(config['PARAMETERS']['n_iterations'])
mini_batches_list = ast.literal_eval(config['PARAMETERS']['mini_batches'])
lambda_2_list = ast.literal_eval(config['PARAMETERS']['lambda_2'])

train(experiment_name, results_path=results_path, n_inp=int(args.number_inputs), depth_list=depth_list,
      n_sel_list=number_selection_heads_list, n_samples=int(args.number_samples), n_mini_batch_list=mini_batches_list,
      n_iterations_list=n_iterations_list, lambda_entropy_list=lambda_2_list)
