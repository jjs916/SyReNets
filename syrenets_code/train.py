from neurips_supplementary import train
import argparse
import time

msg = 'train a model in a given experiment modality'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
parser.add_argument('experiment', help='experiment type in string form (\'direct\' or \'indirect\')')
parser.add_argument('-ninp', '--number_inputs', help='number of input to the model (q, dq, ddq)', default=6)
parser.add_argument('-d', '--depth', help='number of symbolic layers in the network', default=3)
parser.add_argument('-sl', '--number_selection_heads', help='number of selection heads in each symbolic layer',
                    default=12)
parser.add_argument('-mb', '--number_mini_match', help='number of mini batches in training', default=1000)
parser.add_argument('-ns', '--number_samples', help='number of samples in each mini match', default=32)
parser.add_argument('-niter', '--number_iterations', help='number of training iterations', default=100)
parser.add_argument('-lambda_2', '--lambda_entropy', help='Lambda hyperparamter for the entorpy loss term', default=0.001)
args = parser.parse_args()

train(args.experiment, n_inp=int(args.number_inputs), depth=int(args.depth), n_sel=int(args.number_selection_heads),
      n_samples=int(args.number_samples), n_mini_batch=int(args.number_mini_match),
      n_iterations=int(args.number_iterations), lambda_entropy=float(args.lambda_entropy))
