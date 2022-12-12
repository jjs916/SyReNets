from neurips_supplementary import train
import argparse
import time

msg = 'train a model in a given experiment modality'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
parser.add_argument('experiment', help='experiment type in string form (\'direct\' or \'indirect\')')
parser.add_argument('-ni','--number_inputs', help='number of input to the model (q, dq, ddq)', default=6)
parser.add_argument('-d','--depth', help='number of symbolic layers in the network', default=3)
parser.add_argument('-sl','--number_selection_heads', help='number of selection heads in each symbolic layer', default=12)
parser.add_argument('-mb','--number_mini_match', help='number of mini batches in training', default=1000)
parser.add_argument('-ns','--number_samples', help='number of samples in each mini match', default=32)
args = parser.parse_args()

train(args.experiment, n_inp=args.number_inputs, depth=args.depth, n_sel=args.number_selection_heads, \
    n_samples=args.number_samples, n_mini_batch=args.number_mini_match)