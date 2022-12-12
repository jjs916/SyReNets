from neurips_supplementary import eval
import argparse
import time

msg = 'evaluate a given model in a given experiment modality'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
parser.add_argument('experiment', help='experiment type in string form (\'direct\' or \'indirect\')')
parser.add_argument('model_name', help='string with name of model to use')
args = parser.parse_args()

eval(args.experiment, args.model_name)