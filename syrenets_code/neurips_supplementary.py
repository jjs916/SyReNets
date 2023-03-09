import torch
from matplotlib.pyplot import pause
import model_learner as ml
import data_generator as dtgen
import visualizer_w_seed as vis_w_seed
from custom_memory import PickleMemory
from evaluator import Evaluator
import syrenets_learner_20210511 as syrenets_learner
import sympy
from itertools import product
import pandas as pd
import os
import datetime

devices = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        devices.append(torch.device("cuda:" + str(i)))
        print("Running on the GPU" + str(i))
else:
    devices.append(torch.device("cpu"))
    print("Running on the CPU")


def run(seed, experiment, is_train=True, file_str='', n_inp=6, depth=3, n_sel=12, n_samples=32, n_mini_batch=1000,
        n_iterations=100, lambda_entropy=0.01, results_path='', optimizer='adam'):
    if experiment == 'direct':
        data = dtgen.Lagrangian(n_inp, n_samples, n_mini_batch, device=devices[0])
    elif experiment == 'indirect':
        data = dtgen.LagrangianTorque(n_inp, n_samples, n_mini_batch, device=devices[0])
    elif experiment.startswith('math'):
        data = dtgen.MathematicalFormula(n_inp, n_samples, n_mini_batch, device=devices[0],
                                         experiment_name=experiment[4:])
        n_inp = data.n_inp
        input_names = data.input_names
    else:
        print('Experiment incorrectly specified')
        raise NotImplementedError
    visualizer_seed = vis_w_seed.LearnerVisualizer(results_path, depth, is_save=True)
    visualizer_seed._reg_seed(seed)
    if is_train:
        syrenets_model = syrenets_learner.Syrenets(n_inp, depth, n_sel, use_autoencoder=True, input_names=input_names,
                                                   lambda_entropy=lambda_entropy, device=devices[0])
        evaluator = Evaluator(results_path, syrenets_model, data, visualizer_seed, optimizer=optimizer)
        evaluator.train(n_iterations=n_iterations)

        mse = evaluator.test()
        formula = syrenets_model.get_formula()
        formula = round_expr(formula, num_digits=3)
        info = {'experiment': experiment, 'lambda_entropy': lambda_entropy, 'n_inp': n_inp, 'depth': depth,
                'n_selection_heads': n_sel, 'samples': n_samples * n_mini_batch, 'n_iteration': n_iterations,
                'optimizer': optimizer,
                'rmse': torch.sqrt(mse).item(), 'formula': str(formula)}
        print(f'Formula: {formula}')
        evaluator.memory.json_save(info, 'info')
        return info
    else:
        syrenets_model = visualizer_seed.load_save(file_str, '', time=100)
        # do something with the model
        # for example:
        print('example of evaluation with new test sampling:')
        evaluator = Evaluator(syrenets_model, data, visualizer_seed)
        mse = evaluator.test()
        print('tested mse: {}'.format(mse))
        pause(100)


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sympy.Number)})


def train(experiment: str, results_path, n_inp=6, depth_list=[3], n_sel_list=[12], n_samples=32,
          n_mini_batch_list=[1000], optimizers_list=['adam'],
          n_iterations_list=[1000], lambda_entropy_list=[0.001]):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64)
    start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    if experiment == 'direct':
        # 10 random seeds used for training
        seeds = [14580102103227399718, 17441082891176872975, 2843389671395920724, 3883733847499856553,
                 17355078531221592105, \
                 5820662690886495578, 12002690724912713046, 5653086920993002921, 15151708528336866775,
                 9670921334992269608]
    elif experiment == 'indirect':
        # 10 random seeds used for training
        seeds = [11135967900375438520, 4706712063282134104, 5015929490409337763, 12097533881851730994,
                 16778785190008374537, \
                 513093016403998969, 8044476352635938794, 12365878020051336999, 6692205310915783516,
                 1557701051186569073]
    elif experiment.startswith('math'):
        # 10 random seeds used for training
        seeds = [11135967900375438520]
    else:
        print('Experiment incorrectly specified')
        raise NotImplementedError
    for rd in seeds:
        print('\n#### starting new seed ####\n')
        torch.manual_seed(rd)
        print('random seed is: {}'.format(rd))
        info_list = []
        for (depth, n_sel, n_mini_batch, n_iterations, lambda_entropy, optimizer) in product(depth_list, n_sel_list,
                                                                                             n_mini_batch_list,
                                                                                             n_iterations_list,
                                                                                             lambda_entropy_list,
                                                                                             optimizers_list):
            info_list.append(
                run(rd, experiment, is_train=True, n_inp=n_inp, depth=depth, n_sel=n_sel, n_samples=n_samples,
                    n_mini_batch=n_mini_batch, n_iterations=n_iterations, lambda_entropy=lambda_entropy,
                    results_path=results_path, optimizer=optimizer))
            info_df = pd.DataFrame(info_list)
            info_df.to_csv(os.path.join(results_path, experiment + '_' + start_time + '.csv'))


def eval(experiment: str, model_name: str):
    base_name = model_name.split('_best_model.pt')[0]
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64)
    rd = torch.seed()
    torch.manual_seed(rd)
    print('random seed is: {}'.format(rd))
    run(rd, experiment, is_train=False, file_str=base_name)


if __name__ == "__main__":
    train('direct')
