import torch
from matplotlib.pyplot import pause
import model_learner as ml 
import data_generator as dtgen
import visualizer_w_seed as vis_w_seed
from custom_memory import PickleMemory
from evaluator import Evaluator
import syrenets_learner_20210511 as syrenets_learner

devices = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        devices.append(torch.device("cuda:"+str(i)))
        print("Running on the GPU"+str(i))
else:
    devices.append(torch.device("cpu"))
    print("Running on the CPU")


def run(seed, experiment, is_train=True, file_str='', n_inp=6, depth=3, n_sel=12, n_samples=32, n_mini_batch=1000):
    if experiment == 'direct':
        data = dtgen.Lagrangian(n_inp, n_samples, n_mini_batch, device=devices[0])
    elif experiment == 'indirect':
        data = dtgen.LagrangianTorque(n_inp, n_samples, n_mini_batch, device=devices[0])
    else:
        print('Experiment incorrectly specified')
        raise NotImplementedError
    visualizer_seed = vis_w_seed.LearnerVisualizer(depth,is_save=True)
    visualizer_seed._reg_seed(seed)
    if is_train:    
        syrenets_model = syrenets_learner.Syrenets(n_inp, depth, n_sel, use_autoencoder=True, device=devices[0])
        evaluator = Evaluator(syrenets_model, data, visualizer_seed)
        evaluator.train(50001)
    else:
        syrenets_model = visualizer_seed.load_save(file_str, '', time=100)
        # do something with the model
        # for example:
        print('example of evaluation with new test sampling:')
        evaluator = Evaluator(syrenets_model, data, visualizer_seed)
        mse = evaluator.test()
        print('tested mse: {}'.format(mse))
        pause(100)



def train(experiment: str, n_inp=6, depth=3, n_sel=12, n_samples=32, n_mini_batch=1000):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float64)
    if experiment == 'direct':
        #10 random seeds used for training
        seeds = [14580102103227399718, 17441082891176872975, 2843389671395920724, 3883733847499856553, 17355078531221592105, \
            5820662690886495578, 12002690724912713046, 5653086920993002921, 15151708528336866775, 9670921334992269608]
    elif experiment == 'indirect':
        #10 random seeds used for training
        seeds = [11135967900375438520, 4706712063282134104, 5015929490409337763, 12097533881851730994, 16778785190008374537, \
            513093016403998969, 8044476352635938794, 12365878020051336999, 6692205310915783516, 1557701051186569073]
    else:
        print('Experiment incorrectly specified')
        raise NotImplementedError
    for rd in seeds:
        print('\n#### starting new seed ####\n')
        torch.manual_seed(rd)
        print('random seed is: {}'.format(rd))
        run(rd, experiment, is_train=True, n_inp=n_inp, depth=depth, n_sel=n_sel, n_samples=n_samples, n_mini_batch=n_mini_batch)


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
