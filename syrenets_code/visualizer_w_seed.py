from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from custom_memory import PickleMemory
import glob

class ErrorStatistics():
    def __init__(self) -> None:
        self.mse = []
        self.rmse = []
        self.mean = []
        self.std = []
        self.e_max = []
        self.e_min = []
        self.histogram = []
        self.mse_train = []
        self.best_mse_train = []
        self.time_train = []
        self.seed = None

    def save(self):
        all_stats = []
        all_stats.append(self.mse_train)
        all_stats.append(self.best_mse_train)
        all_stats.append(self.time_train)
        all_stats.append(self.mse)
        all_stats.append(self.rmse)
        all_stats.append(self.mean)
        all_stats.append(self.std)
        all_stats.append(self.e_max)
        all_stats.append(self.e_min)
        all_stats.append(self.histogram)
        all_stats.append(self.seed)
        return all_stats

    def load(self, data):
        self.mse_train = data[0]
        self.best_mse_train = data[1]
        self.time_train = data[2]
        self.mse = data[3]
        self.rmse = data[4]
        self.mean = data[5]
        self.std = data[6]
        self.e_max = data[7]
        self.e_min = data[8]
        self.histogram = data[9]
        self.seed = data[10]


class IVisualizer():
    memory: PickleMemory
    def __init__(self):
        raise NotImplementedError

    def generalization_test(self, model, model_x, real_y):
        raise NotImplementedError

    def plot_2d(self, x, y):
        raise NotImplementedError

class LearnerVisualizer(IVisualizer):
    def __init__(self, depth, is_save=True):
        self.is_save = is_save
        self.memory = PickleMemory()
        self.error_stats = ErrorStatistics()
        plt.ion()
        plt.show()
        # fig, axs = plt.subplots(depth, 2)
        # self.axs = [axs] if depth == 1 else axs
        fig2, self.axs2 = plt.subplots(2,2,figsize=(12,9))

    def _reg_seed(self, seed):
        self.error_stats.seed = seed

    def generalization_test(self, model_y, real_y, mse, best_mse, t):
        error = (real_y-model_y)
        self.error_stats.mse_train.append(mse.detach().cpu().item())
        self.error_stats.best_mse_train.append(best_mse.detach().cpu().item())
        self.error_stats.time_train.append(t)
        self.error_stats.mse.append(error.pow(2).mean().detach().cpu().item())
        self.error_stats.rmse.append(error.pow(2).mean().pow(0.5).detach().cpu().item())
        self.error_stats.mean.append(error.mean().detach().cpu().item())
        self.error_stats.std.append(error.std().detach().cpu().item())
        self.error_stats.e_max.append(error.max().detach().cpu().item())
        self.error_stats.e_min.append(error.min().detach().cpu().item())
        self.error_stats.histogram.append(list(np.histogram(error.detach().cpu().numpy(), bins=100)[0]))
        if self.is_save:
            self.memory.save(self.error_stats.save(), 'error_stats.pt')

    def plot_selection(self, best_model):
        pass
    # advanced plot, selection debug
    # def plot_selection(self, best_model):
    #     for i, ax in enumerate(self.axs):
    #         ax[0].clear()
    #         ax[0].plot(best_model.weights[i].detach().cpu())
    #         ax[0].set_ylim([0, 1])
    #         ax[1].clear()
    #         ax[1].plot(best_model.scalings[i].detach().cpu())
    #         ax[1].set_ylim([-10, 10])
    #     plt.show()
    #     plt.pause(0.001)

    def plot_error_stats(self, time=0.001):
        print('model\'s performance on test data:')
        print('mse: {}, rmse: {}, mean: {}\nstd: {}, min: {}, max: {}\n'.format(self.error_stats.mse[-1], \
            self.error_stats.rmse[-1],self.error_stats.mean[-1],self.error_stats.std[-1],self.error_stats.e_min[-1], \
            self.error_stats.e_max[-1]))
        x = range(0, len(self.error_stats.mse)*10, 10)
        t = self.error_stats.time_train
        self.axs2[0,0].clear()
        self.axs2[0,0].bar(np.linspace(self.error_stats.e_min[-1], self.error_stats.e_max[-1], 100).tolist(), self.error_stats.histogram[-1], \
            width=(self.error_stats.e_max[-1]-self.error_stats.e_min[-1])/100)
        self.axs2[0,0].set_title('Error histogram for test data')
        self.axs2[0,0].set_xlabel('Error')
        self.axs2[0,0].set_ylabel('Bins (total = 10000)')
        self.axs2[0,1].clear()
        # self.axs2[0,1].plot(x, np.log(self.error_stats.mse_train), 'b')
        self.axs2[0,1].plot(t, (self.error_stats.best_mse_train), 'g')
        self.axs2[0,1].plot(t, (self.error_stats.mse), color='orange')
        self.axs2[0,1].set_yscale('log')
        self.axs2[0,1].legend(('train data','test data'), loc='upper right')
        self.axs2[0,1].set_title('log scale of MSE')
        self.axs2[0,1].set_xlabel('Time (s)')
        self.axs2[0,1].set_ylabel('Log scale of MSE')
        self.axs2[1,0].clear()
        self.axs2[1,0].plot(x, self.error_stats.rmse)
        self.axs2[1,0].set_yscale('log')
        self.axs2[1,0].set_title('log scale RMSE for test data',y=1.0, pad=-14)
        self.axs2[1,0].set_xlabel('Iterations')
        self.axs2[1,0].set_ylabel('Log scale of RMSE')
        self.axs2[1,1].clear()
        self.axs2[1,1].set_title('Mean and std for test data',y=1.0, pad=-14)
        self.axs2[1,1].errorbar(t, self.error_stats.mean, yerr=self.error_stats.std, fmt='lightsteelblue', linestyle='')
        self.axs2[1,1].plot(t, self.error_stats.mean, 'b')
        self.axs2[1,1].set_xlabel('Time (s)')
        self.axs2[1,1].set_ylabel('Error')
        plt.show()
        plt.pause(0.001)

    def load_save(self, timestamp_str, address='', time=0.001):
        best_model = self.memory.torch_load(address+timestamp_str+'_best_model.pt')
        self.error_stats.load(self.memory.load(address+timestamp_str+'_error_stats.pt'))
        # self.plot_selection(best_model.model)
        self.plot_error_stats(time)
        return best_model

    def load_save_error_stats(self, address=''):
        timestamp_str_list = glob.glob('history/'+address+'*_error_stats.pt')
        memory_list = []
        for ts in timestamp_str_list:
            self.error_stats = ErrorStatistics()
            try:
                self.error_stats.load(self.memory.load(ts[8:]))
            except:
                print('problem with {}'.format(ts))
            memory_list.append(self.error_stats)
            # std.append(evalu.get_train_std(model))
        # self.plot_error_stats()
        return memory_list


        