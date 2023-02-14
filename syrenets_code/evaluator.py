import torch
import numpy as np
import pickle
import time
import model_learner as ml
import data_generator as dtgen
import visualizer_w_seed as vis_w_seed
from custom_memory import PickleMemory
import matplotlib.pyplot as plt
import os

class Evaluator():
    def __init__(self, path, model: ml.IModelLearner, data: dtgen.IDataGenerator, visualizer=vis_w_seed.IVisualizer):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=1E-3)
        self.best_model = pickle.loads(pickle.dumps(self.model, -1))
        self.data = data
        self.memory = PickleMemory(path)
        visualizer.memory = self.memory
        self.visualizer = visualizer
        self.path = path

    def _check_grad_error(self):
        for param in self.model.model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print('grad is nan or inf')
                raise ValueError

    def train(self, n_iterations):
        with torch.no_grad():
            # Get a first minibatch which will be used as the 'old mini batch'
            model_input, real_out = self.data.get_train_mini_batch(0)
            # Later se will be used to be the standard error, so worst_fit (see below) shows for each mini batch, which
            # training points were the worst performing ones. Later the lower half of them will be remembered and added
            # to the current minibatch to train on them again. Therefore, it is possible to keep a really weird (noisy?)
            # in the current batch forever. Not sure if that is common or smart ...
            se = torch.zeros_like(model_input[:, 0], device=model_input.device)
            best_mse = np.inf  # Initialize the best MSE
            tic = time.perf_counter()  # Time counter
        # Iterations over mini batches
        loss_list = []
        mse_list = []
        for i in torch.arange(n_iterations):
            with torch.no_grad():
                new_x, new_y = self.data.get_train_mini_batch(i)  # Get in every iteration a new minibatch
                # The training examples from the last batch which were the worst fit
                worst_fit = se.argsort(0)[-round(new_x.shape[0] / 2):]
                model_input = torch.cat([new_x, model_input[worst_fit, :]],
                                        dim=0).detach()  # Concatenate the new minibatch with parts (in a different order) of the one above; why?
                real_out = torch.cat([new_y, real_out[worst_fit, :]], dim=0).detach()  # Does the same now to y
            # If we change to arbitrary number of inputs we have to make sure that this will be changed as well,
            # since we cannot use lists later
            model_input = model_input.requires_grad_(True)
            # model_input_leaf = [model_i.requires_grad_(True) for model_i in model_input]
            raw_model_out, o = self.model.learn(model_input, real_out)
            model_out = self.data.transform_output(raw_model_out, model_input)
            se = (real_out - model_out).pow(2).mean(1, keepdim=False)
            mse = se.mean()
            self.model.params = [mse.detach().clone()]
            loss = mse.sum() + o.sum()
            loss_list.append(loss)
            mse_list.append(mse.sum())
            try:
                loss.backward()
                with torch.no_grad():
                    self._check_grad_error()
                self.optimizer.step()
            except:
                pass
            self.optimizer.zero_grad()
            if mse < best_mse:
                with torch.no_grad():
                    cnt = 0
                    self.best_model = pickle.loads(pickle.dumps(self.model, -1))
                    best_mse = mse
                    if self.visualizer.is_save:
                        self.memory.torch_save(self.best_model, 'best_model.pt')
                    x_test, y_test = self.data.get_test_batch()
                # x_test_leaf = [x_test_i.requires_grad_(True) for x_test_i in x_test]
                x_test = x_test.requires_grad_(True)
                raw_prediction = self.best_model.predict(x_test)
                prediction = self.data.transform_output(raw_prediction, x_test)
            else:
                cnt += 1
                if cnt % 2000 == 1999 and self.optimizer.param_groups[0]['lr'] > 1E-5:
                    self.optimizer.param_groups[0]['lr'] *= 0.1
                    print('\n##########\nreduced lr to {}\n##########\n'.format(self.optimizer.param_groups[0]['lr']))
                    cnt = 0
            with torch.no_grad():
                if i % 10 == 0:
                    toc = time.perf_counter()
                    self.visualizer.generalization_test(prediction, y_test, mse, best_mse, toc - tic)
                    if toc - tic >= 2000:
                        print('2000 seconds time exceeded\n')
                        self.visualizer.plot_error_stats()
                        print(
                            'iteration: {}, loss: {:5f}, current train MSE: {}, lowers train MSE: {}, training time: {:0.4f}s'.format(
                                i, loss.item(), mse, best_mse, toc - tic))
                        break
                    if i % 2000 == 0:
                        # self.visualizer.plot_selection(self.best_model.model)
                        self.visualizer.plot_error_stats()
                        print(
                            'iteration: {}, loss: {:5f}, current train MSE: {}, lowers train MSE: {}, training time: {:0.4f}s'.format(
                                i, loss.item(), mse, best_mse, toc - tic))
                    plt.close()
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(
            torch.tensor([loss - min(torch.tensor(loss_list)) + 0.001 for loss in loss_list]).cpu().detach().numpy())
        axs[0, 0].set_yscale('log')
        axs[0, 0].set_title(f'Loss + {- min(torch.tensor(loss_list).cpu().detach().numpy()) + 0.001}')

        axs[0, 1].plot(torch.tensor(mse_list).cpu().detach().numpy())
        axs[0, 1].set_yscale('log')
        axs[0, 1].set_title(f'MSE')

        plt.tight_layout()

        plt.savefig(os.path.join(self.path, self.memory.datetime + 'statistics.png'))
        plt.close()


    def test(self):
        x_test, y_test = self.data.get_test_batch()
        # x_test_leaf = [x_test_i.requires_grad_(True) for x_test_i in x_test]  # Why requires_grad_(True)?
        raw_prediction = self.best_model.predict(x_test)
        prediction = self.data.transform_output(raw_prediction, x_test)
        se = (y_test - prediction).pow(2).mean(1, keepdim=True)
        mse = se.mean()
        return mse

    def get_train_std(self, model):
        with torch.no_grad():
            model_input, real_out = self.data.get_train_mini_batch(0)
            se = torch.zeros_like(model_input[0], device=model_input.device)
            best_mse = np.inf
            tic = time.perf_counter()
        for i in torch.arange(1):
            with torch.no_grad():
                new_x, new_y = self.data.get_train_mini_batch(i)
                worst_fit = se.argsort(0)[-16:, 0]
                model_input = torch.cat([new_x, model_input[:, worst_fit, :]], dim=1).detach()
                real_out = torch.cat([new_y, real_out[worst_fit, :]], dim=0).detach()
            model_input_leaf = [model_i.type(model.model.scalings.dtype).requires_grad_(True) for model_i in
                                model_input]
            # self.optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            raw_model_out = model.model.predict(*model_input_leaf)
            model_out = self.data.transform_output(raw_model_out, model_input_leaf)
            e = (real_out - model_out)
            return e.std()
