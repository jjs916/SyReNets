import torch
import numpy as np
import pickle
import time
import model_learner as ml 
import data_generator as dtgen
import visualizer_w_seed as vis_w_seed
from custom_memory import PickleMemory

class Evaluator():
    def __init__(self, model: ml.IModelLearner, data: dtgen.IDataGenerator, visualizer=vis_w_seed.IVisualizer):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.model.parameters(), lr=1E-3)
        self.best_model = pickle.loads(pickle.dumps(self.model, -1))
        self.data = data
        self.memory = PickleMemory()
        visualizer.memory = self.memory
        self.visualizer = visualizer

    def _check_grad_error(self):
        for param in self.model.model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print('grad is nan or inf')
                raise ValueError

    def train(self, n_iterations):
        with torch.no_grad():
            model_input, real_out = self.data.get_train_mini_batch(0)
            se = torch.zeros_like(model_input[0], device=model_input.device)
            best_mse = np.inf
            tic = time.perf_counter()
        for i in torch.arange(n_iterations):
            with torch.no_grad():
                new_x, new_y = self.data.get_train_mini_batch(i)
                worst_fit = se.argsort(0)[-round(new_x.shape[1]/2):,0]
                model_input = torch.cat([new_x, model_input[:,worst_fit,:]],dim=1).detach()
                real_out = torch.cat([new_y, real_out[worst_fit,:]],dim=0).detach()
            model_input_leaf = [model_i.requires_grad_(True) for model_i in model_input]
            raw_model_out, o = self.model.learn(model_input_leaf, real_out)
            model_out = self.data.transform_output(raw_model_out, model_input_leaf)
            se = (real_out - model_out).pow(2).mean(1, keepdim=True)
            mse = se.mean()
            self.model.params = [mse.detach().clone()]
            loss = mse.sum() + o.sum() 
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
                        self.memory.torch_save(self.best_model,'best_model.pt')
                    x_test, y_test = self.data.get_test_batch()
                x_test_leaf = [x_test_i.requires_grad_(True) for x_test_i in x_test]
                raw_prediction = self.best_model.predict(x_test_leaf)
                prediction = self.data.transform_output(raw_prediction, x_test_leaf)
            else:
                cnt += 1
                if cnt%2000 == 1999 and self.optimizer.param_groups[0]['lr'] > 1E-5:
                    self.optimizer.param_groups[0]['lr'] *= 0.1
                    print('\n##########\nreduced lr to {}\n##########\n'.format(self.optimizer.param_groups[0]['lr']))
                    cnt = 0
            with torch.no_grad():
                if i%10==0:
                    toc = time.perf_counter()
                    self.visualizer.generalization_test(prediction, y_test, mse, best_mse, toc-tic)
                    if toc-tic >= 2000:
                        print('2000 seconds time exceeded\n')
                        self.visualizer.plot_error_stats()
                        print('iteration: {}, loss: {:5f}, current train MSE: {}, lowers train MSE: {}, training time: {:0.4f}s'.format(i, loss.item(), mse, best_mse, toc-tic))
                        break
                    if i%2000==0:
                        # self.visualizer.plot_selection(self.best_model.model)
                        self.visualizer.plot_error_stats()
                        print('iteration: {}, loss: {:5f}, current train MSE: {}, lowers train MSE: {}, training time: {:0.4f}s'.format(i, loss.item(), mse, best_mse, toc-tic))

    def test(self):
        x_test, y_test = self.data.get_test_batch()
        x_test_leaf = [x_test_i.requires_grad_(True) for x_test_i in x_test]
        raw_prediction = self.best_model.predict(x_test_leaf)
        prediction = self.data.transform_output(raw_prediction, x_test_leaf)
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
                worst_fit = se.argsort(0)[-16:,0]
                model_input = torch.cat([new_x, model_input[:,worst_fit,:]],dim=1).detach()
                real_out = torch.cat([new_y, real_out[worst_fit,:]],dim=0).detach()
            model_input_leaf = [model_i.type(model.model.scalings.dtype).requires_grad_(True) for model_i in model_input]
            # self.optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            raw_model_out = model.model.predict(*model_input_leaf)
            model_out = self.data.transform_output(raw_model_out, model_input_leaf)
            e = (real_out - model_out) 
            return e.std()

        
