from numpy.core.fromnumeric import var
import torch
import numpy as np
from gradient import gradient

RANGE = [-np.pi, np.pi]
M1, L1 = 3, 2.67
M2, L2 = 1, 1.67

class IDataGenerator():
    x_train: torch.tensor
    y_train: torch.tensor
    x_test: torch.tensor
    y_test: torch.tensor
    def __init__(self, n_inp, n_samples, n_mini_batch, n_test_samples, device=torch.device("cpu")):
        raise NotImplementedError

    def get_random_sample(self, n_samples, is_train=False):
        raise NotImplementedError

    def get_train_mini_batch(self, n):
        raise NotImplementedError

    def get_test_batch(self):
        raise NotImplementedError

    def transform_output(self, function_output, function_input): # for when the output is in a different state space
        return function_output


def calculate_torque(lagrangian, q, qdot, qdotdot, create_graph=True):
    def is_none(variable, zero_like_var):
        if variable is None:
            return torch.zeros_like(zero_like_var, device=zero_like_var.device)
        return variable
    dL_q = torch.autograd.grad(lagrangian.sum(dim=0), q, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
    dL_q = is_none(dL_q,q)
    dL_qdot = torch.autograd.grad(lagrangian.sum(dim=0), qdot, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]
    dL_qdot = is_none(dL_qdot,qdot)
    ddL_qdot_q = gradient(dL_qdot.sum(dim=0), q, create_graph=create_graph, strict=False).transpose(dim0=0, dim1=1)
    ddL_qdot_qdot = gradient(dL_qdot.sum(dim=0), qdot, create_graph=create_graph, strict=False).transpose(dim0=0, dim1=1)
    mass_torque = torch.bmm(ddL_qdot_qdot.transpose(-2,-1), qdotdot.unsqueeze(-1)).squeeze(-1) # torch.einsum('nij, ni -> nj', ddL_qdot_qdot, qdotdot)
    christoffelTorque = torch.bmm(ddL_qdot_q.transpose(-2,-1), qdot.unsqueeze(-1)).squeeze(-1) # torch.einsum('nij, ni -> nj', ddL_qdot_q, qdot)
    return mass_torque + christoffelTorque - dL_q


def func(q, qdot, qdotdot, is_compute_torque=True):
    T = 0.5*(M1/3 + M2)*L1**2*qdot[:,0]**2 + 0.5*(M2/3)*L2**2*qdot[:,1]**2 + 0.5*M2*L1*L2*qdot[:,0]*qdot[:,1]*torch.cos(q[:,0]-q[:,1])
    V = -(M1/2+M2)*9.81*L1*torch.cos(q[:,0]) - (M2/2)*9.81*L2*torch.cos(q[:,1])
    eq = T - V
    if is_compute_torque:
        return calculate_torque(eq, q, qdot, qdotdot)
    return eq.unsqueeze(-1)


class Lagrangian(IDataGenerator):
    def __init__(self, n_inp, n_samples, n_mini_batch, n_test_samples=10000, device=torch.device("cpu")):
        self.n_mini_batch = n_mini_batch
        self.device = device
        self.is_torque = False
        self.x_train, self.y_train = self._random_sample(n_inp, n_samples, n_mini_batch, device, is_train=True)
        self.x_test, self.y_test = self._random_sample(n_inp, n_test_samples, 1, device, is_train=False)

    def _random_sample(self, n_inp, n_samples, n_mini_batch, device, is_train):
        q_full   = torch.rand(n_samples, n_inp//3, n_mini_batch, device=device).uniform_(*RANGE)
        dq_full  = torch.rand(n_samples, n_inp//3, n_mini_batch, device=device).uniform_(*RANGE)
        ddq_full = torch.rand(n_samples, n_inp//3, n_mini_batch, device=device).uniform_(*RANGE)
        q_full   = q_full.requires_grad_(True) 
        dq_full  = dq_full.requires_grad_(True)
        ddq_full = ddq_full.requires_grad_(True)
        real_torque_out_full = func(q_full.transpose(2,1).reshape(-1, n_inp//3), 
                                    dq_full.transpose(2,1).reshape(-1, n_inp//3), 
                                    ddq_full.transpose(2,1).reshape(-1, n_inp//3), self.is_torque).reshape(n_samples, n_mini_batch, -1).transpose(2,1)
        q_full   = q_full.detach().clone()
        dq_full  = dq_full.detach().clone()
        ddq_full = ddq_full.detach().clone()
        real_torque_out_full = real_torque_out_full.detach().clone()
        return torch.stack([q_full, dq_full, ddq_full],dim=0), real_torque_out_full

    def get_random_sample(self, n_samples, is_train=False):
        x, y = self._random_sample(self.x_train.shape[1]*3, n_samples, 1, self.device, is_train)
        return x.squeeze(-1), y.squeeze(-1)

    def get_train_mini_batch(self, n):
        return self.x_train[:,:,:,n%self.n_mini_batch], self.y_train[:,:,n%self.n_mini_batch]
        
    def get_test_batch(self):
        return self.x_test[:,:,:,0], self.y_test[:,:,0]


class LagrangianTorque(Lagrangian):
    def __init__(self, n_inp, n_samples, n_mini_batch, n_test_samples=10000, device=torch.device("cpu")):
        self.n_mini_batch = n_mini_batch
        self.device = device
        self.is_torque = True
        self.x_train, self.y_train = self._random_sample(n_inp, n_samples, n_mini_batch, device, is_train=True)
        self.x_test, self.y_test = self._random_sample(n_inp, n_test_samples, 1, device, is_train=False)

    def transform_output(self, function_output, function_input):
        q, dq, ddq = function_input[0], function_input[1], function_input[2]
        lagrangian = function_output
        return calculate_torque(lagrangian, q, dq, ddq)
