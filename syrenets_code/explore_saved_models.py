from custom_memory import PickleMemory
import torch
from visualizer_w_seed import ErrorStatistics
from visualizer_w_seed import LearnerVisualizer
import matplotlib.pyplot as plt

memory = PickleMemory()
device = 'cpu'

model = torch.load('20230131_161535_best_model.pt', map_location=torch.device(device))

scalings = model.model.scalings

n_samples = 3
dim = 2
x = torch.rand(n_samples, 2, device=device, dtype=torch.float64)
y = x[:, 0] ** 2 + x[:, 1]
y
model.predict(x)

input_names = ['x_1', 'x_2']

model.get_formula()