from custom_memory import PickleMemory
import torch
from visualizer_w_seed import ErrorStatistics
from visualizer_w_seed import LearnerVisualizer
import matplotlib.pyplot as plt

memory = PickleMemory()
device = 'cpu'

model = torch.load('20230131_113004_best_model.pt', map_location=torch.device(device))

scalings = model.model.scalings


n_samples = 3
dim = 2
x = torch.rand(n_samples, 2, device=device, dtype=torch.float64)
y = x[:, 0] ** 2 + x[:, 1]
y
model.predict(x)

error_stats = ErrorStatistics()
error_stats.load(memory.load('20230131_113004_best_model.pt'))
visualizer = LearnerVisualizer(depth=0)
visualizer.error_stats = error_stats
visualizer.plot_error_stats()

scaling = scalings[0]

sum([scalings > 0.001])
scalings_active = [torch.abs(scalings) > 0.001]


def get_outer_names(input_names):
    outer_names = []
    for i, input_name in enumerate(input_names):
        outer_names.append(input_name)
        for input_name_2 in input_names[i + 1:]:
            outer_names.append(f'({input_name} + {input_name_2})')
    for i, input_name in enumerate(input_names):
        outer_names.append(f'{input_name}^2')
        for input_name_2 in input_names[i + 1:]:
            outer_names.append(f'{input_name} * {input_name_2}')
    for input_name in input_names:
        outer_names.append(f'sin({input_name})')
    for input_name in input_names:
        outer_names.append(f'cos({input_name})')
    return outer_names


outer_names = get_outer_names(['x_1', 'x_2', 0, 0])

scaling = scalings[0]

for i in range(scaling.shape[0]):
    if torch.abs(scaling[i, 0]) > 0.001:
        print(f'{scaling[i, 0]} * {outer_names[i]}')


def translate_one_selection_head(outer_names, scaling):
    sum = ''
    for i in range(scaling.shape[0]):
        if torch.abs(scaling[i]) > 0.01:
            print(i)
            sum += f'{scaling[i]} * {outer_names[i]}'
    return sum


translate_one_selection_head(outer_names, scaling[:, 0])
translate_one_selection_head(outer_names, scaling[:, 1])

import copy


def translate_selection_heads(input_names, scalings):
    formula = ''
    n_selector_heads = scalings.shape[2]
    variables = input_names + ['0'] * 2
    for scaling in scalings:
        outer_names = get_outer_names(variables)
        variables = copy.copy(input_names)
        for i in range(n_selector_heads):
            variables.append(translate_one_selection_head(outer_names, scaling[:, i]))
        print(variables)
    return variables


input_names = ['x_1', 'x_2']
translate_selection_heads(input_names, scalings)

input_names

input = ['1']
output = input
output.append('2')
input + ['0'] * 2

outer_names
