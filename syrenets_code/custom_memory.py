import datetime
import os
import pickle
import torch
import json

class PickleMemory():
    def __init__(self):
        self.datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
        self.path = os.getcwd()+''
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def make_name(self, name):
        return self.path+'/'+self.datetime+name

    def save(self, data, name):
        with open(self.make_name(name), 'wb') as f:
            pickle.dump(data, f)

    def torch_save(self, data, name):
        torch.save(data, self.make_name(name))

    def json_save(self, data, name):
        with open(self.make_name(name), "w", encoding="utf8") as f:
            json.dump(data, f)
    def load(self, name):
        with open(self.path+'/'+name, 'rb') as f:
            pick = pickle.load(f)
        return pick

    def torch_load(self, name):
        return torch.load(self.path+'/'+name)