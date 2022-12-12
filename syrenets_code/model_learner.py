


class IModelLearner():
    params: list
    def __init__(self, n_inp, depth, n_selectors, use_autoencoder, device):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def learn(self, x, y):
        raise NotImplementedError
