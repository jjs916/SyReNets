import torch
import model_learner as ml
from toolbox_functions import get_size


# @torch.jit.script
def outer_ops(inp, inp2):
    # Creates the outer product out of sum, prod, sin and cos
    # Output dimension: n_mini_batch \times (m_sum + m_prod + m_sin + m_cos)
    # m_sum = m_prod = \sum_{k=1}^{n_input+n_selectors) k
    # m_sin = m_cos = n_input+n_selectors
    inp_sum = (inp + inp2) - torch.diag_embed(inp.squeeze(-1))
    triu = torch.triu(torch.ones_like(inp_sum, device=inp.device) == 1)
    inp_sum = inp_sum[triu].view(inp.shape[0], -1)  # x_1, x_1 + x_2, x_1 + x_2, ..., x_2, x_2 + x_3, ...
    inp_prod = inp * inp2
    inp_prod = inp_prod[triu].view(inp.shape[0], -1)  # x_1**2, x_1 * x_2, x_1 * x_2, ..., x_2**2, x_2 * x_3, ...
    inp_sin = torch.sin(inp).squeeze(-1)  # sin(x_1), sin(x_2), ...
    inp_cos = torch.cos(inp).squeeze(-1)  # cos(x_1), cos(x_2)
    return torch.cat([inp_sum, inp_prod, inp_sin, inp_cos], dim=1)


class ParallelHeadNet(torch.nn.Module):
    def __init__(self, n_inp, n_out, n_heads, n_hidden=64, n_sm=1, device=torch.device("cpu")):
        super().__init__()
        self.n_heads = n_heads
        self.sm = torch.nn.functional.softmax
        # Convolutional layer that I cannot see in the paper
        self.q = torch.nn.Conv2d(1 * n_heads, 1 * n_heads, bias=False, kernel_size=(1, n_inp), groups=n_heads)
        self.l1 = torch.nn.Conv1d(n_inp * n_heads, n_hidden * n_heads, bias=False, kernel_size=1, groups=n_heads)
        self.ln_l1 = torch.nn.GroupNorm(n_heads, n_hidden * n_heads)
        self.l2_pi = torch.nn.Conv1d(n_hidden * n_heads, n_sm * n_heads, bias=False, kernel_size=1, groups=n_heads)
        self.l2_n = torch.nn.ModuleList(
            [torch.nn.Conv1d(n_hidden * n_heads, n_out * n_heads, bias=False, kernel_size=1, groups=n_heads) for _ in
             range(n_sm)])
        self.ln_l2_n = torch.nn.ModuleList([torch.nn.GroupNorm(n_heads, n_out * n_heads) for _ in range(n_sm)])
        self.scalar = torch.nn.Conv1d(1 * n_out * n_heads, n_out * n_heads, bias=False, kernel_size=1, groups=n_heads)
        self.on_off = torch.nn.Conv1d(2 * n_out * n_heads, 1 * n_heads, bias=False, kernel_size=1, groups=n_heads)
        self.ones = torch.ones(1, n_out * n_heads, 1, device=device)

    def forward(self, x, last_out_sm):
        out_sm = self._neural_network(x)  # First part of the Selection heads
        inp_last = last_out_sm.prod(1, keepdim=True)  # Product of the distribution from the last layer (Eq. 6)
        inp_current = out_sm
        inp = torch.cat([inp_last.repeat(1, self.n_heads), inp_current], dim=0).t().reshape(-1).unsqueeze(0).unsqueeze(
            -1)  # Put the old and the new distribution together
        # Conv1d (In the paper: Feed forward + Linear) (for constants)
        scalar = self.scalar(self.ones)[0].view(self.n_heads, -1).t()
        # Conv1d + Sigmoid on both distributions together (In the paper: Feed forward + Sigmoid) (Eq. 7)
        # Supposed to be the probability if the distribution of the given head should be used
        on_off = torch.sigmoid(self.on_off(inp).sum(0).view(self.n_heads, -1).t())
        # The product between: constants, the new distributions and the calculations on the old and new distributions
        on_off_out_sm = on_off * out_sm
        out_scaled = on_off_out_sm * scalar  # Eq. 8
        out_scaled[out_scaled.abs() <= 1E-14] = 1E-14
        return on_off_out_sm.clamp(1E-14), out_scaled  # out_scaled is with constants

    def _neural_network(self, x):
        # Repeat the input and then apply conv2d (Think for every head on its own) (In the paper instead of a
        # convolutional layer, a feedforward dxd->dx1 layer is used)
        sub_selection = self.q(x.unsqueeze(0).unsqueeze(0).repeat(1, self.n_heads, 1, 1)).view(1, -1, 1)
        # Apply first another conv1d, then a Group Norm layer and afterwards the softplus function (In the paper instead
        # of a convolutional layer, a feedforward dx1->1xr layer is used) The normalization is on the head level.
        l1 = torch.nn.functional.softplus(self.ln_l1(self.l1(sub_selection).view(1, -1))).view(1, -1, 1)
        # Apply (possible for different cases, but I think the length of this is always 1) a conv1d, a Layer Norm and
        # SoftMax (In the paper instead of a convolutional layer, a feedforward 1xr->1xd_0 layer is used)
        l2 = torch.stack(
            [self.sm(ln_l2_i(l2_i(l1).view(1, -1)).view(self.n_heads, -1).t(), dim=0) for (ln_l2_i, l2_i) in
             zip(self.ln_l2_n, self.l2_n)])
        return l2.sum(0)


class SelectorsNet(torch.nn.Module):
    def __init__(self, n_inp, n_out, n_heads=1, lambda_entropy=0.001, device=torch.device("cpu")):
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.parallel_heads = ParallelHeadNet(n_inp, n_out, n_heads, device=device)

    def _reset_head(self, head_num):
        with torch.no_grad():
            self.parallel_heads.reset_parameters(head_num)

    def forward(self, attention, last_dist):
        new_dists, scaled_dist = self.parallel_heads.forward(attention, last_dist)
        gram_cross_entropy = -new_dists.t() @ torch.log(new_dists + 1E-5)
        # \lambda_2 is fixed as 0.001 here (also the mean is being taken, opposite how it is shown in the paper),
        # increasing it would help make the probability more concentrated; \lambda_3 on the other hand is fixed as 1
        return new_dists, scaled_dist, self.lambda_entropy * gram_cross_entropy.diag().mean() - gram_cross_entropy.fill_diagonal_(
            0).mean()

class Autoencoder(torch.nn.Module):
    def __init__(self, n_inp, n_hidden, n_latent, device=torch.device("cpu")):
        super().__init__()
        self.encoder1 = torch.nn.Linear(n_inp, n_hidden)
        self.encoder2 = torch.nn.Linear(n_hidden, n_hidden)
        self.encoder3 = torch.nn.Linear(n_hidden, n_latent)
        self.decoder1 = torch.nn.Linear(n_latent, n_hidden)
        self.decoder2 = torch.nn.Linear(n_hidden, n_hidden)
        self.decoder3 = torch.nn.Linear(n_hidden, n_inp)

    def cost(self, x, x_hat, z):
        def encoder(x):
            return self._encoder(x).sum(dim=0)

        return (x - x_hat).pow(2).mean() + torch.autograd.functional.jacobian(encoder, x, create_graph=True).pow(
            2).mean()  # torch.autograd.grad(z.sum(), x, retain_graph=True, create_graph=True)[0].abs().sum()

    def _encoder(self, x):
        out = torch.nn.functional.softplus(self.encoder1(x))
        out = torch.nn.functional.softplus(self.encoder2(out))
        return self.encoder3(out)

    def _decoder(self, z):
        out = torch.nn.functional.softplus(self.decoder1(z))
        out = torch.nn.functional.softplus(self.decoder2(out))
        return self.decoder3(out)

    def forward(self, x):
        z = self._encoder(x)
        x_hat = self._decoder(z)
        return z, self.cost(x, x_hat, z)


class IdentityAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x, torch.zeros(1, device=x.device)


class Net(torch.nn.Module):
    def __init__(self, n_inp, depth=1, n_selectors=1, use_autoencoder=True, device=torch.device("cpu"),
                 lambda_entropy=0.01):
        super().__init__()
        self.n_selectors = n_selectors
        self.device = device
        sz = get_size(n_inp + n_selectors)
        self.sz = sz
        latent_sz = max(n_inp, 16)
        self.selectors = torch.nn.ModuleList(
            [SelectorsNet(latent_sz, sz, n_selectors, lambda_entropy=lambda_entropy, device=device) for _ in range(depth)])
        self.last_scale_layer = torch.nn.Linear(1, 1, bias=False)
        self.scalings = []
        self.weights = []
        self.epsilon = torch.tensor(1E-5, device=device)
        self.queries1 = torch.nn.ModuleList([torch.nn.Linear(latent_sz, latent_sz) for _ in range(depth)])
        self.layer_norm = torch.nn.GroupNorm(1, sz)
        if use_autoencoder:
            self.autoencoder = Autoencoder(sz, 128, latent_sz, device=device)
        else:
            self.autoencoder = IdentityAE()
        self.out = None

    def _outer(self, x):
        inp = x.unsqueeze(-1)
        inp2 = x.unsqueeze(-2)
        outer_matrix = outer_ops(inp, inp2)
        return outer_matrix

    def predict(self, x):
        # x = torch.cat([q, dq, ddq], dim=1)
        out = torch.zeros(x.shape[0], self.n_selectors, device=x.device)
        for i, scaling in enumerate(self.scalings.to(x.device)):
            new_x = torch.cat([x, out], dim=1)
            outer_x = self._outer(new_x)
            out = (outer_x @ scaling)
        return out.sum(-1).unsqueeze(-1)

    def forward(self, x):
        l2s = []
        scalars = []
        comp_loss = 0
        # x = torch.cat([q, dq, ddq], dim=1)  # (48,6)
        out = torch.zeros(x.shape[0], self.n_selectors, device=x.device)  # Denotes the output of the previous layer
        l2 = torch.ones(self.sz, self.n_selectors,
                        device=x.device) / self.n_selectors  # Starts with a uniform distribution, but why over the distributions? It should be over the outerproduct...
        for i, (selector, querie1) in enumerate(zip(self.selectors, self.queries1)):
            new_x = torch.cat([x, out], dim=1)  # Concatenate
            raw_outer_x = self._outer(new_x)  # Outer-product
            outer_x, cst = self.autoencoder(self.layer_norm(raw_outer_x))  # Layer Norm and Contractive Auto-Encoder
            outer_x_norm2 = outer_x.abs()
            global_self_attention = (outer_x.t() @ outer_x) / torch.maximum(outer_x_norm2.t() @ outer_x_norm2,
                                                                            self.epsilon)  # Cosine similarity attention
            local_self_attention = (querie1(global_self_attention))  # Feed forward
            l2, scaling, g = selector(local_self_attention, l2)  # Selection head
            l2s.append(l2)
            scalars.append(scaling)
            out = (raw_outer_x @ scalars[
                -1])  # Applies the scaled probabilities to the outer product, however does not sum up
            comp_loss = comp_loss + g + cst
        self.scalings = torch.stack(scalars, dim=0)
        self.weights = l2s
        self.out = out
        result = out.sum(-1).unsqueeze(-1)  # Here the summing up happens, to sum over all terms
        return result, comp_loss


class Syrenets(ml.IModelLearner):
    def __init__(self, n_inp, depth=1, n_selectors=1, lambda_entropy=0.001, use_autoencoder=True, device=torch.device("cpu")):
        self.model = Net(n_inp, depth, n_selectors, use_autoencoder, lambda_entropy=lambda_entropy, device=device).to(device)
        self.params = [0]

    def predict(self, x):
        # q, dq, ddq = x[0], x[1], x[2]  # Right now this method is only written for problems with 3 inputs, plus this is even completely unneccessary as those 3 inputs will be plugged together immediately in the next step by Net
        return self.model.predict(x)

    def learn(self, x, y=None):
        # q, dq, ddq = x[0], x[1], x[2]  # Same here as above
        return self.model(x)
