import torch


class cReLU(torch.nn.Module):
    def forward(self, z):
        if not torch.is_complex(z):
            return torch.nn.ReLU()(z)
        return torch.nn.ReLU()(z.real) + 1j * torch.nn.ReLU()(z.imag)


ACTIVATION_MAP = {
    'relu': torch.nn.ReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'elu': torch.nn.ELU(),
    'gelu': torch.nn.GELU(),
    'geluapproximatenone': torch.nn.GELU(),
    'silu': torch.nn.SiLU(),
    'softplus': torch.nn.Softplus(),
    'identity': torch.nn.Identity(),
    'crelu': cReLU()
}