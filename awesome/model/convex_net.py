import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awesome.model.real_nvp.resnet_1d import WNLinear, weights_init_uniform, weights_init_normal
from awesome.util.pixelize import pixelize

# Global variable for activation function
activation_function = 'relu'

def apply_activation(weights: torch.Tensor) -> None:
    if activation_function == "relu":
        weights.data = F.relu(weights.data)
    elif activation_function == "softplus":
        weights.data = F.softplus(weights.data)
    elif activation_function == "sigmoid":
        weights.data = torch.sigmoid(weights.data)
    elif activation_function == "tanh":
        weights.data = torch.tanh(weights.data)
    elif activation_function == "exp":
        weights.data = torch.exp(weights.data)
    elif activation_function == "softmax":
        weights.data = F.softmax(weights.data, dim=0)
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")


class ConvexNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_channels: int = 2,
                 **kwargs):
        super().__init__()
        self.W0y = nn.Linear(in_channels, n_hidden)
        self.W1z = nn.Linear(n_hidden, n_hidden)
        self.W2z = nn.Linear(n_hidden, 1)
        self.W1y = nn.Linear(in_channels, n_hidden, bias=False)
        self.W2y = nn.Linear(in_channels, 1, bias=False)

    @pixelize()
    def forward(self, x):
        x_input = x
        x = F.relu(self.W0y(x))
        x = F.relu(self.W1z(x) + self.W1y(x_input))
        x = self.W2z(x) + self.W2y(x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            apply_activation(self.W1z.weight)
            apply_activation(self.W2z.weight)


class SkipBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 in_skip_features: int,
                 **kwargs):
        super().__init__()
        self.ln = nn.Linear(in_features, out_features)
        self.skp = nn.Linear(in_skip_features, out_features, bias=False)

    def forward(self, x, x_input):
        return F.relu(self.ln(x) + self.skp(x_input))

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            apply_activation(self.ln.weight)
            apply_activation(self.skp.weight)


class OutBlock(SkipBlock):
    def forward(self, x, x_input):
        return self.ln(x) + self.skp(x_input)


class ConvexNextNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_features: int = 2,
                 out_features: int = 1,
                 n_hidden_layers: int = 1,
                 **kwargs):
        super().__init__()
        self.input = nn.Linear(in_features, n_hidden)
        self.skip = nn.ModuleList([
            SkipBlock(n_hidden, n_hidden, in_features) for _ in range(n_hidden_layers)
        ])
        self.out = OutBlock(n_hidden, out_features, in_features)

    def forward(self, x):
        x_input = x
        x = F.relu(self.input(x))
        for block in self.skip:
            x = block(x, x_input)
        x = self.out(x, x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            apply_activation(self.input.weight)
            for block in self.skip:
                block.enforce_convexity()
            self.out.enforce_convexity()


# # Example usage
# model = ConvexNextNet()
# activation_function = 'softplus'
# model.enforce_convexity()
