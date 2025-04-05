import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awesome.model.real_nvp.resnet_1d import WNLinear, weights_init_uniform, weights_init_normal

from awesome.util.pixelize import pixelize


class ConvexNextNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_features: int = 2,
                 out_features: int = 1,
                 n_hidden_layers: int = 1,
                 activation_function: str = "relu",
                 ** kwargs):
        super().__init__()

        self.input = nn.Linear(in_features, n_hidden)
        self.activation_function = activation_function

        self.skip = nn.ModuleList([
            SkipBlock(in_features=n_hidden, 
                      out_features=n_hidden, 
                      in_skip_features=in_features,
                      activation_function=activation_function)
            for _ in range(n_hidden_layers)
        ])

        self.out = OutBlock(in_features=n_hidden, 
                            out_features=out_features, 
                            in_skip_features=in_features,
                            activation_function=activation_function)

    def forward(self, x):
        x_input = x
        x = F.relu(self.input(x))
        for i in range(len(self.skip)):
            x = self.skip[i](x, x_input=x_input)
        x = self.out(x, x_input=x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            for i in range(len(self.skip)):
                self.skip[i].enforce_convexity(self.activation_function)  # Pass stored activation function
            self.out.enforce_convexity(self.activation_function)  


class WNSkipBlock(nn.Module):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 130,
                 in_skip_features: int = 2,
                 **kwargs) -> None:
        super().__init__()
        self.ln = WNLinear(in_features, out_features)
        self.skp = WNLinear(in_skip_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return F.relu(self.ln(x) + self.skp(x_input))

    def reset_parameters(self) -> None:
        self.ln.reset_parameters('relu')
        self.skp.reset_parameters('relu')

    def enforce_convexity(self) -> None:
        self.ln.linear.weight_v.data = F.relu(self.ln.linear.weight_v.data)
        self.ln.linear.weight_g.data = F.relu(self.ln.linear.weight_g.data)

        self.skp.linear.weight_v.data = F.relu(self.skp.linear.weight_v.data)
        self.skp.linear.weight_g.data = F.relu(self.skp.linear.weight_g.data)


class WNOutBlock(WNSkipBlock):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 1,
                 in_skip_features: int = 2,
                 **kwargs) -> None:
        super().__init__(
            in_features=in_features, 
            out_features=out_features, 
            in_skip_features=in_skip_features
        )

    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return self.ln(x) + self.skp(x_input)

    def reset_parameters(self) -> None:
        self.ln.reset_parameters('linear')
        self.skp.reset_parameters('linear')

# Does not work .... 
class WNConvexNextNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_features: int = 2,
                 out_features: int = 1,
                 n_hidden_layers: int = 1,
                 ** kwargs):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.input = WNLinear(in_features, n_hidden)
        self.skip = nn.ModuleList([
                        WNSkipBlock(in_features=n_hidden, 
                                  out_features=n_hidden, 
                                  in_skip_features=in_features) for _ in range(n_hidden_layers)])
        self.out = WNOutBlock(
            in_features=n_hidden, 
            out_features=out_features, 
            in_skip_features=in_features)

    def reset_parameters(self) -> None:
        self.input.reset_parameters("linear")
        for i in range(len(self.skip)):
            self.skip[i].reset_parameters()
        self.out.reset_parameters()

    @pixelize()
    def forward(self, x):
        # define forward pass
        # Input of shape (batch_size, 2)
        x_input = x
        x = F.relu(self.input(x))
        for i in range(len(self.skip)):
            x = self.skip[i](x, x_input=x_input)
        x = self.out(x, x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            for i in range(len(self.skip)):
                self.skip[i].enforce_convexity()
            self.out.enforce_convexity()



class SkipBlock(nn.Module):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 130,
                 in_skip_features: int = 2,
                 activation_function: str = "relu",
                 **kwargs) -> None:
        super().__init__()
        self.ln = nn.Linear(in_features, out_features)
        self.skp = nn.Linear(in_skip_features, out_features, bias=False)
        self.activation_function = activation_function  # Store activation function type

    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return self.apply_activation(self.ln(x) + self.skp(x_input))

    def apply_activation(self, x):
        if self.activation_function == "relu":
            return F.relu(x)
        elif self.activation_function == "softplus":
            return F.softplus(x)
        elif self.activation_function == "exp":
            return torch.exp(x)
        elif self.activation_function == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation_function == "softmax":
            return F.softmax(x, dim=0)
        elif self.activation_function == "tanh":
            return torch.tanh(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            if self.activation_function == "relu":
                self.ln.weight.data = F.relu(self.ln.weight.data)
                self.skp.weight.data = F.relu(self.skp.weight.data)
            elif self.activation_function == "softplus":
                self.ln.weight.data = F.softplus(self.ln.weight.data)
                self.skp.weight.data = F.softplus(self.skp.weight.data)
            elif self.activation_function == "exp":
                self.ln.weight.data = torch.exp(self.ln.weight.data)
                self.skp.weight.data = torch.exp(self.skp.weight.data)
            elif self.activation_function == "sigmoid":
                self.ln.weight.data = torch.sigmoid(self.ln.weight.data)
                self.skp.weight.data = torch.sigmoid(self.skp.weight.data)
            elif self.activation_function == "softmax":
                self.ln.weight.data = F.softmax(self.ln.weight.data, dim=0)
                self.skp.weight.data = F.softmax(self.skp.weight.data, dim=0)
            elif self.activation_function == "tanh":
                self.ln.weight.data = torch.tanh(self.ln.weight.data)
                self.skp.weight.data = torch.tanh(self.skp.weight.data)





class OutBlock(SkipBlock):
    def __init__(self,
                 in_features: int = 130,
                 out_features: int = 1,
                 in_skip_features: int = 2,
                 activation_function: str = "relu",
                 **kwargs) -> None:
        super().__init__(in_features, out_features, in_skip_features, activation_function)


    def forward(self, x: torch.Tensor, x_input: torch.Tensor):
        return self.ln(x) + self.skp(x_input)

    def reset_parameters(self) -> None:
        self.ln.apply(weights_init_uniform('linear'))
        self.skp.apply(weights_init_uniform('linear'))

class ConvexNextNet(nn.Module):
    def __init__(self,
                 n_hidden: int = 130,
                 in_features: int = 2,
                 out_features: int = 1,
                 n_hidden_layers: int = 1,
                 ** kwargs):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.input = nn.Linear(in_features, n_hidden)
        self.skip = nn.ModuleList([
                        SkipBlock(in_features=n_hidden, 
                                  out_features=n_hidden, 
                                  in_skip_features=in_features) for _ in range(n_hidden_layers)])
        self.out = OutBlock(
            in_features=n_hidden, 
            out_features=out_features, 
            in_skip_features=in_features)

    def reset_parameters(self) -> None:
        self.input.apply(weights_init_uniform('linear'))
        for i in range(len(self.skip)):
            self.skip[i].reset_parameters()
        self.out.reset_parameters()
        return True

    @pixelize()
    def forward(self, x):
        # define forward pass
        # Input of shape (batch_size, 2)
        x_input = x
        x = F.relu(self.input(x))
        for i in range(len(self.skip)):
            x = self.skip[i](x, x_input=x_input)
        x = self.out(x, x_input=x_input)
        return x

    def enforce_convexity(self) -> None:
        with torch.no_grad():
            for i in range(len(self.skip)):
                self.skip[i].enforce_convexity()
            self.out.enforce_convexity()