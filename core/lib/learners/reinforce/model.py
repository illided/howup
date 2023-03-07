import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ...description import ParametersDescription, DiscreteParameterDescription, ContinuousParameterDescription
from typing import Dict, Tuple


Policy = Tuple[Dict[str, Tensor], Dict[str, Tuple[Tensor, Tensor]]]


class REINFORCEModel(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim: int, parameters: ParametersDescription) -> None:
        super().__init__()

        self.common_layer_l1 = nn.Linear(inp_dim, hidden_dim)
        self.common_layer_l2 = nn.Linear(hidden_dim, hidden_dim)

        self.discrete_layers = nn.ModuleDict()
        self.continuous_layers_means = nn.ModuleDict()
        self.continuous_layers_std = nn.ModuleDict()

        for p_name, p_desc in parameters.items():
            if isinstance(p_desc, DiscreteParameterDescription):
                self.discrete_layers[p_name] = nn.Linear(hidden_dim, p_desc.n_categories)
            elif isinstance(p_desc, ContinuousParameterDescription):
                self.continuous_layers_means[p_name] = nn.Linear(hidden_dim, 1)
                self.continuous_layers_std[p_name] = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor) -> Policy:
        x = F.relu(self.common_layer_l1(x))
        x = F.relu(self.common_layer_l2(x))

        discrete_policy = {}
        for p_name, p_layer in self.discrete_layers.items():
            discrete_policy[p_name] = F.softmax(p_layer(x), dim=-1)

        continuous_policy = {}
        for p_name in self.continuous_layers_means.keys():
            mean_layer = self.continuous_layers_means[p_name]
            std_layer = self.continuous_layers_std[p_name]

            mean_policy = F.sigmoid(mean_layer(x))
            std_policy = F.softplus(std_layer(x))

            continuous_policy[p_name] = (mean_policy, std_policy)

        return discrete_policy, continuous_policy
