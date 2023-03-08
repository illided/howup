from .model import REINFORCEModel, Policy
from ...description import ParametersDescription, NormalizedParameters
from numpy import ndarray
import torch


def gready_sampling(policy: Policy) -> NormalizedParameters:
    discrete = {}
    for p_name, p_vector in policy.discrete.items():
        discrete[p_name] = torch.argmax(p_vector).item()
    continuous = {}
    for p_name, p_mean in policy.mean.items():
        continuous[p_name] = p_mean.item()
    return NormalizedParameters(discrete, continuous)


def inference_reinforce(model: REINFORCEModel, state: ndarray, param_desc: ParametersDescription):
    policy = model(torch.from_numpy(state).float())
    action = gready_sampling(policy)
    parameters = param_desc.decode_parameters(action)
    return parameters
