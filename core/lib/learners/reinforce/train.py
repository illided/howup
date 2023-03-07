import torch
from torch import Tensor
from torch.optim import Adam
from tqdm import trange
import os

from .model import Policy, REINFORCEModel
from ...description import NormalizedParameters
from ...environment import Environment

from typing import Optional


def sample_from_policy(policy: Policy) -> NormalizedParameters:
    discrete_params = {}
    for p_name, p_vector in policy.discrete.items():
        discrete_params[p_name] = torch.distributions.Categorical(probs=p_vector).sample().item()
    continuous_params = {}
    for (p_name, p_mean), p_std in zip(policy.mean.items(), policy.std.values()):
        continuous_params[p_name] = torch.distributions.Normal(p_mean, p_std).sample().clip(0, 1).item()
    return NormalizedParameters(discrete_params, continuous_params)


def discrete_logprob(dist: Tensor, action: int) -> Tensor:
    return torch.log(dist[action])


# this is not an actual probability, but it will suit well during training
def normal_logprob(mu_v: Tensor, var_v: Tensor, actions_v: Tensor) -> Tensor:
    p1 = -((mu_v - actions_v) ** 2 / (2 * var_v.clamp(min=1e-3)))
    p2 = -torch.log(2 * var_v * torch.pi) / 2
    return p1 + p2


def discrete_entropy(dist: Tensor) -> Tensor:
    return -torch.sum(dist * torch.log(dist))


def normal_entropy(var_v: Tensor) -> Tensor:
    return torch.log(torch.e * torch.pi * var_v) / 2


def train_reinforce(model: REINFORCEModel,
                    env: Environment,
                    epochs: int,
                    batch_size: int = 32,
                    save_model_every_n_steps: int = 10,
                    checkpoint_path: Optional[str] = None,
                    entropy_beta: float = 0.05):
    optimizer = Adam(params=model.parameters())
    optimizer.zero_grad()

    # Gradient clipping
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

    loss_sum = torch.Tensor([0])
    for episode in trange(epochs):
        # Slowly decreasing entropy multiplier
        entropy_mult = ((epochs - episode) / epochs) * entropy_beta

        # Forward pass
        env.reset()
        state = env.current_state()
        policy = model(torch.from_numpy(state))
        action = sample_from_policy(policy)
        parameters = env.parameters_description.decode_parameters(action)
        reward = env.try_parameters(parameters)

        # Calculating loss
        loss = torch.Tensor([0]).float()
        for name, sub_action in policy.discrete.items():
            loss += - discrete_logprob(sub_action, action.discrete[name]) * reward
            loss += entropy_mult * discrete_entropy(sub_action)
        for name, mu_v in policy.mean.items():
            std_v = policy.std[name]
            var_v = std_v ** 2
            loss += - normal_logprob(mu_v, var_v, torch.Tensor([action.continuous[name]])) * reward
            loss += entropy_mult * normal_entropy(var_v)
        loss_sum += loss

        # Backward pass
        if episode % batch_size == 0 and episode != 0:
            loss_sum = loss_sum / batch_size
            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum = torch.Tensor([0])

            # Saving model if checkpoint path provided
            if checkpoint_path is not None and episode % (batch_size * save_model_every_n_steps) == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_path, f"epoch_{episode}.pt"))

    # Last backward pass
    loss_sum.backward()
    optimizer.step()
