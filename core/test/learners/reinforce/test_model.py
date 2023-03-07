from ....lib.learners.reinforce.model import REINFORCEModel
from ....lib.description import ParametersDescription
from ...utils import compare_floats

import torch
from torch import Tensor
import pytest


def sums_to_one(prob_v: Tensor) -> bool:
    return compare_floats(prob_v.sum().item(), 1)


def test_no_parameters():
    parameters = ParametersDescription()
    model = REINFORCEModel(1, 10, parameters)
    x = torch.rand((1))
    policy = model(x)
    assert policy.discrete == {}


@pytest.mark.repeat(100)
def test_one_discrete_parameter():
    parameters = ParametersDescription().add_discrete("p1", [1, 2, 3])
    model = REINFORCEModel(1, 10, parameters)
    x = torch.rand((1))

    policy = model(x)
    assert len(policy.mean) == 0
    assert len(policy.std) == 0
    assert len(policy.discrete) == 1

    p1_v = policy.discrete["p1"]
    assert list(p1_v.shape) == [3]
    assert sums_to_one(p1_v)


@pytest.mark.repeat(100)
def test_multiple_discrete():
    parameters = ParametersDescription()
    n_discrete_parameters = 100
    for i in range(n_discrete_parameters):
        parameters = parameters.add_discrete(f"p_{i}", [1, 2, 3])

    model = REINFORCEModel(1, 10, parameters)
    x = torch.rand((1))

    policy = model(x)
    assert len(policy.discrete) == n_discrete_parameters
    assert len(policy.mean) == 0
    assert len(policy.std) == 0

    for _, v in policy.discrete.items():
        assert list(v.shape) == [3]
        assert sums_to_one(v)


@pytest.mark.repeat(100)
def test_one_continuous():
    parameters = ParametersDescription()
    parameters.add_continuous("p1", min_v=-5, max_v=5)

    model = REINFORCEModel(1, 10, parameters)
    x = torch.rand((1))

    policy = model(x)
    assert len(policy.discrete) == 0
    assert len(policy.mean) == 1
    assert len(policy.std) == 1

    p1_mean, p1_std = policy.mean["p1"], policy.std["p1"]
    assert 0.0 <= p1_mean.item() <= 1.0
    assert p1_std >= 0.0
