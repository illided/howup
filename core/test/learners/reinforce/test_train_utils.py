from ....lib.learners.reinforce.train import sample_from_policy, discrete_logprob, normal_logprob, discrete_entropy, normal_entropy
from ....lib.learners.reinforce.model import Policy
import random
import pytest
import torch
from collections import defaultdict
from ...utils import compare_floats


@pytest.mark.repeat(100)
def test_sampling_from_discrete_degenerate():
    choosed_action = random.randint(0, 9)
    dist = torch.zeros((10)).float()
    dist[choosed_action] = 1.0

    policy = Policy({"p1": dist}, {}, {})
    assert sample_from_policy(policy).discrete == {"p1": choosed_action}


def test_sampling_from_discrete_uniform():
    dist = torch.full(size=[10], fill_value=0.1)
    policy = Policy({"p1": dist}, {}, {})

    sampled_actions = defaultdict(lambda: 0)
    for _ in range(1_000):
        action = sample_from_policy(policy).discrete["p1"]
        sampled_actions[action] += 1

    for i in range(10):
        assert sampled_actions[i] != 0


def test_sampling_from_continuous_zero_std():
    policy = Policy({}, {"p1": torch.Tensor([0.5])}, {"p1": torch.Tensor([0.00001])})
    action = sample_from_policy(policy).continuous["p1"]
    assert compare_floats(action, 0.5)


def test_sampling_from_continuous_nonzero_std():
    policy = Policy({}, {"p1": torch.Tensor([0.5])}, {"p1": torch.Tensor([1.0])})
    sampled_actions = []
    for _ in range(1_000):
        action = sample_from_policy(policy).continuous["p1"]
        sampled_actions.append(action)
        assert 0.0 <= action <= 1.0
    rounded = [round(a, 1) for a in sampled_actions]

    for i in range(0, 10):
        assert i / 10 in rounded


def test_logprob_from_degenerate_discrete():
    dist = torch.zeros(10)
    dist[0] = 1.0
    assert compare_floats(discrete_logprob(dist, 0).item(), 0.0)


def test_logprob_from_uniform_discrete():
    dist = torch.full(size=[10], fill_value=0.1)
    assert compare_floats(discrete_logprob(dist, 0).item(), -2.30258)


def test_logprob_from_degenerate_continuous():
    mu_v = torch.Tensor([0.5])
    var_v = torch.Tensor([0.001])
    action = torch.Tensor([0.5])
    assert compare_floats(normal_logprob(mu_v, var_v, action).item(), 2.5349)


def test_logprob_from_nontrivial_continuous():
    mu_v = torch.Tensor([0.5])
    var_v = torch.Tensor([0.1])
    action = torch.Tensor([0.5])
    assert compare_floats(normal_logprob(mu_v, var_v, action).item(), 0.23235)


@pytest.mark.repeat(100)
def test_discrete_entropy_with_comparison():
    uniform_dist = torch.softmax(torch.rand(10), -1)
    
    spiked_uniform = torch.rand((10))
    spiked_uniform[0] = 10
    spiked_uniform = torch.softmax(spiked_uniform, -1)
    assert discrete_entropy(uniform_dist) > discrete_entropy(spiked_uniform)


@pytest.mark.repeat(100)
def test_continuous_entropy_with_comparison():
    v1, v2 = random.random(), random.random()
    v1, v2 = min(v1, v2), max(v1, v2)
    assert normal_entropy(torch.Tensor([v2])) > normal_entropy(torch.Tensor([v1]))