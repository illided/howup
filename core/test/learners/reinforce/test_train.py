from ...utils import SimpleEnvironment
from ....lib.learners.reinforce.model import REINFORCEModel
from ....lib.learners.reinforce.train import train_reinforce
from ....lib.learners.reinforce.inference import inference_reinforce
from ....lib.environment import Environment
import pytest


def score_reinforce(model: REINFORCEModel, env: Environment, n_test_episodes: int = 100) -> float:
    score = 0
    for _ in range(n_test_episodes):
        env.reset()
        params = inference_reinforce(model, env.current_state(), env.parameters_description)
        score += env.score(params)
    return score / n_test_episodes


@pytest.mark.repeat(10)
def test_train_on_simple_env():
    env = SimpleEnvironment()
    model = REINFORCEModel(4, 32, env.parameters_description)
    initial_score = score_reinforce(model, env)
    train_reinforce(model, env, 7000, 32)
    score_after_training = score_reinforce(model, env)
    assert score_after_training > initial_score
