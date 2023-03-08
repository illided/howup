from ..lib.environment import Environment
from ..lib.description import ParametersDescription
import numpy as np
import random
from typing import Dict, Any


def compare_floats(f1: float, f2: float, e: float = 0.0001):
    return abs(f1 - f2) < e


class SimpleEnvironment(Environment):
    _parameters_description = ParametersDescription() \
        .add_discrete("p1", [1, 2, 3]) \
        .add_discrete("p2", [1, 2, 3]) \
        .add_continuous("p3", min_v=0, max_v=5) \
        .add_continuous("p4", 0, 5)

    @property
    def parameters_description(self) -> ParametersDescription:
        return self._parameters_description

    def reset(self):
        self.p1 = random.randint(1, 3)
        self.p2 = random.randint(1, 3)
        self.p3 = random.random() * 5
        self.p4 = random.random() * 5

    def current_state(self) -> np.ndarray:
        return np.array([self.p1, self.p2, self.p3, self.p4])

    def score(self, parameters: Dict[str, Any]) -> float:
        error = 0
        if parameters["p1"] != self.p1:
            error += 1
        if parameters["p2"] != self.p2:
            error += 1
        error += abs(parameters["p3"] - self.p3) / 5
        error += abs(parameters["p4"] - self.p4) / 5
        return error / 4

