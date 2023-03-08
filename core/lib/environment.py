from abc import ABC, abstractmethod
from typing import Dict, Any
from .description import ParametersDescription
import numpy as np


class Environment(ABC):
    def __init__(self) -> None:
        self.reset()

    @property
    @abstractmethod
    def parameters_description(self) -> ParametersDescription:
        ...

    @abstractmethod
    def score(self, parameters: Dict[str, Any]) -> float:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def current_state(self) -> np.ndarray:
        ...
