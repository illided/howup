from abc import ABC, abstractmethod
from typing import Dict, Any
from .description import ParametersDescription
from numpy import ndarray


class Environment(ABC):
    @property
    @abstractmethod
    def parameters_description(self) -> ParametersDescription:
        ...

    @abstractmethod
    def try_parameters(self, parameters: Dict[str, Any]) -> float:
        ...

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def current_state(self) -> ndarray:
        ...
