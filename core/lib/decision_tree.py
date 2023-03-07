from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Dict, Any


class IDecisionTree(ABC):
    @abstractmethod
    def predict(self, state: ndarray) -> Dict[str, Any]:
        ...

    @abstractmethod
    def save(self, filename: str):
        ...

    @abstractmethod
    def pretty_save(self, filename: str):
        ...


class DefaultDecisionTree(IDecisionTree):
    ...
