from abc import abstractmethod, ABC
from ..decision_tree import IDecisionTree


class Learner(ABC):
    @abstractmethod
    def fit(self):
        ...

    @abstractmethod
    def generate_tree(self) -> IDecisionTree:
        ...
