from typing import List, Any, Dict, Iterable, Tuple
from dataclasses import dataclass


@dataclass
class DiscreteParameterDescription:
    values: List[Any]

    def decode(self, i: int) -> Any:
        """
        Decode categorical parameter from index

        Arguments:
        i - index. Must be in values indices
        """
        assert i < len(self.values), "Index must be in values indices"
        return self.values[i]

    @property
    def n_categories(self):
        return len(self.values)


@dataclass
class ContinuousParameterDescription:
    min_v: float
    max_v: float

    def scale(self, i: float) -> float:
        """
        Scale continuous parameter from normalized float
        value between 0 and 1.

        Arguments:
        i - normalized value. Must be between 0 and 1
        """
        assert 0.0 <= i <= 1.0, "Normalized value must be between 0 and 1"
        return (self.max_v - self.min_v) * i + self.min_v


@dataclass
class NormalizedParameters:
    discrete: Dict[str, int]
    continuous: Dict[str, float]


class ParametersDescription:
    def __init__(self) -> None:
        self._parameters = {}

    def add_discrete(self, name: str, values: List[Any]) -> 'ParametersDescription':
        self._parameters[name] = DiscreteParameterDescription(values)
        return self

    def add_continuous(self, name: str, min_v: float, max_v: float) -> 'ParametersDescription':
        self._parameters[name] = ContinuousParameterDescription(min_v, max_v)
        return self

    def decode_parameters(self, normalized_parameters: NormalizedParameters) -> Dict[str, Any]:
        decoded: Dict[str, Any] = {}
        for p_name, p_value in normalized_parameters.discrete.items():
            assert p_name in self._parameters, f"Couldn't find discrete parameter with name {p_name}"
            p_decoder = self._parameters[p_name]
            assert isinstance(p_decoder, DiscreteParameterDescription), \
                f"Parameter {p_name} was declared as discrete but passed as continuous"
            decoded[p_name] = self._parameters[p_name].decode(p_value)

        for p_name, p_value in normalized_parameters.continuous.items():
            assert p_name in self._parameters, f"Couldn't find continuous parameter with name {p_name}"
            p_decoder = self._parameters[p_name]
            assert isinstance(p_decoder, ContinuousParameterDescription), \
                f"Parameter {p_name} was declared as continuous but passed as discrete"
            decoded[p_name] = self._parameters[p_name].scale(p_value)
        return decoded

    def items(self) -> Iterable[Tuple[str, DiscreteParameterDescription | ContinuousParameterDescription]]:
        return self._parameters.items()
