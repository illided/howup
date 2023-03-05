from typing import List, Any, Dict
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


class ParametersDescription:
    def __init__(self) -> None:
        self.parameters = {}

    def add_discrete(self, name: str, values: List[Any]) -> 'ParametersDescription':
        self.parameters[name] = DiscreteParameterDescription(values)
        return self

    def add_continuous(self, name: str, min_v: float, max_v: float) -> 'ParametersDescription':
        self.parameters[name] = ContinuousParameterDescription(min_v, max_v)
        return self

    def decode_parameters(self, normalized_parameters: Dict[str, int | float]) -> Dict[str, Any]:
        decoded: Dict[str, Any] = {}
        for p_name, p_value in normalized_parameters.items():
            assert p_name in self.parameters, f"Parameter with name {p_name} not found in description"
            param_desc = self.parameters[p_name]
            if isinstance(param_desc, ContinuousParameterDescription):
                decoded[p_name] = param_desc.scale(p_value)
            elif isinstance(param_desc, DiscreteParameterDescription):
                assert isinstance(p_value, int), "Discrete parameter must be encoded as integer index"
                decoded[p_name] = param_desc.decode(p_value)
        return decoded
