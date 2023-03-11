from ...description import ParametersDescription, NormalizedParameters, DiscreteParameterDescription, ContinuousParameterDescription
import random
from typing import Optional
from copy import copy
from .crossover import blend_crossover


class Individual:
    def __init__(self,
                 parameters_desc: ParametersDescription,
                 parameters_values: Optional[NormalizedParameters] = None):
        self._discrete = {}
        self._continuous = {}
        self._param_desc = parameters_desc

        if parameters_values is None:
            self.mutate(ind_mut_pb=1.0)
        else:
            self._discrete = copy(parameters_values.discrete)
            self._continuous = copy(parameters_values.continuous)

    def mutate(self, ind_mut_pb=0.5):
        for p_name, p_decoder in self._param_desc.items():
            if random.random() <= ind_mut_pb:
                if isinstance(p_decoder, DiscreteParameterDescription):
                    self._discrete[p_name] = random.randint(0, p_decoder.n_categories)
                elif isinstance(p_decoder, ContinuousParameterDescription):
                    self._continuous[p_name] = random.random()
                else:
                    raise TypeError("Unknown parameter type")

    def crossover(self, other: 'Individual') -> 'Individual':
        new_discrete = {}
        new_continuous = {}
        other_parameters = other.behave()
        for p_name, other_p_value in other_parameters.discrete.items():
            new_discrete[p_name] = random.choice([self._discrete[p_name], other_p_value])
        for p_name, other_p_value in other_parameters.continuous.items():
            new_continuous[p_name] = blend_crossover(self._continuous[p_name], other_p_value, lower_b=0.0, upper_b=1.0)
        return Individual(self._param_desc, NormalizedParameters(new_discrete, new_continuous))

    def behave(self) -> NormalizedParameters:
        return NormalizedParameters(copy(self._discrete), copy(self._continuous))
