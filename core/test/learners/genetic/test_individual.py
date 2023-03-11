from ....lib.learners.genetic.individual import Individual
from ....lib.description import ParametersDescription
from copy import deepcopy
import pytest

def create_example_individual() -> Individual:
    parameters = ParametersDescription() \
        .add_discrete("p1", [1, 2, 3, 4]) \
        .add_continuous("p2", min_v = 0, max_v = 5)
    individual = Individual(parameters)
    return individual


def test_initializing():
    output = create_example_individual().behave()
    assert "p1" in output.discrete
    assert "p2" in output.continuous


def test_mutation_changes_parameters():
    individual = create_example_individual()
    previous_parameters = individual.behave()
    parameters_copy = deepcopy(previous_parameters)
    individual.mutate(ind_mut_pb=1.0)
    new_parameters = individual.behave()
    assert previous_parameters == parameters_copy
    assert new_parameters != previous_parameters


def test_crossover_creates_new_parameters():
    ind1 = create_example_individual()
    ind2 = create_example_individual()

    ind1_params = deepcopy(ind1.behave())
    ind2_params = deepcopy(ind2.behave())

    offspring = ind1.crossover(ind2)

    assert offspring.behave() != ind1_params
    assert offspring.behave() != ind2_params


def test_crossover_keeps_parents_parameters():
    ind1 = create_example_individual()
    ind2 = create_example_individual()

    ind1_params = deepcopy(ind1.behave())
    ind2_params = deepcopy(ind2.behave())

    off1 = ind1.crossover(ind2)
    off2 = ind2.crossover(ind1)
    off1.mutate()
    off2.mutate()

    assert ind1.behave() == ind1_params
    assert ind2.behave() == ind2_params