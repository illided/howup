from ..lib.description import ParametersDescription, NormalizedParameters
from .utils import compare_floats


def test_empty():
    desc = ParametersDescription()
    param_values = NormalizedParameters({}, {})
    assert desc.decode_parameters(param_values) == {}


def test_multple_discrete():
    desc = ParametersDescription() \
        .add_discrete("p1", [1, 2, 3]) \
        .add_discrete("p2", [4.0, 5.0, 6.0]) \
        .add_discrete("p3", ["hello", "world"])
    normalized_parameters = NormalizedParameters({"p1": 0, "p2": 1, "p3": 1}, {})
    assert desc.decode_parameters(normalized_parameters) == {"p1": 1, "p2": 5.0, "p3": "world"}


def test_multiple_continuous():
    desc = ParametersDescription() \
        .add_continuous("p1", 0, 10) \
        .add_continuous("p2", -10, 0) \
        .add_continuous("p3", -5, 5)
    target = {"p1": 5, "p2": -7, "p3": 1}
    normalized = NormalizedParameters({}, {"p1": 0.5, "p2": 0.3, "p3": 0.6})
    decoded = desc.decode_parameters(normalized)
    for k, v in target.items():
        assert compare_floats(decoded[k], v)


def test_discrete_wrong_index():
    desc = ParametersDescription().add_discrete('p1', [1, 2])
    try:
        desc.decode_parameters(NormalizedParameters({"p1": 2}, {}))
        assert False
    except AssertionError:
        assert True


def test_continuous_wrong_normalized():
    desc = ParametersDescription().add_continuous("p1", 0, 10)
    try:
        desc.decode_parameters(NormalizedParameters({}, {"p1": 2.0}))
        assert False
    except AssertionError:
        assert True
