from ..lib.description import ParametersDescription


def test_empty():
    desc = ParametersDescription()
    assert desc.decode_parameters({}) == {}


def test_multple_discrete():
    desc = ParametersDescription() \
        .add_discrete("p1", [1, 2, 3]) \
        .add_discrete("p2", [4.0, 5.0, 6.0]) \
        .add_discrete("p3", ["hello", "world"])
    assert desc.decode_parameters({"p1": 0, "p2": 1, "p3": 1}) == {"p1": 1, "p2": 5.0, "p3": "world"}


def compare_floats(f1: float, f2: float, e: float = 0.0001):
    return abs(f1 - f2) < e


def test_multiple_continuous():
    desc = ParametersDescription() \
        .add_continuous("p1", 0, 10) \
        .add_continuous("p2", -10, 0) \
        .add_continuous("p3", -5, 5)
    target = {"p1": 5, "p2": -7, "p3": 1}
    normalized = {"p1": 0.5, "p2": 0.3, "p3": 0.6}
    decoded = desc.decode_parameters(normalized)
    for k, v in target.items():
        assert compare_floats(decoded[k], v)


def test_discrete_wrong_index():
    desc = ParametersDescription().add_discrete('p1', [1, 2])
    try:
        desc.decode_parameters({"p1": 4})
        assert False
    except AssertionError:
        assert True


def test_continuous_wrong_normalized():
    desc = ParametersDescription().add_continuous("p1", 0, 10)
    try:
        desc.decode_parameters({"p1": 2.0})
        assert False
    except AssertionError:
        assert True
