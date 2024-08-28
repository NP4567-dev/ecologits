import pytest
from ecologits import EcoLogits
from ecologits.exceptions import EcoLogitsError


@pytest.mark.skip(reason="Double init does not raise an error anymore, but we should test that it works correctly.")
def test_double_init(tracer_init):
    with pytest.raises(EcoLogitsError) as e:
        EcoLogits.init()   # Second initialization


def test_one_string_publisher():
    EcoLogits.init("openai")

def test_list_publishers():
    EcoLogits.init(["openai", "cohere"])

def test_list_force_none():
    EcoLogits.init(None)

def test_list_publishers_one_wrong(caplog):
    EcoLogits.init(["openbi", "cohere"])
    assert "The following publishers were not found: openbi" in caplog.messages

def test_non_existing_publisher(caplog):
    EcoLogits.init("openbi")
    assert "The following publishers were not found: openbi" in caplog.messages
