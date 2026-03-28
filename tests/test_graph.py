import pytest
from latentmesh.graph import latent_reducer, LatentState
from latentmesh.core import AgentOutput


def test_reducer_none_to_value():
    """Updating from None takes the new value."""
    s = AgentOutput(text="First", tokens=5)
    assert latent_reducer(None, s).text == "First"


def test_reducer_value_to_none():
    """Updating with None keeps the old value."""
    s = AgentOutput(text="Existing", tokens=3)
    assert latent_reducer(s, None).text == "Existing"


def test_reducer_merge_text():
    """Two non-None states have their text concatenated."""
    left = AgentOutput(text="First", tokens=5, input_tokens_uncached=10, cached_tokens=0, output_tokens=5)
    right = AgentOutput(text="Second", tokens=10, input_tokens_uncached=3, cached_tokens=7, output_tokens=10)

    merged = latent_reducer(left, right)
    assert merged.text == "First\nSecond"
    assert merged.tokens == 15
    assert merged.input_tokens_uncached == 13
    assert merged.cached_tokens == 7
    assert merged.output_tokens == 15


def test_reducer_no_text():
    """Reducer handles states with None text correctly."""
    empty = AgentOutput(tokens=2)
    full = AgentOutput(text="Hi", tokens=3)

    assert latent_reducer(empty, full).text == "Hi"
    assert latent_reducer(empty, empty).text is None


def test_reducer_both_none():
    """Both None returns a fresh AgentOutput."""
    result = latent_reducer(None, None)
    assert isinstance(result, AgentOutput)
    assert result.tokens == 0
