import pytest
from unittest.mock import MagicMock
from latentmesh.primitives import ReasonPrimitive, PlanPrimitive, VotingPrimitive, AgentPrimitive
from latentmesh.core import LatentLLM, AgentOutput

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def llm():
    return LatentLLM(model_name=MODEL_NAME, device="cpu", dtype="float32")


def test_agent_primitive_trigger_text(llm):
    """AgentPrimitive injects trigger text and generates output."""
    primitive = AgentPrimitive(name="TestAgent", llm=llm, trigger_text="I am testing: ", max_new_tokens=2)
    state = {"messages": [{"role": "user", "content": "Hello"}], "latent": None}

    result = primitive(state)
    assert "latent" in result
    latent = result["latent"]
    assert isinstance(latent, AgentOutput)
    assert latent.tokens == 2


def test_specialized_primitives(llm):
    """Reason and Plan primitives generate the expected number of tokens."""
    reason = ReasonPrimitive(llm, max_new_tokens=1)
    plan = PlanPrimitive(llm, max_new_tokens=1)

    state = {"messages": [{"role": "user", "content": "How to cook pasta?"}], "latent": None}

    assert reason(state)["latent"].tokens == 1
    assert plan(state)["latent"].tokens == 1


def test_voting_primitive_selects_best_logprob():
    """VotingPrimitive selects the candidate with the highest mean_logprob."""
    cand1 = MagicMock(spec=AgentPrimitive)
    cand2 = MagicMock(spec=AgentPrimitive)

    high_conf = AgentOutput(text="confident answer", tokens=3, debug_text=["mean_logprob:-0.1"])
    low_conf = AgentOutput(text="uncertain answer", tokens=3, debug_text=["mean_logprob:-5.0"])

    cand1.side_effect = lambda s: {"latent": low_conf, "tokens_so_far": 3}
    cand2.side_effect = lambda s: {"latent": high_conf, "tokens_so_far": 3}

    voter = VotingPrimitive(name="Voter", candidates=[cand1, cand2])
    result = voter({"messages": [], "latent": None})

    # Should select cand2 (higher logprob)
    assert result["latent"].text == "confident answer"


def test_voting_primitive_requires_candidates():
    """VotingPrimitive raises if no candidates given."""
    with pytest.raises(ValueError):
        VotingPrimitive(name="Empty", candidates=[])
