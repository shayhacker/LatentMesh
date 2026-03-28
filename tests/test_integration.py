import pytest
from latentmesh.core import LatentLLM, AgentOutput
from latentmesh.primitives import ReasonPrimitive, PlanPrimitive, VotingPrimitive
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


@pytest.fixture(scope="module")
def llm():
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)
    return LatentLLM(model_name=MODEL_NAME, device="cpu", dtype="float32", global_cache=cache)


def test_end_to_end_prefix_caching(llm):
    """First generation populates the cache; second generation hits it."""
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    state1 = llm.generate(messages=messages, max_new_tokens=2)

    assert state1.text is not None
    assert state1.tokens > 0
    assert state1.cached_tokens == 0  # cold start

    # The trie should have an entry
    assert len(llm.global_cache._trie) > 0

    # Second call — extend the conversation
    messages.append({"role": "assistant", "content": state1.text})
    messages.append({"role": "user", "content": "Tell me more."})

    state2 = llm.generate(messages=messages, max_new_tokens=2)

    assert state2.text is not None
    assert state2.tokens > 0

    # Cache should have grown
    assert len(llm.global_cache._trie) > 1


def test_voting_primitive_real_execution(llm):
    """VotingPrimitive runs candidates and selects one with valid output."""
    cand1 = ReasonPrimitive(llm, max_new_tokens=1)
    cand2 = PlanPrimitive(llm, max_new_tokens=1)

    voter = VotingPrimitive(name="Voter", candidates=[cand1, cand2])

    graph_state = {"messages": [{"role": "user", "content": "What is 2+2?"}], "latent": None}
    result = voter(graph_state)

    latent = result["latent"]
    assert latent is not None
    assert isinstance(latent.text, str)
    assert latent.tokens > 0
