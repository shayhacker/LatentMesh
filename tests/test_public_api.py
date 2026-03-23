from __future__ import annotations

import latentmesh as lm


def test_public_api_surface() -> None:
    assert hasattr(lm, "LatentLLM")
    assert hasattr(lm, "LatentGraph")
    assert hasattr(lm, "LatentWorkflow")
    assert hasattr(lm, "LatentServe")
