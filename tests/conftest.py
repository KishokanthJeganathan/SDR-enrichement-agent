import pytest


@pytest.fixture
def base_brief_dict() -> dict:
    """A minimal, valid Brief as a dict: one verified claim, one inference
    resting on its evidence, one source. Tests mutate a deep copy of this
    to construct the specific violation they want to check for.
    """
    return {
        "company": {
            "value": "Acme Inc",
            "confidence": "HIGH",
            "evidence": [1],
            "kind": "verified",
        },
        "likely_use_case": {
            "value": "Probably wants X",
            "confidence": "MEDIUM",
            "evidence": [1],
            "kind": "inference",
        },
        "suggested_opening": "Hi there",
        "sources": [
            {
                "index": 1,
                "url": "https://acme.example.com",
                "fetched_at": "2026-01-01T00:00:00Z",
            }
        ],
        "overall_confidence": "MEDIUM",
        "generated_at": "2026-01-01T00:00:00Z",
    }
