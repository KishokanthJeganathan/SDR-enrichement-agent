import copy

from sdr.models import Brief
from sdr.validate import validate_brief


def test_valid_brief_has_no_violations(base_brief_dict):
    brief = Brief.model_validate(base_brief_dict)
    assert validate_brief(brief) == []


def test_verified_claim_with_no_evidence_is_rejected(base_brief_dict):
    data = copy.deepcopy(base_brief_dict)
    data["company"]["evidence"] = []

    brief = Brief.model_validate(data)
    violations = validate_brief(brief)

    assert any(v.field == "company" for v in violations)


def test_evidence_index_with_no_matching_source_is_rejected(base_brief_dict):
    data = copy.deepcopy(base_brief_dict)
    data["company"]["evidence"] = [99]

    brief = Brief.model_validate(data)
    violations = validate_brief(brief)

    assert any(v.field == "company" and "99" in v.message for v in violations)


def test_inference_with_no_underlying_verified_claim_is_rejected(base_brief_dict):
    data = copy.deepcopy(base_brief_dict)
    data["sources"].append(
        {"index": 2, "url": "https://other.example.com", "fetched_at": "2026-01-01T00:00:00Z"}
    )
    # Evidence index 2 exists (so this isn't a missing-source violation), but
    # no verified claim rests on it — the inference has nothing to stand on.
    data["likely_use_case"]["evidence"] = [2]

    brief = Brief.model_validate(data)
    violations = validate_brief(brief)

    assert any(v.field == "likely_use_case" for v in violations)
