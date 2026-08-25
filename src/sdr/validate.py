

from dataclasses import dataclass

from .models import Brief, Claim


@dataclass(frozen=True)
class Violation:
    field: str
    message: str


def validate_brief(brief: Brief) -> list[Violation]:
    violations: list[Violation] = []
    source_indices = {source.index for source in brief.sources}

    claims: list[tuple[str, Claim]] = [("company", brief.company)]
    if brief.funding is not None:
        claims.append(("funding", brief.funding))
    claims.extend(
        (f"hiring_signals[{i}]", claim) for i, claim in enumerate(brief.hiring_signals)
    )
    claims.append(("likely_use_case", brief.likely_use_case))
    if brief.deal_band is not None:
        claims.append(("deal_band", brief.deal_band))

    verified_evidence: set[int] = set()

    for field, claim in claims:
        for idx in claim.evidence:
            if idx not in source_indices:
                violations.append(Violation(field, f"evidence index {idx} has no matching source"))
        if claim.kind == "verified":
            if not claim.evidence:
                violations.append(Violation(field, "verified claim has no evidence"))
            else:
                verified_evidence.update(claim.evidence)

    for field, claim in claims:
        if claim.kind == "inference" and not set(claim.evidence) & verified_evidence:
            violations.append(
                Violation(field, "inference cites no evidence shared with a verified claim")
            )

    for i, conflict in enumerate(brief.conflicts):
        for idx in conflict.evidence:
            if idx not in source_indices:
                violations.append(
                    Violation(f"conflicts[{i}]", f"evidence index {idx} has no matching source")
                )

    return violations
