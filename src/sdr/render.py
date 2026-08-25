import json
import sys
from datetime import datetime
from pathlib import Path

from .models import Brief, Claim, Conflict, Source

_FOOTER = (
    "---\n"
    "*This brief was generated from verified and inferred claims only. "
    "Fields marked LOW confidence or listed under Unverified should be treated "
    "as leads for the call, not facts to repeat back to the prospect.*"
)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _citations(evidence: list[int]) -> str:
    return "".join(f"[{i}]" for i in evidence)


def _claim_line(claim: Claim) -> str:
    lead = "·" if claim.kind == "verified" else "· built on"
    return f"`{claim.confidence}` {lead} {_citations(claim.evidence)}"


def _render_company(claim: Claim) -> str:
    return f"## Company\n\n**{claim.value}**\n{_claim_line(claim)}"


def _render_funding(claim: Claim | None) -> str | None:
    if claim is None:
        return None
    return f"## Funding\n\n**{claim.value}**\n{_claim_line(claim)}"


def _render_hiring_signals(claims: list[Claim]) -> str | None:
    if not claims:
        return None
    lines = "\n".join(f"- {claim.value} {_claim_line(claim)}" for claim in claims)
    return f"## Hiring Signals\n\n{lines}"


def _render_likely_use_case(claim: Claim) -> str:
    return f"## Likely Use Case *(inference)*\n\n{claim.value}\n{_claim_line(claim)}"


def _render_deal_band(claim: Claim | None) -> str | None:
    if claim is None:
        return None
    warning = (
        "> ⚠ **Internal estimate only.** Do not state this figure on the call "
        "or in writing to the prospect."
    )
    return (
        "## Deal Band *(inference — DO NOT QUOTE TO CUSTOMER)*\n\n"
        f"{warning}\n\n{claim.value}\n{_claim_line(claim)}"
    )


def _render_unverified(items: list[str]) -> str | None:
    if not items:
        return None
    lines = "\n".join(f"- {item}" for item in items)
    return (
        "## ⚠ Unverified\n\n"
        "We could not confirm the following. Do not present these as fact on the call.\n\n"
        f"{lines}"
    )


def _render_conflicts(conflicts: list[Conflict]) -> str | None:
    if not conflicts:
        return None
    blocks = []
    for conflict in conflicts:
        positions = "\n".join(f"- {position}" for position in conflict.positions)
        blocks.append(
            f"**Field: {conflict.field}**\n\n{positions}\n- **Resolution:** {conflict.resolution}"
        )
    return "## Conflicts\n\n" + "\n\n".join(blocks)


def _render_suggested_opening(text: str) -> str:
    return f'## Suggested Opening\n\n> "{text}"'


def _render_sources(sources: list[Source]) -> str:
    rows = ["| # | URL | Fetched | Published |", "| - | --- | ------- | --------- |"]
    for source in sources:
        published = _fmt_dt(source.published_at) if source.published_at else "—"
        fetched = _fmt_dt(source.fetched_at)
        rows.append(f"| {source.index} | {source.url} | {fetched} | {published} |")
    return "## Sources\n\n" + "\n".join(rows)


def render_brief(brief: Brief) -> str:
    header = (
        f"# Account Brief: {brief.company.value}\n\n"
        f"Generated {_fmt_dt(brief.generated_at)} · "
        f"Overall confidence: **{brief.overall_confidence}**"
    )
    sections = [
        header,
        _render_unverified(brief.unverified),
        _render_company(brief.company),
        _render_funding(brief.funding),
        _render_hiring_signals(brief.hiring_signals),
        _render_likely_use_case(brief.likely_use_case),
        _render_deal_band(brief.deal_band),
        _render_conflicts(brief.conflicts),
        _render_suggested_opening(brief.suggested_opening),
        _render_sources(brief.sources),
        _FOOTER,
    ]
    return "\n\n".join(section for section in sections if section is not None)


def main() -> None:
    path = Path(sys.argv[1])
    brief = Brief.model_validate(json.loads(path.read_text()))
    print(render_brief(brief))


if __name__ == "__main__":
    main()
