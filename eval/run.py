import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))  # so `import judge` works regardless of cwd
from judge import judge_citation, validate_judge  # noqa: E402

from sdr.models import Brief, Claim, Enquiry  # noqa: E402
from sdr.tools.fetch import fetch_text  # noqa: E402
from sdr.trace import RunFailed, RunTrace  # noqa: E402

GOLDEN_PATH = EVAL_DIR / "golden.json"
JUDGE_LABELS_PATH = EVAL_DIR / "judge_labels.json"
JUDGE_ACCURACY_THRESHOLD = 0.8


def _run_fn(agent: str) -> Callable[[Enquiry], RunTrace]:
    if agent == "simple":
        from sdr.agent_simple import run
    elif agent == "deep":
        from sdr.agent_deep import run
    else:
        raise ValueError(f"unknown agent: {agent!r}")
    return run

_STOPWORDS = {"or", "and", "the", "a", "current", "exact"}
_COUNTRY_SYNONYMS = {"US": ["US", "United States", "U.S."]}


def _enquiry_for(entry: dict) -> Enquiry:
    return Enquiry(
        contact_name="SDR Prospect",
        contact_email=f"research@{entry['domain']}",
        company_name=entry["company"],
        message=entry["enquiry_message"],
    )


def _verified_claims(brief: Brief) -> list[tuple[str, Claim]]:
    claims = [("company", brief.company)]
    if brief.funding is not None:
        claims.append(("funding", brief.funding))
    claims.extend((f"hiring_signals[{i}]", c) for i, c in enumerate(brief.hiring_signals))
    return [(field, claim) for field, claim in claims if claim.kind == "verified"]


def _source_url(brief: Brief, index: int) -> str | None:
    return next((s.url for s in brief.sources if s.index == index), None)


def _citation_validity(brief: Brief) -> tuple[float, list[dict]]:
    """For every verified claim, re-fetch its cited source(s) live and judge
    whether at least one actually supports the claim (not just relates to
    it). A claim needs one supporting source, not unanimous support — same
    "at least one" contract as the evidence field itself.
    """
    details = []
    for field, claim in _verified_claims(brief):
        supported, reasoning = False, "no fetchable evidence"
        for index in claim.evidence:
            url = _source_url(brief, index)
            text = fetch_text(url) if url else None
            if not text:
                continue
            supported, reasoning = judge_citation(claim.value, text)
            if supported:
                break
        details.append(
            {"field": field, "claim": claim.value, "supported": supported, "reasoning": reasoning}
        )
    if not details:
        return 1.0, details  # no verified claims to check — vacuously fine
    return sum(d["supported"] for d in details) / len(details), details


def _fact_supported(brief: Brief, key: str, expected) -> bool:
    if key == "has_open_engineering_roles":
        if expected is True:
            return any("engineer" in c.value.lower() for c in brief.hiring_signals)
        return True  # the pilot has no False cases for this field

    text = " ".join(
        [brief.company.value, brief.likely_use_case.value]
        + ([brief.funding.value] if brief.funding else [])
        + ([brief.deal_band.value] if brief.deal_band else [])
        + [c.value for c in brief.hiring_signals]
    ).lower()
    candidates = _COUNTRY_SYNONYMS.get(str(expected), [str(expected)])
    return any(candidate.lower() in text for candidate in candidates)


def _factual_accuracy(brief: Brief, expected_facts: dict) -> tuple[float, dict]:
    if not expected_facts:
        return 1.0, {}
    results = {key: _fact_supported(brief, key, value) for key, value in expected_facts.items()}
    return sum(results.values()) / len(results), results


def _keywords(label: str) -> list[str]:
    return [w for w in label.replace("_", " ").split() if w not in _STOPWORDS]


def _abstention_correctness(brief: Brief, must_abstain: list[str]) -> tuple[float, dict]:
    if not must_abstain:
        return 1.0, {}
    unverified_text = " ".join(brief.unverified).lower()
    results = {
        label: any(kw in unverified_text for kw in _keywords(label)) for label in must_abstain
    }
    return sum(results.values()) / len(results), results


def _conflict_detection(brief: Brief, known_conflicts: list[str]) -> tuple[float, dict]:
    if not known_conflicts:
        return 1.0, {}
    conflict_fields = " ".join(c.field for c in brief.conflicts).lower()
    results = {
        label: any(kw in conflict_fields for kw in _keywords(label)) for label in known_conflicts
    }
    return sum(results.values()) / len(results), results


def _print_misses(label: str, detail: dict) -> None:
    for key, ok in detail.items():
        if not ok:
            print(f"    MISS ({label}): {key}")


def run_eval(agent: str = "simple") -> None:
    run = _run_fn(agent)

    print("=== Validating citation judge against hand-labeled examples ===")
    labels = json.loads(JUDGE_LABELS_PATH.read_text())
    judge_accuracy, mismatches = validate_judge(JUDGE_LABELS_PATH)
    n_correct = len(labels) - len(mismatches)
    print(f"Judge accuracy: {judge_accuracy:.0%} ({n_correct}/{len(labels)} correct)")
    for m in mismatches:
        print(f"  MISS: claim={m['claim']!r} expected={m['expected']} predicted={m['predicted']}")
    if judge_accuracy < JUDGE_ACCURACY_THRESHOLD:
        print(
            f"WARNING: judge accuracy below {JUDGE_ACCURACY_THRESHOLD:.0%} — "
            "citation-validity numbers below are NOT trustworthy.\n"
        )
    else:
        print()

    results_dir = EVAL_DIR / "results" / agent
    results_dir.mkdir(parents=True, exist_ok=True)

    golden = json.loads(GOLDEN_PATH.read_text())
    rows = []
    for entry in golden:
        company = entry["company"]
        print(f"--- {company} ---")
        try:
            trace = run(_enquiry_for(entry))
        except RunFailed as exc:
            print(f"  RUN FAILED: {exc}")
            rows.append({"company": company, "failed": True})
            continue

        brief = trace.brief
        citation_score, citation_detail = _citation_validity(brief)
        accuracy_score, accuracy_detail = _factual_accuracy(
            brief, entry.get("expected_facts", {})
        )
        abstain_score, abstain_detail = _abstention_correctness(
            brief, entry.get("must_abstain", [])
        )
        conflict_score, conflict_detail = _conflict_detection(
            brief, entry.get("known_conflicts", [])
        )

        slug = company.lower().replace(" ", "_")
        (results_dir / f"{slug}.json").write_text(
            json.dumps(
                {
                    "brief": brief.model_dump(mode="json"),
                    "citation_detail": citation_detail,
                    "accuracy_detail": accuracy_detail,
                    "abstain_detail": abstain_detail,
                    "conflict_detail": conflict_detail,
                    "trace": {
                        "repair_attempts": trace.repair_attempts,
                        "initial_violations": trace.initial_violations,
                        "model_calls": trace.model_calls,
                        "tool_calls": trace.tool_calls,
                        "input_tokens": trace.input_tokens,
                        "output_tokens": trace.output_tokens,
                        "estimated_cost_usd": trace.estimated_cost_usd,
                        "latency_seconds": trace.latency_seconds,
                    },
                },
                indent=2,
            )
        )
        for d in citation_detail:
            mark = "OK  " if d["supported"] else "MISS"
            print(f"    {mark} citation[{d['field']}]: {d['reasoning'][:140]}")

        rows.append(
            {
                "company": company,
                "failed": False,
                "citation_validity": citation_score,
                "factual_accuracy": accuracy_score,
                "initial_violations": trace.initial_violations,
                "needed_repair": trace.repair_attempts > 0,
                "abstention_correctness": abstain_score,
                "conflict_detection": conflict_score,
                "model_calls": trace.model_calls,
                "tool_calls": trace.tool_calls,
                "tokens": trace.input_tokens + trace.output_tokens,
                "cost_usd": trace.estimated_cost_usd,
                "latency_seconds": trace.latency_seconds,
            }
        )
        print(
            f"  citation_validity={citation_score:.0%}  factual_accuracy={accuracy_score:.0%}  "
            f"abstention={abstain_score:.0%}  conflict_detection={conflict_score:.0%}  "
            f"initial_violations={trace.initial_violations}  cost=${trace.estimated_cost_usd:.4f}"
        )
        _print_misses("factual_accuracy", accuracy_detail)
        _print_misses("abstention", abstain_detail)
        _print_misses("conflict_detection", conflict_detail)

    print("\n=== Summary ===")
    completed = [r for r in rows if not r["failed"]]
    failed = [r for r in rows if r["failed"]]
    print(f"Runs completed: {len(completed)}/{len(rows)} (failed: {len(failed)})")
    if completed:
        avg = lambda key: sum(r[key] for r in completed) / len(completed)  # noqa: E731
        print(f"Citation validity:        {avg('citation_validity'):.0%}")
        print(f"Factual accuracy:         {avg('factual_accuracy'):.0%}")
        print(f"Avg initial violations:   {avg('initial_violations'):.2f}  (pre-repair, per brief)")
        print(f"Fraction needing repair:  {avg('needed_repair'):.0%}")
        print(f"Abstention correctness:   {avg('abstention_correctness'):.0%}")
        print(f"Conflict detection:       {avg('conflict_detection'):.0%}")
        print(f"Avg model calls/brief:    {avg('model_calls'):.1f}")
        print(f"Avg tool calls/brief:     {avg('tool_calls'):.1f}")
        print(f"Avg tokens/brief:         {avg('tokens'):.0f}")
        print(f"Avg cost/brief:           ${avg('cost_usd'):.4f}")
        print(f"Avg latency/brief:        {avg('latency_seconds'):.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["simple", "deep"], default="simple")
    args = parser.parse_args()
    run_eval(args.agent)
