"""Phase 2: a single plain LangGraph tool-calling agent. See CLAUDE.md §6.

No sub-agents, no Deep Agents harness, no context isolation. This is
expected to degrade as research grows within one shared context window —
citations drifting, claims attributed to the wrong source. That failure is
the point: it's the documented baseline Phase 4's isolated sub-agents have
to beat.

response_format=Brief gets structural validation for free (required fields,
types, Literal values) via Claude's native structured output. It does not
enforce the citation cross-referencing rules in validate.py — those need the
whole Brief and are business logic, not shape — so this module layers its
own repair loop on top for those specifically.
"""

import time
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from .models import Brief, Enquiry
from .tools import ALL_TOOLS
from .validate import Violation, validate_brief

MODEL = "gpt-5.6-terra"
MAX_REPAIR_ATTEMPTS = 2
INPUT_PRICE_PER_MTOK = 2.0  # gpt-5.6-terra, intro pricing as of Aug 2026
OUTPUT_PRICE_PER_MTOK = 12.0

SYSTEM_PROMPT = """You are a B2B sales research assistant. Given a company \
name, research that COMPANY ONLY and produce a sourced account brief for an \
SDR ahead of a call.

Hard rules, non-negotiable:

1. Research the company, not the person who submitted the enquiry. Never \
search for, fetch pages about, or otherwise investigate the contact's name \
or email address. The company's website domain (derived from the contact's \
email) is fair game as a place to look for company facts — the person \
behind that email is not.
2. Every claim you make must be backed by at least one source you actually \
fetched. Never state a fact you did not get from a tool result. If you \
can't verify something, add it to `unverified` instead of guessing.
3. Distinguish `kind="verified"` (you fetched a source that states this \
directly) from `kind="inference"` (you're reasoning from verified facts to \
a judgment call, e.g. likely_use_case or deal_band). An inference's \
`evidence` list must include at least one index that a verified claim also \
cites — it has to trace back to something you actually confirmed, not just \
correlate with the topic.
4. Build the `sources` list as you go: every URL you fetch (via web_search, \
fetch_page, fetch_careers_page, or lookup_funding) that you end up citing \
gets one entry, numbered sequentially starting at 1, with the fetched_at \
timestamp the tool gave you. Every `evidence` index anywhere in the brief \
must point at a source in this list.
5. `deal_band` is optional and should default to `LOW` confidence unless \
you have unusually strong signal — it is never shown to the customer, so \
err toward including your best estimate with honest low confidence rather \
than omitting it, but never fabricate the underlying reasoning.
6. If two sources disagree on a fact (e.g. headcount), do not silently pick \
one — add an entry to `conflicts` describing both positions and how you're \
resolving the discrepancy (or that you're leaving it unresolved).
7. `company` and `likely_use_case` are required on every brief. `funding` \
and `deal_band` are nullable — use null rather than a low-effort guess if \
you found nothing.
8. Job postings are the strongest available signal for what a company is \
investing in. Always check the careers page before finalizing likely_use_case.

Work efficiently: a handful of well-chosen tool calls beats exhaustively \
searching. When you've gathered enough to write a useful, honestly-hedged \
brief, stop and produce it."""


class RunFailed(Exception):
    def __init__(self, violations: list[Violation]):
        self.violations = violations
        super().__init__(f"brief failed validation after repair attempts: {violations}")


@dataclass
class RunTrace:
    """A Brief plus the trajectory data behind it — how many repair attempts
    it took, how many model/tool calls were made, and what it cost. Phase 3's
    eval harness needs this; Phase 2's CLI just reads .brief off it.
    """

    brief: Brief
    repair_attempts: int  # 0 = passed validation on the first attempt
    initial_violations: int  # violation count on the *first* attempt, pre-repair
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    latency_seconds: float

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_PRICE_PER_MTOK
            + self.output_tokens * OUTPUT_PRICE_PER_MTOK
        ) / 1_000_000


def _new_message_stats(messages: list) -> tuple[int, int, int, int]:
    """Count model/tool calls and tokens in a slice of messages — must be
    called with only the messages a single agent.invoke() call produced, not
    the accumulated history, or repair attempts double-count the prior turn.
    """
    model_calls = tool_calls = input_tokens = output_tokens = 0
    for message in messages:
        if isinstance(message, AIMessage):
            model_calls += 1
            usage = message.usage_metadata or {}
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
        elif isinstance(message, ToolMessage):
            tool_calls += 1
    return model_calls, tool_calls, input_tokens, output_tokens


def _domain_from_email(email: str) -> str:
    return email.rsplit("@", 1)[-1]


def _user_message(enquiry: Enquiry) -> str:
    return (
        f"Company: {enquiry.company_name}\n"
        f"Company website domain (from the contact's email — do not "
        f"research the contact themselves): {_domain_from_email(enquiry.contact_email)}\n"
        f"Enquiry message: {enquiry.message}\n\n"
        "Research this company and produce an account brief."
    )


def _repair_message(violations: list[Violation]) -> str:
    lines = "\n".join(f"- {v.field}: {v.message}" for v in violations)
    return (
        "Your brief failed validation. Fix these specific problems and "
        "return a corrected brief:\n" + lines
    )


def run(enquiry: Enquiry) -> RunTrace:
    agent = create_agent(
        # gpt-5.6-terra is a reasoning-tier model: function/tool calling on
        # the older Chat Completions endpoint rejects it outright unless
        # reasoning is disabled, so route through the Responses API instead.
        model=ChatOpenAI(model=MODEL, use_responses_api=True),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        response_format=Brief,
    )

    messages = [{"role": "user", "content": _user_message(enquiry)}]
    start = time.perf_counter()
    model_calls = tool_calls = input_tokens = output_tokens = 0
    initial_violations = 0

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        prior_len = len(messages)
        result = agent.invoke({"messages": messages})
        mc, tc, itok, otok = _new_message_stats(result["messages"][prior_len:])
        model_calls += mc
        tool_calls += tc
        input_tokens += itok
        output_tokens += otok

        brief = result["structured_response"]
        violations = validate_brief(brief)
        if attempt == 0:
            initial_violations = len(violations)
        if not violations:
            return RunTrace(
                brief=brief,
                repair_attempts=attempt,
                initial_violations=initial_violations,
                model_calls=model_calls,
                tool_calls=tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=time.perf_counter() - start,
            )
        if attempt == MAX_REPAIR_ATTEMPTS:
            raise RunFailed(violations)
        messages = result["messages"] + [{"role": "user", "content": _repair_message(violations)}]

    raise AssertionError("unreachable")  # loop always returns or raises
