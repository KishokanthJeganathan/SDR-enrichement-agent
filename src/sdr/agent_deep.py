

import time
import uuid
from datetime import date

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from .models import Brief, Enquiry
from .tools import fetch_careers_page, fetch_page, lookup_funding, web_search
from .trace import RunFailed, RunTrace
from .validate import Violation, validate_brief

MODEL = "gpt-5.6-terra"
MAX_REPAIR_ATTEMPTS = 2
# deepagents defaults recursion_limit to 10,000 across nested sub-agent
# delegation — sized for much deeper multi-expert setups than ours (four
# sub-agents, one level deep). This is a blunt safety net on concurrency/
# steps, set low enough to stop a genuine runaway well before 10,000 steps
# of spend, high enough not to truncate four sub-agents doing real
# multi-tool research.
RECURSION_LIMIT = 150

# deepagents auto-adds a `general-purpose` subagent (filesystem tools only,
# but otherwise unscoped and unprompted) to the parent's task tool whenever
# no declared subagent uses that name. Disabled here so the parent's only
# delegation options are the four we designed — exactly four isolated
# sub-agents, not five.
register_harness_profile(
    f"openai:{MODEL}",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)

_HARD_RULES = """Hard rules, non-negotiable:

1. Research the company, not the person who submitted the enquiry. Never \
search for, fetch pages about, or otherwise investigate the contact's name \
or email address. The company's website domain (given to you below) is \
fair game as a place to look for company facts — the person behind that \
email is not.
2. Every claim you make must be backed by at least one source you actually \
fetched. Never state a fact you did not get from a tool result. If you \
can't verify something, say so plainly rather than guessing.
3. For every fact you record, include: the URL you fetched it from, and \
the fetched_at timestamp the tool gave you (and published_at if the \
source states one). Write every date as a full ISO 8601 datetime, e.g. \
'2026-05-14T00:00:00Z' — never a bare year, a partial date like '2026-05', \
or a qualifier tacked onto the string like '2021-10-08 (updated \
2022-04-21)'. If a source only gives an approximate or qualified date, \
describe that in prose elsewhere in the fact — do not put it in place of \
a valid timestamp, and omit published_at entirely rather than write an \
invalid one. A fact without a source and a clean timestamp is useless to \
the parent agent that reads your file — it cannot cite what you don't \
record.
4. If two sources disagree on a fact (e.g. headcount), do not silently \
pick one — record both positions, their sources, and which one (if \
either) looks more reliable and why."""


def _subagent_prompt(focus: str, filename: str) -> str:
    return f"""{_HARD_RULES}

Your specific focus: {focus}

Work efficiently: a handful of well-chosen tool calls beats exhaustively \
searching. You are one of four specialists researching this company in \
parallel — you do not need to verify every possible angle of your focus \
area, just enough to write a useful, honestly-hedged findings file. Once \
you have that, stop searching.

When you are done, call write_file to save your findings to \
`findings/{filename}.md`. Structure it as a markdown list of facts, each \
with its source URL and fetched_at timestamp, so the parent agent can \
build citations from it directly. If you found nothing verifiable for \
something in your focus area, say so explicitly in the file rather than \
omitting it silently."""


_COMPANY_PROFILE_PROMPT = _subagent_prompt(
    "What the company does, its industry, headcount, HQ location, and "
    "stage (startup, growth, established). General company facts only — "
    "funding, hiring, and news are handled by other sub-agents.",
    "company",
)

_FUNDING_PROMPT = _subagent_prompt(
    "Funding history — most recent round, amount, investors, and "
    "valuation if disclosed. Public sources only.",
    "funding",
)

_HIRING_PROMPT = _subagent_prompt(
    "Open engineering and product roles, and what they reveal about the "
    "company's roadmap and priorities. Job postings are the strongest "
    "available signal for what a company is investing in — always check "
    "the careers page.",
    "hiring",
)


def _news_prompt() -> str:
    return _subagent_prompt(
        f"Recent news, announcements, and product launches — the current "
        f"year is {date.today().year}. Prioritize recency: search with "
        f"the current year and terms like 'announcement', 'launch', "
        f"'news'. Older cached knowledge about this company may already "
        f"be stale — your job is to catch what's changed since.",
        "news",
    )


_PARENT_SYSTEM_PROMPT = f"""You are a B2B sales research coordinator. Given \
a company name, delegate research to your four specialist sub-agents and \
synthesize their findings into a sourced account brief for an SDR ahead of \
a call.

{_HARD_RULES}

Process, in order:
1. In a single turn, call the task tool four times — once each for \
company-profile, funding, hiring, and news (use those exact names as \
subagent_type). Do not research anything yourself directly; that's what \
the sub-agents are for.
2. Once all four have finished, read every one of their findings files: \
findings/company.md, findings/funding.md, findings/hiring.md, \
findings/news.md.
3. Build the Brief's `sources` list yourself from every URL any sub-agent \
cited across all four files — number entries sequentially starting at 1, \
using each source's recorded fetched_at (and published_at if given). This \
is your responsibility alone; the sub-agent files do not contain your \
final source numbering.
4. Every `evidence` index in every Claim and Conflict must point at a \
source in the list you just built.
5. Distinguish `kind="verified"` (a sub-agent's file states this as a \
fact it fetched) from `kind="inference"` (you're reasoning from verified \
facts to a judgment call, e.g. likely_use_case or deal_band). An \
inference's `evidence` list must include at least one index a verified \
claim also cites.
6. If a sub-agent's file records two sources disagreeing on a fact, add a \
`conflicts` entry describing both positions rather than silently picking \
one.
7. `company` and `likely_use_case` are required on every brief. `funding` \
and `deal_band` are nullable — use null rather than a low-effort guess if \
nothing verifiable turned up.
8. `deal_band` defaults to LOW confidence unless the sub-agents' findings \
give unusually strong signal — never fabricate the reasoning behind it.
9. If a sub-agent's file reports finding nothing verifiable for \
something, add it to `unverified` rather than silently omitting it."""


class _StatsCallback(BaseCallbackHandler):
    """Model/tool call counter. Must be attached via
    config={"callbacks": [...]} on the top-level agent.invoke() call —
    LangChain threads that config down through deepagents' task tool into
    each sub-agent's own nested run, so one instance here captures cost
    incurred inside every sub-agent too, not just the parent's own
    top-level calls. Verified empirically this session: counting only
    result["messages"] (the approach agent_simple.py uses, which works
    fine for its single flat graph) would miss all sub-agent-internal
    model/tool calls, since those live in a nested subgraph that never
    surfaces in the parent's top-level message list.
    """

    def __init__(self):
        self.model_calls = 0
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._tool_run_ids: set = set()

    def on_llm_end(self, response, **kwargs):
        self.model_calls += 1
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None) if message else None
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._tool_run_ids.add(run_id)

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        # Some of our tools call another traced tool internally
        # (lookup_funding -> web_search -> TavilySearch): LangChain fires
        # on_tool_start/on_tool_end for that inner call too, which would
        # double- or triple-count what was really one model-issued tool
        # call. Only count a tool call whose parent isn't itself a tool
        # call — confirmed via a real LangSmith trace this session, where
        # this pattern inflated tool_calls by 24% (100 counted, 76 real).
        if parent_run_id not in self._tool_run_ids:
            self.tool_calls += 1


def _build_agent():
    return create_deep_agent(
        # gpt-5.6-terra is a reasoning-tier model: function/tool calling on
        # the older Chat Completions endpoint rejects it outright unless
        # reasoning is disabled, so route through the Responses API instead.
        model=ChatOpenAI(model=MODEL, use_responses_api=True),
        # A checkpointer + a stable thread_id (set per run() call, see below)
        # means a StructuredOutputValidationError on the final synthesis
        # step doesn't discard the four sub-agents' completed research: the
        # retry resumes from the last checkpoint instead of re-invoking from
        # scratch. In-memory only — no cross-process persistence needed
        # here, that's Phase 5's concern with a real Postgres checkpointer.
        checkpointer=InMemorySaver(),
        system_prompt=_PARENT_SYSTEM_PROMPT,
        subagents=[
            {
                "name": "company-profile",
                "description": (
                    "Researches what the company does, its industry, "
                    "headcount, HQ location, and stage. Always call this "
                    "for every enquiry."
                ),
                "system_prompt": _COMPANY_PROFILE_PROMPT,
                "tools": [web_search, fetch_page],
            },
            {
                "name": "funding",
                "description": (
                    "Researches funding history — most recent round, "
                    "amount, investors, valuation. Public sources only."
                ),
                "system_prompt": _FUNDING_PROMPT,
                "tools": [web_search, fetch_page, lookup_funding],
            },
            {
                "name": "hiring",
                "description": (
                    "Researches open engineering/product roles and what "
                    "they reveal about the company's roadmap."
                ),
                "system_prompt": _HIRING_PROMPT,
                "tools": [web_search, fetch_page, fetch_careers_page],
            },
            {
                "name": "news",
                "description": (
                    "Researches recent news, announcements, and product "
                    "launches. Recency-focused — catches anything that "
                    "happened after older cached knowledge."
                ),
                "system_prompt": _news_prompt(),
                "tools": [web_search, fetch_page],
            },
        ],
        response_format=Brief,
    )


def _domain_from_email(email: str) -> str:
    return email.rsplit("@", 1)[-1]


def _user_message(enquiry: Enquiry) -> str:
    return (
        f"Company: {enquiry.company_name}\n"
        f"Company website domain (from the contact's email — do not "
        f"research the contact themselves): {_domain_from_email(enquiry.contact_email)}\n"
        f"Enquiry message: {enquiry.message}\n\n"
        "Delegate research to your four sub-agents and produce an account brief."
    )


def _repair_message(violations: list[Violation]) -> str:
    lines = "\n".join(f"- {v.field}: {v.message}" for v in violations)
    return (
        "Your brief failed validation. Fix these specific problems and "
        "return a corrected brief:\n" + lines
    )


def run(enquiry: Enquiry) -> RunTrace:
    agent = _build_agent()
    # One thread per brief. With the checkpointer on _build_agent(), every
    # agent.invoke() call below on this thread_id resumes from the last
    # completed checkpoint rather than starting over — so a repair attempt
    # only has to redo the final synthesis step, not the four sub-agents'
    # research (DeepAgentState's DeltaChannel on messages is built for
    # exactly this: send just the new message, let the checkpoint carry the
    # rest of the accumulated state, including findings/*.md).
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": RECURSION_LIMIT}

    next_input = {"messages": [{"role": "user", "content": _user_message(enquiry)}]}
    start = time.perf_counter()
    model_calls = tool_calls = input_tokens = output_tokens = 0
    initial_violations = 0

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        stats = _StatsCallback()
        try:
            result = agent.invoke(next_input, config={**config, "callbacks": [stats]})
        except StructuredOutputValidationError as exc:
            model_calls += stats.model_calls
            tool_calls += stats.tool_calls
            input_tokens += stats.input_tokens
            output_tokens += stats.output_tokens
            # Pydantic-level shape errors (e.g. a partial date like "2026-05"
            # instead of a full ISO datetime) raise here, inside LangChain's
            # own structured-output parsing — before validate_brief ever
            # gets a Brief to check. The exception unwinds this invoke()
            # call with no result, but the same thread_id means the retry
            # resumes from the last checkpoint (all four sub-agents' work
            # already done) rather than re-researching from scratch.
            malformed_output_violation = Violation("structured_output", str(exc))
            if attempt == MAX_REPAIR_ATTEMPTS:
                raise RunFailed([malformed_output_violation]) from exc
            next_input = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Your structured output failed schema validation: "
                            f"{exc}\n\nEvery datetime field (fetched_at, "
                            "published_at, generated_at) must be a full ISO 8601 "
                            "datetime, e.g. '2026-05-14T00:00:00Z' — not a "
                            "partial date like '2026-05'. If you don't know the "
                            "exact date, omit the field (use null) rather than "
                            "giving a partial one. Everything you already "
                            "found is still available in this thread — fix "
                            "the formatting, do not re-delegate to the "
                            "sub-agents or re-read the findings files."
                        ),
                    }
                ]
            }
            continue

        model_calls += stats.model_calls
        tool_calls += stats.tool_calls
        input_tokens += stats.input_tokens
        output_tokens += stats.output_tokens

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
        next_input = {
            "messages": [
                {
                    "role": "user",
                    "content": _repair_message(violations)
                    + " Everything you already found is still available in "
                    "this thread — fix these specific fields, do not "
                    "re-delegate to the sub-agents or re-read the findings "
                    "files unless the violation genuinely requires new "
                    "information.",
                }
            ]
        }

    raise AssertionError("unreachable")  # loop always returns or raises
