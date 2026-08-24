# Phase 4 plan — Deep Agents harness

Handoff doc for picking this up in a fresh Claude Code session (so the
`docs-langchain` / `reference-langchain` MCP tools are actually live —
they were added mid-session here and never connected in this conversation).
Start that session with: "Read CLAUDE.md and docs/phase4-plan.md, then
verify the open API questions below via the docs-langchain/reference-langchain
MCP tools before writing code."

## Goal (from CLAUDE.md §4, §6)

Replace the single Phase 2 agent with four sub-agents running in isolated
contexts — company-profile, funding, hiring, news — each writing findings to
its own file. The parent plans, delegates, reads only the four finished
findings files (never a sub-agent's raw tool-call history), and synthesizes
the `Brief`. Re-run the same Phase 3 eval harness against it; the delta vs.
Phase 2 is the project's headline result.

**Why this specific fix, grounded in what Phase 3 actually found** (see
README's Phase 3 section): the Phase 2 agent's citation drift traces to
`tool_calls / brief` dropping to 7.1 after fixing the robots.txt bug — the
system prompt's "work efficiently" instruction makes it settle for the first
credible source per topic rather than cross-checking. Isolating each
research area into its own sub-agent, with its own tool budget, is meant to
fix exactly that — the funding sub-agent should be free to be thorough about
funding without competing for budget against hiring/news/company research.

## Current codebase state (as of this handoff)

- `agent_simple.py` (Phase 2) is intact and self-contained — do not modify
  it beyond the shared-module extraction below. It's the fixed baseline
  Phase 4 gets compared against.
- **A `trace.py` extraction (pulling `RunTrace`, `RunFailed`,
  `new_message_stats` out of `agent_simple.py` into a shared module) was
  done earlier this session and then reverted** — some IDE/git action
  undid it mid-session (working tree currently matches a clean `git
  status`, and `trace.py` doesn't exist on disk). Redo this first — it's a
  small, low-risk refactor, and `agent_deep.py` needs the same `RunTrace`
  shape to give `eval/run.py` a consistent contract across both agents. Do
  it before anything else so `agent_deep.py` is built against the final
  shared module, not a copy that'll need re-syncing.
- Everything from Phase 3 (`eval/golden.json` — 20 companies,
  `eval/judge.py`, `eval/judge_labels.json` — 30 examples, `eval/run.py`)
  is in place and working. `eval/run.py` currently hardcodes
  `from sdr.agent_simple import run` — needs a small parameterization to
  run against either agent (see checklist).
- Two real bugs were found and fixed in `fetch.py` and `agent_simple.py`
  this session (robots.txt false-positive; a `StructuredOutputValidationError`
  crash with no repair path) — both fixes are still in place and should be
  reused as-is by `agent_deep.py` (same `fetch.py` tools, same
  exception-handling pattern in the repair loop).

## Confirmed API facts (verified via WebFetch this session — re-confirm via MCP docs since WebFetch summarizes/can drop detail)

- Package: `deepagents` (`pip install deepagents`), main entry point
  `create_deep_agent`.
- Signature is close to `langchain.agents.create_agent`'s:
  `create_deep_agent(model=..., tools=..., system_prompt=..., subagents=...,
  response_format=..., ...)`. **`response_format=Brief` works the same way**
  — same repair-loop pattern from `agent_simple.py` transfers directly.
- `subagents` is a list of dicts: `{"name": ..., "description": ...,
  "system_prompt": ..., "tools": [...]}`. The parent delegates via a
  built-in `task` tool, choosing a sub-agent by matching its `description`
  (semantic routing — you don't call sub-agents by name explicitly in code).
- Built-in filesystem tools (`write_file`, `read_file`, `ls`, ...) are
  available automatically — no need to hand-roll `write_findings`/
  `read_findings` as CLAUDE.md's tool table (§5) suggests; the framework
  already provides file I/O.
- **Important correction to CLAUDE.md's architecture description**: a
  sub-agent's output does NOT automatically land in a file. It comes back
  to the parent as plain message text (a `ToolMessage`) unless the
  sub-agent's own `system_prompt` explicitly instructs it to call
  `write_file` before finishing. Each of the four sub-agent prompts needs
  an explicit "when done, write your findings to findings/<area>.md"
  instruction — this is not automatic.
- No documented per-sub-agent step/iteration limit. The stated cost lever
  is scoping each sub-agent's `tools` list narrowly. Plan: give each
  sub-agent only the tools it needs (e.g. funding doesn't get
  `fetch_careers_page`), plus a `recursion_limit` on the *outer*
  `agent.invoke()` call as a blunt safety net across the whole run.
  **Verify via MCP docs**: is there actually no per-sub-agent cap (e.g. a
  `middleware` option, or a field on the `SubAgent` dict) before accepting
  this limitation — the WebFetch summary may have missed something.

## Open questions to resolve via MCP docs before/while writing code

1. **Token/tool-call tracking across sub-agent sub-graphs.** A sub-agent's
   internal messages (its own tool calls, its own model calls) live inside
   its isolated sub-graph and do NOT appear in the parent's top-level
   `result["messages"]` — only the final summary text does. Reading
   `result["messages"]` the way `agent_simple.py` does would badly
   undercount Phase 4's real cost (missing all sub-agent-internal work).
   Confirmed via GitHub source (not the polished reference page) that
   `langchain_core.callbacks.usage.get_usage_metadata_callback` is a
   context manager yielding a `UsageMetadataCallbackHandler` with a
   `.usage_metadata: dict[str, UsageMetadata]` attribute, and that
   LangChain callbacks propagate through nested/sub-graph runs when passed
   via `config={"callbacks": [...]}` — meaning a callback should correctly
   capture sub-agent-internal LLM calls too. Plan was to write a small
   custom `BaseCallbackHandler` subclass (`on_llm_end` for model
   calls/tokens, `on_tool_end` for tool-call counts) rather than using the
   built-in handler alone, since it only tracks tokens, not tool-call
   counts. **Verify via MCP docs**: confirm this callback-propagation
   claim explicitly for deepagents' sub-graph/task-tool architecture
   specifically (not just LangGraph subgraphs generically) — deepagents'
   `task` tool might have its own wrapping that affects this. Also confirm
   `on_llm_end`'s `response: LLMResult` shape
   (`response.generations[i][j].message.usage_metadata`) is still current.
2. **`backend` parameter for the virtual filesystem.** Docs mention options
   including in-memory, local disk, LangGraph store, composite routing.
   Confirm the default (no `backend` passed) is sufficient for our use case
   (ephemeral findings within one run, no cross-run persistence needed) —
   don't want files silently persisting to real disk between eval runs, or
   silently vanishing between the sub-agent's write and the parent's read.
3. **Exact tool names/casing**: confirm `write_file`/`read_file`/`ls` are
   the real tool names an agent would call (or discover — might be
   presented as `ls`, ‑ but double check there isn't a required `edit_file`
   or different read signature), since sub-agent prompts need to reference
   them accurately if any manual guidance is given.

## Implementation checklist

1. Redo the `trace.py` extraction (`RunTrace`, `RunFailed`,
   `new_message_stats`, pricing constants) — reverted mid-session, needs
   to exist before `agent_deep.py` can import from it. Update
   `agent_simple.py` to import from it; confirm `pytest`/`ruff` still pass
   after.
2. Write `agent_deep.py`:
   - Same `MODEL = "gpt-5.6-terra"` as Phase 2, on purpose — isolates the
     architecture variable from a model-choice variable in the comparison.
   - Four sub-agent configs (company-profile, funding, hiring, news), each
     with a scoped `tools` list from `tools/__init__.py`'s exports and a
     `system_prompt` carrying: the same hard rules from `agent_simple.py`'s
     `SYSTEM_PROMPT` (research the company not the contact, cite every
     claim, distinguish verified/inference), its specific research focus,
     and an explicit instruction to call `write_file` on
     `findings/<area>.md` before finishing.
   - News sub-agent is new (Phase 2 doesn't have one) — its focus should be
     explicitly recency-oriented (search with the current year, look for
     "announcement"/"news"/recent product launches), directly targeting
     the staleness pattern Phase 3 found (monday.com's 2019 funding claim,
     Ramp's missed June 2026 round, Airtable's missed August 2026
     acquisition).
   - Parent `system_prompt`: delegate research to all four sub-agents
     (don't research directly), then `read_file` every file under
     `findings/` before synthesizing, then produce the `Brief` via
     `response_format=Brief` — same citation/evidence/conflict rules as
     Phase 2's prompt, adapted to reference "specialist findings" instead
     of "tool results" as the evidence source.
   - Reuse the exact `StructuredOutputValidationError` catch-and-repair
     block from `agent_simple.py`'s `run()` (same bug can happen here too).
   - Custom stats callback (see Open Question 1) wired into every
     `agent.invoke()` call via `config={"callbacks": [...], "recursion_limit": N}`.
   - Same `run(enquiry) -> RunTrace` signature as `agent_simple.run`.
3. Update `eval/run.py` to accept which agent to run against (e.g. a
   `--agent simple|deep` CLI flag, or an env var) rather than hardcoding
   the import — needs to produce two separate result sets
   (`eval/results/simple/` vs `eval/results/deep/`, or similar) so both
   are inspectable after the fact, matching how `eval/results/*.json` was
   invaluable for debugging Phase 3.
4. **Test small before the full run** — same lesson as Phase 3, where a
   crash on company #8 of 20 cost a full re-run. Run `agent_deep.py`
   manually against 1-2 companies via a quick script first (not through
   the full eval harness), confirm it produces a valid `Brief` and that
   `findings/*.md` files actually get written and read back correctly,
   before running the full 20-company comparison.
5. Run the full 20-company eval against Phase 4, compare against Phase 2's
   numbers (README's Phase 3 section has the baseline table), and write up
   the delta honestly — including whether the isolation actually improved
   citation validity/factual accuracy relative to its cost multiplier, not
   just whether it "worked."

## What "done" looks like

- `agent_deep.py` exists, passes the same kind of manual smoke test Phase 2
  got (`sdr.cli`-equivalent single-company run producing a valid, readable
  brief).
- `eval/run.py` can run either agent against the same 20-company golden
  set and judge.
- A full Phase 4 run completes (20/20, ideally 0 crashes — reuse the two
  Phase 3 bug fixes so those specific failure modes don't recur).
- README gets a Phase 4 section with the same rigor as Phase 3's: a
  metrics table, honest analysis of what changed and why (not just "it got
  better"), and the cost/latency multiplier actually measured, not
  estimated.
