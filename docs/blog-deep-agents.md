# One Agent vs. Four Agents: The Pros and Cons of Context Isolation

I've been building an SDR research agent — an inbound enquiry goes in, a sourced account brief comes out. The interesting problem isn't finding facts about a company; any agent with a search tool can do that. It's knowing which of those facts you're actually allowed to assert: every claim traces back to a source, every abstention is explicit, every conflict between sources gets surfaced instead of silently resolved. That's the real engineering problem, and it's a harder one than it sounds.

It's not a hypothetical problem, either. The AI-SDR space got its own cautionary tale in 2025-2026: 11x.ai, once valued in the hundreds of millions on the promise of a fully autonomous AI SDR, lost 70-80% of its logos after customers found fabricated case studies in its own marketing — bad enough that ZoomInfo sent it a legal threat over it. The market's read on that wasn't "AI SDRs don't work," it was "don't ship one whose outputs a rep can't check." The whole industry narrative shifted from *replace the SDR* to *give the SDR a tool they can actually trust going into a call* — not one they have to fact-check afterward. Salesforce's own Agentforce SDR agent researches prospects grounded in Data Cloud and Customer 360, which is real capability — but grounding data isn't the same as showing a rep, per claim, where a specific number came from or whether the agent is confident or just filling a gap. Most 2026 buyer guides for these tools boil down to "ask the vendor what happens when it's wrong," which is a strange question to still have to ask. So instead of asking, I built the answer into the schema: every claim carries its source, every gap is named instead of guessed at, and nothing ships without both.

That's the actual point of this project. The reason I built it in LangChain and LangGraph specifically, though, was different. I've been shipping AI applications in TypeScript for a while now, mostly on Vercel's AI SDK, so this wasn't "what does a tool-calling agent even do" — it was "does the same class of problem feel different on the other side of the ecosystem." The stack is LangChain and LangGraph for the agent, Tavily for web search — no vector database, no PGVector, nothing RAG-shaped. This is a live-research problem, not a retrieval one, and I wanted to see how that felt in Python specifically. 

The plan 

1. Create the tools, pass them to one agent, and eval the output against companies in the golden set.
2. Check that with an LLM judge.
3. Use that feedback to improve the agent and re-eval.
4. Swap in a Deep Agent with sub-agents and eval the same set.
5. Check that with an LLM judge.
6. Wire up tracing for real observability — I expected sub-agents to increase accuracy at the cost of more token usage, and wanted to actually see that trade rather than guess at it.
7. Use that feedback to improve the agent, re-eval, and start reducing cost.
8. Action the cost reductions.
9. Use that feedback to improve the agent again and re-eval.


## 01. The baseline: one agent, one shared context

Phase one of this thing is a plain LangGraph tool-calling loop. One agent, four tools (`web_search`, `fetch_page`, `fetch_careers_page`, `lookup_funding`), one shared conversation. I expected it to degrade, and it did with context filling up, the model attributing a funding figure to a source that was actually about headcount, citations drift. On the golden set (20 companies, hand-labelled), it settled around **7.1 tool calls per brief**

That number matters later. Keep it in your pocket.

The fix, on paper, is well-known: give each research area its own isolated context so funding research can't bleed into hiring research. LangChain's Deep Agents package (`create_deep_agent`, on top of LangGraph) does exactly this — sub-agents, a virtual filesystem, a `task` tool for delegation. So phase four was: four sub-agents (company-profile, funding, hiring, news), each in its own context, each writing a findings file, with the parent reading only the four summaries and never the raw research.

## 02. The eval that makes every claim in this post checkable

Before touching any of the sub-agent stuff, I built a golden set: 20 companies, hand-labelled, and deliberately not 20 easy ones. ~14 are ordinary companies with discoverable facts. 3 are companies where a key fact is genuinely not public — private headcount, undisclosed funding — where the only correct answer is abstaining, and getting that subset right matters more than getting any of the easy ones right. 3 have known conflicting sources (the company site says one headcount, LinkedIn says another), where the correct behavior is surfacing the disagreement, not silently picking a side.

The metrics, on purpose, are almost all deterministic — no LLM judge required:

| metric | judge needed? |
|---|---|
| factual accuracy vs. expected facts | no |
| hallucination rate (schema violations) | no |
| abstention correctness | no |
| conflict detection | no |
| steps / cost / latency per brief | no |
| citation validity (does the source actually say the claim) | yes, one cheap call per claim |

That's not an accident, it's the actual design of the output schema — every claim carries an `evidence` list of source indices, so "did it abstain correctly" and "did it surface the conflict" become string/set membership checks, not vibes. The rules that make that hold are enforced in code, not left to the prompt to remember:

```python
kind == "verified"  -> len(evidence) >= 1
kind == "inference" -> evidence references at least one verified claim's evidence
every index in any evidence list must exist in sources
```

A brief that violates any of these gets kicked back to the agent for repair, with the specific violation named. Two failed repairs and the run fails loudly instead of shipping something that looks fine and isn't. The one metric that does need a judge (does the cited source actually support the claim) gets validated first, against 30 hand-labelled examples, before I trust a single number it produces. If the judge's own accuracy comes in under 80%, the citation-validity column gets printed with a warning attached rather than quietly presented as fact.

This is the part that made every "before/after" table in this post possible. Without it, "the fixes helped" is a feeling. With it, it's a specific citation-validity delta on a specific golden-set company, checked against a judge I'd already validated. 

## 03. Getting the Parent and Sub agents set up

**Parallel dispatch isn't automatic.** The architecture diagram in my own project brief says "four sub-agents in parallel," and I'd assumed that was just how `task` calls worked. It isn't — the docs are explicit that the main agent delegates via `task` tool calls "in a single turn to run them in parallel," which means parallelism only happens if the *model* chooses to emit four tool calls in one turn. Nothing structural forces it. I had to write that instruction into the parent's prompt by hand:

```
1. In a single turn, call the task tool four times — once each for
company-profile, funding, hiring, and news (use those exact names as
subagent_type). Do not research anything yourself directly; that's
what the sub-agents are for.
```

**Findings files don't write themselves.** My original architecture assumed a sub-agent's output would land in `findings/<area>.md` automatically. It doesn't — a sub-agent's return value is plain text back to the parent unless its own system prompt explicitly calls `write_file` before finishing. Wasnt a deal breaker but worth mentioning

```
When you are done, call write_file to save your findings to
`findings/{filename}.md`. Structure it as a markdown list of facts,
each with its source URL and fetched_at timestamp, so the parent
agent can build citations from it directly.
```

**There's a fifth sub-agent I didn't order.** `create_deep_agent` auto-adds a `general-purpose` sub-agent to the `task` tool's options unless you either name one of your own subagents `general-purpose` or explicitly disable it. I did neither. So for a good chunk of this project, the parent had five delegation options, not four, and nothing in my prompt ever told it about the fifth one or scoped what it could do. More on why this mattered in a minute.

## 05. The bill arrives

First real smoke test, two companies, after all four sub-agents were wired up and the general-purpose leak was still live:

| | Notion Labs | Retool |
|---|---|---|
| cost | $3.08 | $2.54 |
| latency | 311s | 310s |
| tool_calls | 313 | 227 |
| model_calls | 87 | 60 |

Compare that 227-313 to the 7.1 tool calls per brief from the single-agent baseline. That's a 30-45x multiplierm and on first glance kt seems four sub-agents burning tokens independently is the known cost of isolation. But a multiplier that size isn't "more thorough," it's something being wrong.

The output quality was genuinely good, for what it's worth — real conflicts surfaced when compared to the single agent (a headcount disagreement, an ambiguous funding-round label), honest abstentions, well-numbered sources. So this wasn't a case of the architecture not working. It was a case of it working and costing far more than it should to get there.

## 06. Three fixes, and what each one actually saved

Three things, in order of how much they mattered.

**No efficiency budget.** The single-agent version's system prompt ends with an explicit instruction — a handful of well-chosen tool calls beats exhaustively searching, stop once you have enough. I'd never carried that line, or anything like it, into any of the four sub-agent prompts. Nothing told a sub-agent when "enough" was. Added to each one:

```
Work efficiently: a handful of well-chosen tool calls beats
exhaustively searching. You are one of four specialists researching
this company in parallel — you do not need to verify every possible
angle of your focus area, just enough to write a useful,
honestly-hedged findings file. Once you have that, stop searching.
```

The math on the pre-fix version is ugly: 313 tool calls over 3 attempts (two repairs, maxed out every time), roughly 25 tool calls per sub-agent per attempt, against a whole-brief budget of ~7 for the entire single-agent version.

**The fifth sub-agent, still uninvited.** Confirmed via the harness's own `task` tool description — before the fix, the model literally saw `general-purpose` listed as a valid delegation target alongside my four real ones, description-free, scope-free. Turns out it only has filesystem tools by default (no web access), so it wasn't the redundant-research culprit I first suspected — but it was still an unaccounted, unprompted path the architecture wasn't supposed to have. Disabled with one call:

```python
register_harness_profile(
    f"openai:{MODEL}",
    HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)),
)
```

**Checkpointed retries instead of full re-runs.** This is the one I verified before trusting it, because it changes behavior in a way that's easy to get subtly wrong. The repair loop's malformed-date path — `StructuredOutputValidationError`, raised deep inside LangChain's own structured-output parsing — unwinds the whole `agent.invoke()` call with no result object. Without a checkpointer, my retry logic was falling back to the *original* bare enquiry message and starting over: all four sub-agents, from zero, every time a date came back formatted as `"2021-10-08 (updated 2022-04-21)"` instead of ISO 8601. And this happened on *every single attempt* in the first smoke test.

Before trusting a fix for that, I needed to know something the docs only implied: does a checkpoint survive an exception raised mid-graph, the way it survives a purpose-built `interrupt()` pause? Those are documented as the same mechanism, but I hadn't seen it confirmed for a raw exception specifically. So, another five-minute throwaway — a sub-agent whose tool appends to a real Python list, a parent told to delegate then deliberately call a tool that always raises:

```python
try:
    agent.invoke({"messages": [{"role": "user", "content": "go"}]}, config=config)
except RuntimeError:
    pass
# same thread_id, new message
agent.invoke({"messages": [{"role": "user", "content": "ok"}]}, config=config)
```

`call_log` stayed at `['called']` across both invokes. The marker sub-agent did not re-run. Confirmed, not assumed — a checkpointer plus a stable `thread_id` means a crash mid-graph resumes from the last completed superstep, not from scratch, exactly like an interrupt would.

Put those first two fixes together and re-ran Retool:

| | Before | After efficiency + no fifth sub-agent |
|---|---|---|
| cost | $2.54 | $1.20 |
| latency | 310s | 161s |
| tool_calls | 227 | 102 |
| model_calls | 60 | 30 |

Roughly halved, across the board, output quality unchanged. And that's *before* the checkpointing fix even landed — that one doesn't show up as a flat percentage so much as it removes the failure mode entirely: a repair attempt that used to cost a full second (or third) run of four sub-agents now costs only the marginal retry of the synthesis step that actually failed.

## 07. Wiring up the thing that should've come first

My original build plan had observability as phase six, after human-in-the-loop. On paper that made sense — HITL felt like the more "core" feature. In practice, by the time I hit the cost blowup in the last section, I was debugging a black box: four sub-agents, each in their own context window, and the only visibility I had into what any of them actually did was squinting at a finished brief's source list and guessing sub-agent ownership from URL patterns and timestamps. Workable, and I did find a real duplicate fetch that way (`retool.com/careers` fetched three separate times by what were clearly three different code paths, 13 seconds apart) — but indirect, and indirect is exactly the kind of thing that hides a wrong number instead of surfacing it. I swapped the two phases. You can't fix what you can't see, and at that point I couldn't see anything.

So: LangSmith, env vars only, no code changes — `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`. Except the first attempt 403'd on every single trace ingest, silently, while the agent itself ran fine. Turned out to be two separate things stacked on top of each other, and getting to the bottom of both without burning another agent run each time was its own small lesson:

- The API key was scoped to "Full Organization" instead of a specific workspace, which 403s on trace ingestion specifically (a documented LangSmith behavior — organization-scoped service keys need an explicit tenant header that the SDK doesn't send by default).
- Even after fixing that, still 403. Turned out the account was on the EU region and `LANGSMITH_ENDPOINT` was unset, defaulting to US. The maddening part: **reads worked fine against the wrong endpoint the whole time** (`Client.info()` kept succeeding), which is exactly why it took a minute to find — the failure only showed up on the write path.

## 08. What the trace actually showed

Once tracing worked, the real per-sub-agent breakdown for that Retool run:

```
sub-agent          tool_calls  llm_calls   in_tok   out_tok    cost
PARENT                      9          4    23452      4461  $0.074
company-profile             15          6    77935      3058  $0.084
funding                     14          7   108450      2728  $0.088
hiring                      17          6    98627      3279  $0.102
news                        21          7   125032      2782  $0.105
```

Roughly even across the four, which was reassuring — no single sub-agent was pathologically worse than the others, cost tracked fetch volume directly (the sub-agent that fetches the most pages pays the most per LLM call, because every new turn resends the whole growing conversation).

And then the trace turned up a bug I wouldn't have found any other way. `lookup_funding` calls `web_search` internally, which calls Tavily's own tool internally. Each of those inner calls is *itself* a traced LangChain tool, so one decision by the funding sub-agent — "call `lookup_funding` once" — was showing up as three separate tool-call entries. Across the whole trace: **100 counted, 76 real.** Every `tool_calls` number I'd reported all session had been inflated by about 24%, and it had nothing to do with the agent's actual behavior — it was an artifact of how I'd wrapped Tavily.

Fixed with a general rule rather than a hardcoded exclusion list — only count a tool call whose parent isn't itself a tool call — so it holds for any future tool that wraps another tool, not just this one:

```python
def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
    if parent_run_id not in self._tool_run_ids:
        self.tool_calls += 1
```

Verified for free, no LLM spend: invoke `lookup_funding` directly through the fixed callback, confirm it counts 1, not 3.

That fix matters beyond this one trace, because `tool_calls` isn't just a debugging number — it's one of the eval harness's own trajectory metrics, the same table that scores every golden-set run. A 24% inflation baked into a metric the eval prints as fact is exactly the kind of thing section 02 was supposed to prevent, and it slipped through anyway, because the bug lived in the instrumentation, not in anything the eval itself was checking. The eval can only be as honest as what feeds it.

## 09. Where it's left off

Back to the plan from the start of this post: steps one through six happened, more or less, if not in the tidy order this post's nine sections might suggest. Step nine — another eval pass after acting on the cost reductions, the actual 20-company delta this whole loop was building toward — is the one I stopped short of, on purpose, due to costs (this just a mock project after all, and im happy with the evals reached after ten)

That's where it sits — and it's a good place to stop for now, not just a stopping point. On the 10 companies I ran end to end after all three fixes, the isolated version hit 100% citation validity, 100% factual accuracy, and surfaced real conflicts on both — a headcount disagreement, an ambiguous funding-round label — that the single-agent version had run on the *same companies* and missed entirely. It's costing roughly half what it did a session ago. And every bug in this post — the uninvited fifth sub-agent, the missing efficiency budget, a tool getting counted three times for one decision — got caught and fixed before it shipped anywhere, because the eval and the trace were both actually there to catch it. 

The TLDR - build, judge, eval, repeat. Otherwise we are just vibecoding ;)
