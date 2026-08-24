# SDR Account Brief Agent — project brief

Context for an AI coding assistant (Claude Code). Read fully before writing
code. Save this as `CLAUDE.md` in the repo root.

> **How to use this document (note to the developer, not the assistant)**
>
> Don't hand this whole document over and say "build it." Keep it as
> `CLAUDE.md` so it loads as standing context, then start each session with
> a single phase:
>
> `Read CLAUDE.md, build Phase 1, stop when tests pass.`
>
> An assistant given eight phases will attempt all eight and produce
> something too large to review. One phase per session, commit at each
> boundary, and write the README notes for that phase before starting the
> next — by the end you won't remember what Phase 2 looked like when it
> broke, and that detail is what the project's argument rests on.

---

## 1. What this is

An agent that turns an inbound sales enquiry into a **sourced account brief**
for an SDR: what the company is, what they're likely trying to do, what we
can't verify, and what to ask on the call.

Built with **LangChain Deep Agents** on the **LangGraph runtime**, in
**Python**. It is a learning project with a portfolio outcome, built by a
senior full-stack engineer (TypeScript/Node/React/Postgres background)
moving into AI engineering. Python is being learned deliberately as part of
this — explain idiomatic choices as you go rather than silently writing
expert-level code.

### The thesis of the project

Anyone can build a research agent. **The engineering is in knowing which
parts of the output you are allowed to assert.** Per-field confidence,
mandatory citations, explicit abstention, and conflict surfacing are the
point of this project. The research is the boring half.

If a design decision trades away verifiability for richer output, choose
verifiability.

### How to work with the developer

- **Explain trade-offs before implementing**, especially Python idioms and
  Deep Agents / LangGraph abstractions. State the options, then implement.
- **Small, working increments.** Each phase runs end to end before the next.
- **Verify APIs against current docs.** Deep Agents and LangGraph move fast
  and pre-1.0 tutorials are wrong. Do not trust memory for API shapes —
  check `docs.langchain.com` and pin versions.
- **Never skip Phase 3 (evaluation).** See §7.
- **Build only the phase you were asked for.** Stop at its "done when" line
  and hand back, even if the next phase looks obvious. Do not run ahead.
- Do not add features that aren't in this brief without saying so first.

---

## 2. Hard constraints

These are not negotiable and shape the whole design.

1. **Research companies, not people.** Never research, profile, or store
   information about the named individual in the enquiry. The contact's name
   and email are passed through to the brief untouched and are never used as
   a search term. This is a privacy and compliance boundary, not a
   preference.
2. **No scraping of sites that forbid it.** Use public pages, official APIs,
   and web search. Respect `robots.txt`. If a source can't be accessed
   legitimately, the field is reported as unverified.
3. **Every factual claim carries at least one source index.** A claim with no
   evidence fails schema validation and never reaches the output.
4. **Abstention is a first-class output.** "We could not verify this" is a
   successful result, not a failure. Never fill a gap with a plausible guess.
5. **Judgment fields must cite the verified claims they rest on.** An
   inference with no underlying evidence is rejected.

---

## 3. Input and output

### Input

```python
class Enquiry(BaseModel):
    contact_name: str        # passed through, never researched
    contact_email: str       # domain is used; the address is not researched
    company_name: str
    message: str             # free text from the form
```

### Output

```python
Confidence = Literal["HIGH", "MEDIUM", "LOW"]

class Source(BaseModel):
    index: int
    url: str
    fetched_at: datetime
    published_at: datetime | None = None

class Claim(BaseModel):
    value: str
    confidence: Confidence
    evidence: list[int]                    # indices into Brief.sources
    kind: Literal["verified", "inference"]

class Conflict(BaseModel):
    field: str
    positions: list[str]                   # what each source claims
    evidence: list[int]
    resolution: str                        # which was used and why

class Brief(BaseModel):
    company: Claim
    funding: Claim | None
    hiring_signals: list[Claim]
    likely_use_case: Claim                 # inference
    deal_band: Claim | None                # inference, LOW by default
    unverified: list[str]                  # explicit abstentions
    conflicts: list[Conflict]
    suggested_opening: str
    sources: list[Source]
    overall_confidence: Confidence
    generated_at: datetime
```

### Validation rules (enforced in code, not by prompting)

- `kind == "verified"` → `len(evidence) >= 1`
- `kind == "inference"` → must reference at least one verified claim's
  evidence indices
- every index in any `evidence` list exists in `sources`
- `deal_band`, if present, is always emitted with a do-not-quote warning
- a brief failing validation is returned to the agent for repair, with the
  specific violation named; after 2 failed repairs, fail the run loudly

The rendered markdown brief is generated from this object, never written
freehand by the model. See `docs/example_brief.md` for the target output.

---

## 4. Architecture

Deep Agents supplies the harness: planning, sub-agents with isolated context,
a virtual filesystem, and human-in-the-loop interrupts. LangGraph supplies
durable execution and checkpointing underneath.

```
Enquiry
  ↓
Parent agent — plans research areas, writes todos
  ↓
Four sub-agents in ISOLATED context windows, in parallel:
  ├── company-profile   → findings/company.md
  ├── funding           → findings/funding.md
  ├── hiring            → findings/hiring.md
  └── news              → findings/news.md
  ↓
Parent reads only the four summary files (never the raw research)
  ↓
Synthesis → Brief object
  ↓
Schema validation (repair loop on failure)
  ↓
HITL gate if any field is LOW confidence
  ↓
Rendered markdown brief
```

**Why sub-agents, specifically:** without context isolation, news research
bleeds into funding analysis and claims get attributed to the wrong sources.
Isolation is what makes the citation guarantee hold. This is the reason the
harness is justified here — say so in the README.

**Why the parent reads only summaries:** context offloading. The parent never
holds all raw research, which keeps long runs viable.

---

## 5. Tools

Build these as plain functions with careful descriptions. **Tool descriptions
are prompts** — expect to rewrite them repeatedly, and treat that as the main
tuning surface.

| Tool | Purpose | Notes |
| --- | --- | --- |
| `web_search(query)` | General search | Must return URLs, not just snippets |
| `fetch_page(url)` | Retrieve and extract text | Record `fetched_at`; respect robots.txt |
| `fetch_careers_page(domain)` | Open roles | Highest-signal tool — see below |
| `lookup_funding(company)` | Funding events | Public sources only |
| `write_findings(area, content)` | Sub-agent output | Deep Agents filesystem |
| `read_findings(area)` | Parent synthesis | |

**Job postings are the strongest available signal** for what a company is
building and investing in. A company hiring two backend engineers for
"document processing pipelines" is telling you their roadmap. Weight this
tool accordingly in the prompts.

Every tool that produces a fact must return the URL and fetch timestamp
alongside the content, so the citation chain never has to be reconstructed
later.

---

## 6. Build phases

### Phase 1 — skeleton (no agent yet)

Pydantic models, the validation layer, and the markdown renderer. Write a
`Brief` object by hand in a test and render it. Confirm validation rejects:
a verified claim with no evidence, an evidence index that doesn't exist, and
an inference with no underlying verified claim.

**Done when:** `pytest` passes and `python -m sdr.render fixtures/brief.json`
prints something matching `docs/example_brief.md`.

Doing this first means the agent is later built against a contract that
already exists, rather than the contract being retrofitted around whatever
the model happens to emit.

### Phase 2 — single-agent version

One agent, the tools above, no sub-agents, no Deep Agents harness. Plain
LangGraph tool-calling loop. Run it on 3 companies.

**Expect it to degrade.** Context fills, citations drift, claims get
attributed to the wrong source. **Record this** — screenshots, traces, notes.
The failure is the justification for Phase 4 and belongs in the README.

### Phase 3 — evaluation harness (before any improvement)

See §7. Do not proceed to Phase 4 until this prints numbers.

### Phase 4 — Deep Agents harness

Add planning, the four sub-agents with isolated contexts, and the filesystem
findings pattern. Re-run the eval. The delta versus Phase 2 is the headline
result of the project.

### Phase 5 — observability

Langfuse (self-hostable, open source, appears by name in job postings) or
LangSmith. Trace every run: tool calls, sub-agent spans, token cost, wall
time. Add cost-per-brief and steps-per-brief to the eval output.

### Phase 6 — human in the loop

`interrupt()` before any brief with a LOW-confidence field is released.
Resume with approval, edit, or reject. Persist with a Postgres checkpointer so
a paused run survives a restart.

### Phase 7 — API and brief viewer

FastAPI endpoints from §8, and the intake + brief view screens. Read-only
first: submit an enquiry, wait, see the rendered brief with working
citations. No streaming yet.

**Done when:** a brief generated through the UI is indistinguishable from
`docs/example_brief.md`, and every citation marker opens its source.

### Phase 8 — live run view and review queue

SSE streaming, the four sub-agent lanes, and the human-in-the-loop queue
wired to the Phase 6 interrupts. Verify resumability explicitly: start a run,
close the tab, reopen the thread URL, approve from there.

**Done when:** a paused run survives a server restart and can be approved
from a fresh browser session.

---

## 7. Evaluation

The point of this project. Build it in Phase 3, before improving anything.

### Golden set

20 companies, hand-labelled. Composition matters:

- ~14 ordinary companies with reasonably discoverable facts
- **3 companies where a key fact is genuinely not public** (private
  headcount, undisclosed funding). The agent must abstain. This is the most
  important subset in the whole test set.
- **3 companies with known conflicting sources** (site says one headcount,
  LinkedIn another). The agent must surface the conflict rather than silently
  picking one.

```jsonc
{
  "company": "...",
  "domain": "...",
  "enquiry_message": "...",
  "expected_facts": {
    "hq_country": "NL",
    "funding_round": "Series B",
    "has_open_engineering_roles": true
  },
  "must_abstain": ["driver_count"],
  "known_conflicts": ["headcount"]
}
```

### Metrics

| Metric | Method | LLM judge? |
| --- | --- | --- |
| Citation validity | Fetch each source, check the claim is supported | Cheap judge, one call per claim |
| Factual accuracy | Compare against `expected_facts` | No |
| Hallucination rate | Claims failing schema validation | No — deterministic |
| Abstention correctness | Did it abstain on `must_abstain` fields? | No |
| Conflict detection | Did it surface `known_conflicts`? | No |
| Steps / cost / latency per brief | Trace logs | No |

Most of these are deterministic. That is a design achievement, not a
coincidence — the output schema was built to make it so. Lead with this in
the README.

### Judge discipline (for citation validity only)

Binary verdicts, one criterion per call, reasoning before verdict,
temperature 0. Validate the judge against 30 hand-labelled examples before
trusting its numbers. Keep those labels as a regression test for the judge
itself.

### Trajectory metrics

Beyond output quality: number of steps, sub-agents spawned, tools called,
redundant searches, cost. A correct brief produced in 60 steps is a worse
result than the same brief in 15.

---

## 8. Frontend

The brief is only useful if a human can act on it quickly and see what to
distrust. That makes the UI part of the engineering argument, not decoration.

The developer's strongest area is React/Next.js/TypeScript — build this part
properly rather than as a throwaway demo shell.

### Stack

Next.js (App Router), TypeScript, Tailwind, TanStack Query. **SSE, not
WebSockets**, for run streaming — traffic is one-directional, it survives
proxies, and it's less to operate.

### Design principles

- **The UI's job is to make uncertainty legible and approval cheap.** If a
  reviewer has to hunt for what's uncertain, the interface has failed.
- **Never render an unverified claim at the same visual weight as a verified
  one.** Confidence is the primary visual axis of the brief view.
- **A citation must be one click from the claim it supports.** A citation
  nobody can check is decoration.
- **Streaming is not optional.** A 90-second run with no feedback reads as
  broken. Show sub-agent progress as it happens.
- **This is not a chat interface.** It's a workflow tool. Resist the pull
  toward a message thread.

### Screens

**1. Intake** — the enquiry form from §3. Trivial; don't over-invest.

**2. Run view (live)** — the interesting one. Four parallel lanes, one per
sub-agent, each showing tool calls as they happen. The planning tool's todo
list rendered as a checklist that updates. A running token/cost counter.
Thread ID in the URL.

This is the "agent action visibility" pattern several job postings ask for by
name, and almost nobody building agent demos bothers with it.

**3. Brief view** — renders the validated `Brief` object. Confidence shown
per field, not as one overall badge. Citation markers open a source panel
with the URL, `fetched_at`, and the supporting excerpt. The `unverified`
section is rendered **prominently near the top**, not buried at the bottom —
what the system doesn't know is the most decision-relevant part. Conflicts
shown inline at the field they affect. `deal_band`, if present, carries its
do-not-quote warning visually, not just in text.

**4. Review queue** — the human-in-the-loop gate. Briefs paused awaiting
approval, with the triggering LOW-confidence fields surfaced first. Actions:
approve, reject with reason, or re-run a single research area. One-key
approval; a reviewer processing twenty of these should never need the mouse.

**5. Eval dashboard (optional)** — golden-set metrics over time, per commit.
Cheap to add once §7 emits JSON, and it makes the project's central argument
visible at a glance.

### API surface

```
POST /enquiries                     -> { thread_id }
GET  /runs/{thread_id}/events       -> SSE stream
GET  /briefs/{thread_id}            -> Brief
POST /briefs/{thread_id}/approve
POST /briefs/{thread_id}/reject     -> { reason }
POST /briefs/{thread_id}/rerun-area -> { area }
GET  /review-queue                  -> paused runs
```

SSE event types: `plan_updated`, `subagent_started`, `tool_called`,
`subagent_finished`, `validation_failed`, `awaiting_approval`, `complete`,
`error`. Type these once and share the definitions between FastAPI and the
frontend.

### Resumability

Thread ID in the URL plus the Postgres checkpointer means a reviewer can
close the tab mid-run, return an hour later, and find the run exactly where
it paused. Build this deliberately — it's the visible payoff of durable
execution and the single best thing to demo.

---

## 9. Repo layout

```
.
├── src/sdr/
│   ├── models.py          # Pydantic schemas (§3)
│   ├── validate.py        # validation rules + repair loop
│   ├── render.py          # Brief -> markdown
│   ├── tools/             # one module per tool
│   ├── agent_simple.py    # Phase 2: plain LangGraph loop
│   ├── agent_deep.py      # Phase 4: Deep Agents harness
│   ├── api.py             # FastAPI: endpoints + SSE stream
│   └── cli.py
├── web/                   # Next.js app
│   ├── app/
│   │   ├── page.tsx           # intake
│   │   ├── runs/[id]/         # live run view
│   │   ├── briefs/[id]/       # brief view
│   │   └── review/            # HITL queue
│   ├── components/
│   └── lib/events.ts      # SSE event types, mirrored from api.py
├── eval/
│   ├── golden.json
│   └── run.py             # prints the §7 metrics table
├── docs/
│   └── example_brief.md   # the target output format
├── tests/
└── README.md              # ← the deliverable
```

`validate.py` deserves thorough unit tests. It's pure, it's the guarantee the
whole project rests on, and tests make schema changes safe.

---

## 10. Gotchas

1. **Sub-agent cost multiplies.** Four sub-agents per brief, each burning
   tokens. Cap concurrency and steps, and log cost per run from day one.
2. **Deep Agents / LangGraph APIs churn.** Pin versions. Check current docs
   rather than reproducing tutorial code. `AgentExecutor` is deprecated —
   don't use it.
3. **Don't reach for `create_deep_agent` in Phase 2.** The contrast between
   the plain loop and the harness is the project's main finding.
4. **Fetched web content is untrusted input.** A careers page could contain
   text that reads as an instruction. Keep tool output structurally separated
   from instructions in the prompt.
5. **Timestamp everything.** Facts go stale. A brief citing a page fetched
   three months ago is a different claim than one fetched today.
6. **Watch over-abstention too.** An agent that abstains on everything scores
   perfectly on hallucination and is useless. Track both directions.
7. **Never let the model write the final markdown freehand.** It renders from
   the validated object, or the citation guarantee is worthless. This applies
   equally to the frontend: the brief view renders the `Brief` object, it
   does not display model-authored prose.
8. **A polished UI makes weak output look strong.** That's the danger of
   building this well. Keep confidence and abstention visually loud enough
   that a good-looking brief with three unverified fields still reads as
   uncertain.
9. **SSE connections drop.** Reconnect using the thread ID and replay state
   from the checkpointer rather than assuming an unbroken stream.

---

## 11. First actions

1. Confirm the stack and Python tooling choices briefly (`uv` vs `poetry`,
   test runner, formatter).
2. Scaffold the repo and write `docs/example_brief.md` as the target output —
   before any code, so the contract is visible.
3. Build Phase 1: models, validation, renderer, tests.
4. Stop. Have the developer review the schema and the rendered example before
   any agent code is written.
