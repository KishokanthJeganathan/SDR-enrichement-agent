# SDR Account Brief Agent

Turns an inbound sales enquiry into a sourced account brief for an SDR. Full
project brief and hard constraints: [CLAUDE.md](./CLAUDE.md). Target output
format: [docs/example_brief.md](./docs/example_brief.md).

The engineering bet this project is testing: per-field confidence, mandatory
citations, and explicit abstention are enforceable in code even when the
research that feeds them isn't. Phase 1 built that enforcement layer before
any model ever ran against it.

## Setup

```bash
uv sync
cp .env.example .env  # fill in OPENAI_API_KEY and TAVILY_API_KEY
```

## Usage

```bash
# Render a Brief object to markdown (no LLM involved)
uv run python -m sdr.render fixtures/brief.json

# Run the Phase 2 research agent end to end (calls OpenAI + Tavily)
uv run python -m sdr.cli --company "Acme Inc" --email "jane@acme.com" \
  --name "Jane Doe" --message "Looking for a research tool for our SDR team."
```

## Phase log

### Phase 1 — schema, validation, renderer

Pure, deterministic Python — no LLM. `models.py` defines the `Brief`/`Claim`/
`Source`/`Conflict` schema from CLAUDE.md §3; `validate.py` enforces the
citation rules Pydantic can't express on its own (every evidence index
resolves to a real source; a verified claim has evidence; an inference
shares evidence with a verified claim); `render.py` renders a validated
`Brief` to the markdown format in `docs/example_brief.md` — the model never
writes that markdown itself.

`pytest` covers the three required rejection cases (empty evidence on a
verified claim, a dangling evidence index, an inference with no verified
backing) plus a byte-exact render check against the fixture.

### Phase 2 — single-agent version

One `langchain.agents.create_agent` tool-calling loop (the current,
non-deprecated replacement for `create_react_agent`), no sub-agents, no Deep
Agents harness. Model: `gpt-5.6-terra` via `langchain-openai`, chosen over
Claude for cost during dev-iteration-heavy phases. Search: Tavily.

Tools (`src/sdr/tools/`): `web_search`, `fetch_page` (enforces robots.txt in
code via `urllib.robotparser`, not just by instruction), `fetch_careers_page`
(tries common URL paths, falls back to search), `lookup_funding` (a
funding-focused `web_search` wrapper — there's no free structured funding
database, and scraping Crunchbase/PitchBook would violate the
no-forbidden-scraping constraint, so this is honestly a search wrapper, not
a real API integration).

`response_format=Brief` gets structural validation for free via Claude/GPT's
native structured output — but not the citation cross-referencing rules, so
`validate_brief` still runs on every result, with up to 2 repair turns
(re-prompting with the specific violations) before the run fails loudly.

**Bugs hit while wiring this up**, both fixed in code:
- `gpt-5.6-terra` (a reasoning-tier model) rejects function/tool calling on
  the Chat Completions endpoint unless reasoning is disabled — fixed with
  `ChatOpenAI(..., use_responses_api=True)` to route through the Responses
  API instead of crippling the model.
- Tavily's LangChain tool occasionally returned an empty string instead of
  JSON on a transient failure, which crashed the whole agent run on
  `json.loads("")`. `web_search` now catches that and degrades to an
  `{"error": ...}` result instead of taking down the graph.

**Run log — 3 companies (Notion Labs, Airtable, Linear), single agent, single
turn each:**

All three ran clean: no repair-loop retries, no validation failures, correct
citations on manual read, and reasonable abstention (e.g. Airtable's
`hiring_signals` came back empty rather than padded, because the careers
page didn't return usable listings that run). Linear's brief flagged a real
customer-count discrepancy between two of its own pages and reasoned about
it sensibly (different dates, likely growth rather than contradiction)
instead of silently picking one number.

**This is a smaller effect than CLAUDE.md's Phase 2 section predicts** — the
brief expects citation drift and misattribution as context fills. Worth
being honest about why that didn't show up here rather than writing up a
failure that didn't happen: these are three well-known, heavily-documented
public companies, each needing only a handful of tool calls in one short
run. n=3 anecdotal spot checks aren't a measurement, and this doesn't yet
test the two conditions most likely to actually break a single shared
context — much larger research surface within one run, and companies where
verifiable facts are genuinely sparse (the golden set's abstention and
known-conflict subsets in CLAUDE.md §7 exist specifically to probe that).
Phase 3's eval harness is what turns "seemed fine on 3 runs" into an actual
number instead of an impression.

**Update after Phase 3:** the degradation did show up — just across repeated
runs of the same company, not within one run. See "Cross-run inconsistency"
below. Re-running Notion Labs produced a materially different funding claim
than the Phase 2 spot-check above.

### Phase 3 — evaluation harness

Full 20-company golden set (`eval/golden.json`), matching CLAUDE.md §7's
exact composition: 14 ordinary + 3 must-abstain (Retool, Postman, Discord —
each has revenue/headcount that's genuinely undisclosed, only third-party
estimates exist) + 3 known-conflict (Linear's customer count, Miro's user
count, Webflow's total-funding figure — each a real, independently-verified
discrepancy between two live sources, not manufactured). Every fact was
researched independently via live web search, never copied from the agent's
own prior output — using the agent's answers as its own ground truth would
make the eval circular. Scaled up from an initial 6-company pilot once the
harness mechanism was proven working.

**Citation-validity judge** (`eval/judge.py`): `gpt-4o-mini`, temperature 0,
binary verdict with reasoning first. Deliberately *not* a `gpt-5.6-*` model —
that whole family forces `temperature=1` with no override, which would
violate the temperature-0 requirement outright; `gpt-4o-mini` is a classic
model that still honors it. Validated against the full 30 hand-labeled
examples CLAUDE.md §7 specifies (`eval/judge_labels.json`) before trusting
its numbers: **97% accuracy (29/30)**. The one consistent miss is
informative, not noise — it conflates "the $120M figure is numerically
correct" with "the company disclosed it" (it's a third-party analyst
estimate), missing the verified-vs-estimated distinction that's the whole
point of this project. Worth knowing as a real limit on the judge, not
something to prompt-tune away for one label.

**Two real bugs found and fixed while scaling to 20 companies** — both
significant enough that earlier numbers from smaller runs shouldn't be
trusted, which is why this section only reports the final, corrected run:

1. **False robots.txt blocks.** `RobotFileParser.read()` fetches
   `robots.txt` itself via bare `urllib`, with no `User-Agent` header. Sites
   like CNBC 403 that generic request as an anti-bot measure, and Python's
   stdlib treats a 403 on robots.txt as "disallow everything" — even though
   the site's actual policy allows normal crawling. Postman and Discord
   briefs came back with 0% citation validity and "no fetchable evidence" on
   every claim before this was caught; re-fetching the same URLs manually
   with a proper `User-Agent` worked fine. Fixed in `fetch.py` by fetching
   `robots.txt` the same way the real page fetch does — `requests`, same
   header — instead of letting `RobotFileParser` fetch it blind.
2. **A crash with no repair path.** The model occasionally wrote a partial
   date (`"2026-05"`, no day) for a `Source.published_at` field, which
   Pydantic's strict datetime parser rejects. That error is raised *inside*
   LangChain's structured-output parsing, before `validate_brief` or the
   existing repair loop ever sees a `Brief` object — so it wasn't a
   validation failure the harness could recover from, it was an unhandled
   exception that killed the whole 20-company run partway through. Fixed by
   catching `StructuredOutputValidationError` in `agent_simple.run()` and
   feeding it back through the same repair-message mechanism, with explicit
   instructions to use a full ISO datetime or omit the field.

**Results — `uv run python eval/run.py`, 20/20 runs completed, 0 repairs
needed:**

| Metric | Score |
| --- | --- |
| Citation validity | 75% |
| Factual accuracy | 52% |
| Avg. initial violations (pre-repair) | 0.00/brief |
| Fraction of briefs needing ≥1 repair | 0% |
| Abstention correctness | 95% |
| Conflict detection | 85% |
| Avg. cost/brief | $0.067 |
| Avg. latency/brief | 30.7s |
| Avg. tool calls/brief | 7.1 |

Full per-claim detail and raw `Brief` objects are persisted to
`eval/results/<company>.json` for every run — every number above is
traceable back to exactly which claim failed and why.

**What the misses actually are:**

1. **Zero repairs needed, across all 20 companies.** `validate_brief`'s
   citation-structure rules (every evidence index resolves, verified claims
   have evidence, inferences trace to verified evidence) held on the first
   attempt every single time. The schema-level guarantee is solid; the gaps
   that remain are in what the agent chooses to research and how
   thoroughly, not in whether its citations are internally consistent.

2. **Abstention (95%) and conflict detection (85%) — the two metrics this
   project's thesis actually rests on — are the strongest numbers here.**
   Retool, Postman, and Discord all correctly declined to assert undisclosed
   revenue/headcount as fact. Two of three known-conflict cases (Linear,
   Miro) were missed this run — the agent found the more recent number in
   each case (40,000+ customers, 100M users) but didn't happen to
   cross-check against the older, differently-dated figure that would have
   surfaced the discrepancy. Same underlying cause as finding #3 below.

3. **Most "factual accuracy" misses are honest reporting of a source that's
   real but older, not fabrication.** monday.com's funding claim is
   explicitly labeled *"Historical funding signal: TechCrunch reported... a
   $150M Series D... in July 2019"* — correctly attributed, correctly dated,
   MEDIUM confidence — it just never found that monday.com has been public
   on NASDAQ since 2021. Ramp similarly cited a real $115M Series B from
   2021 rather than the $750M Series F from two months before this eval ran.
   Vercel returned `funding: null` outright — an honest abstention, still
   scored as a miss by this blunt metric. The hard constraints (attribute
   sources, hedge confidence, don't guess) are visibly working; what's
   missing is *research depth* — the agent settles for the first credible
   funding source it finds rather than checking whether a more recent one
   exists.

4. **That research-depth gap traces to a concrete, checkable cause: tool
   calls per brief dropped from 9.1 to 7.1 after the robots.txt fix.**
   Fixing false-positive blocks meant more of the agent's *first* fetch
   attempts now succeed — which sounds like a pure win, but the system
   prompt (§5: "a handful of well-chosen tool calls beats exhaustively
   searching," written to control cost) means the agent now stops sooner,
   satisfied with the first funding source that actually loads, rather than
   being forced by fetch failures into trying alternate sources — which had
   the accidental side effect of occasionally surfacing more current
   information. That's a real tension between the cost-control instruction
   this project's own system prompt gives the agent and the freshness this
   eval is measuring — not a bug, but a design trade-off worth naming
   explicitly rather than discovering by accident again in Phase 4.

Point 4 is the clearest, most concrete hypothesis this pilot produced for
what Phase 4's isolated sub-agents should actually fix: not citation
structure (already solid at 0% repair rate) but giving each research area
(funding, hiring, news) its own tool-call budget instead of one shared,
cost-constrained budget across all of them — which is exactly what "isolated
context per sub-agent" (CLAUDE.md §4) provides. The Phase 2 vs. Phase 4
comparison this harness enables should watch tool-calls-per-area and
source-recency specifically, not just the six top-line metrics.
