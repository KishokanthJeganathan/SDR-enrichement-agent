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

6-company pilot golden set (`eval/golden.json`), not yet the full 20 CLAUDE.md
§7 specifies — see "Scope" below. Facts in the golden set were independently
researched via live web search, not copied from the Phase 2 agent's own
output — using the agent's prior answers as ground truth would make the eval
circular. Companies: 4 ordinary (Notion Labs, Airtable, Linear, Vercel) +
1 must-abstain (Retool — revenue/headcount it doesn't disclose) + 1
known-conflict (Linear again — a real, independently-verified customer-count
discrepancy between two of its own live pages).

**Citation-validity judge** (`eval/judge.py`): `gpt-4o-mini`, temperature 0,
binary verdict with reasoning first. Deliberately *not* a `gpt-5.6-*` model —
that whole family forces `temperature=1` with no override, which would
violate the temperature-0 requirement outright; `gpt-4o-mini` is a classic
model that still honors it. Validated against 10 hand-labeled examples
(`eval/judge_labels.json`) before trusting its numbers on the real run: **90%
accuracy (9/10)**. The one miss is informative, not just noise — it
conflated "the $120M figure is numerically correct" with "the company
disclosed it" (it's actually a third-party analyst estimate), missing the
verified-vs-estimated distinction that's the whole point of this project.
Worth knowing as a limit on the judge, not something to prompt-tune away for
one label.

**Results — `uv run python eval/run.py`, 6/6 runs completed:**

| Metric | Score |
| --- | --- |
| Citation validity | 83% |
| Factual accuracy | 58% |
| Avg. initial violations (pre-repair) | 0.50/brief |
| Fraction of briefs needing ≥1 repair | 33% |
| Abstention correctness | 83% |
| Conflict detection | 100% |
| Avg. cost/brief | $0.092 |
| Avg. latency/brief | 34.6s |
| Avg. tool calls/brief | 8.7 |

Full per-claim detail and raw `Brief` objects are persisted to
`eval/results/<company>.json` for each run — the summary numbers above are
traceable back to exactly which claim failed and why, not just an aggregate.

**What the misses actually are** (this is the part worth reading past the
table):

1. **Citation validity is weakest on hiring_signals, not funding/company
   claims** (which hit 100% every run). Every hiring_signals miss was the
   judge saying the re-fetched careers page didn't mention the specific role
   titles the claim named. `fetch_page` is plain `requests` + BeautifulSoup
   with no JS execution — if a careers page renders its listings
   client-side, the agent may be asserting specific titles beyond what its
   own fetched text actually contained. The harness re-fetches live at judge
   time rather than replaying the original tool output, so this can't yet
   fully separate "the agent overreached" from "the page changed between the
   agent's fetch and the judge's" — a known gap, worth fixing before scaling
   to 20 companies by persisting the original tool outputs too.

2. **Most "factual accuracy" misses are honest hedging against a stale or
   narrower source, not fabrication.** Ramp's funding claim explicitly named
   its source and date — "Contrary Research reported... as of its October
   16, 2025 update" — and was right about what that source said. It just
   never found the actual June 2026 $750M Series F my golden label expected,
   because the system prompt tells it to favor a handful of tool calls over
   exhaustive search (a cost-control decision, §5). Airtable's brief didn't
   find the August 2026 Bending Spoons acquisition (3 weeks old at eval
   time) — but it explicitly listed "acquisitions after the December 2021
   Series F" under `unverified` rather than asserting the stale info as
   current. That's the hard constraints working correctly; the metric just
   can't currently tell "wrong" apart from "well-hedged but shallow." Vercel
   returned `funding: null` outright rather than guess — also scored as a
   miss by the same blunt metric, also correct agent behavior. Refining
   `factual_accuracy` to separate active misstatement from calibrated
   non-coverage is the clearest next improvement to the harness itself.

3. **Cross-run inconsistency, found by accident and worth taking
   seriously.** Notion's funding claim in this Phase 3 run: *"Contrary
   Research reports that Notion has raised $343.2M and is at Series C
   stage"* (evidence: 1 source, no conflict raised). Notion's funding claim
   in the Phase 2 spot-check two sections up: `$275M Series D`, with
   `$343.2M` vs `$340.2M` explicitly raised and resolved as a **Conflict**
   between two sources. Same company, same agent, same system prompt — two
   runs, two different verified claims, because which sources a given run's
   handful of tool calls happens to surface isn't deterministic. This is
   exactly the citation-drift Phase 2 predicted, just visible only *across*
   repeated runs rather than within a single long one — the scale at which
   this pilot could actually observe it. It's the strongest single piece of
   evidence in this pilot for why Phase 4's isolated, more thorough
   sub-agents need to exist.

**Scope note:** this is 6 companies, not CLAUDE.md's specified 20. The
mechanism is proven and the numbers above are real, not placeholders — but a
pilot this size shouldn't be read as a final verdict, especially for
abstention/conflict correctness where the pilot has only one example each.
Scaling the golden set (and the judge's label set proportionally) is the
natural next increment before leaning on these numbers for the Phase 4
comparison.
