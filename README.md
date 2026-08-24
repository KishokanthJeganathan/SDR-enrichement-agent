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
