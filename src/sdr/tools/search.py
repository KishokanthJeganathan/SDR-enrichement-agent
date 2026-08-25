import json
from datetime import UTC, datetime

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()  # TavilySearch reads TAVILY_API_KEY at construction, below

_tavily = TavilySearch(max_results=5)


@tool
def web_search(query: str) -> str:
    """Search the web for a query about a company. Returns up to 5 results as
    a JSON list, each with 'url', 'title', 'content' (a snippet), and
    'fetched_at' (an ISO timestamp for when this search ran).

    This returns snippets only, not full page text — use fetch_page on a
    promising URL to read the whole page before citing it as evidence.

    Never search for a person's name or email address. Only search for facts
    about the company itself. Returns {"error": ...} if the search failed —
    try a different query rather than treating that as no results found.
    """
    fetched_at = datetime.now(UTC).isoformat()
    try:
        raw = _tavily.invoke({"query": query})
        data = json.loads(raw) if isinstance(raw, str) else raw
        raw_results = data.get("results", []) if isinstance(data, dict) else []
    except Exception as exc:  # Tavily's own tool wrapper can return "" on
        # a transient failure instead of raising; a bad tool call must not
        # take down the whole agent run, so this degrades to an error result.
        return json.dumps({"error": f"search failed: {exc}", "fetched_at": fetched_at})

    results = [
        {
            "url": result.get("url"),
            "title": result.get("title"),
            "content": result.get("content"),
            "fetched_at": fetched_at,
        }
        for result in raw_results
    ]
    return json.dumps(results)
