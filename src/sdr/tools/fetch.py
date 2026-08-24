"""Page fetching, with robots.txt enforcement. See CLAUDE.md §2, §5.

robots.txt is checked here, in code — not left to the model to remember to
respect. If a page is disallowed, the tool reports that as a fetch error;
it's then on the agent (per its system prompt) to treat the underlying fact
as unverified rather than looking for another way to the same content.
"""

import json
from datetime import UTC, datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

_USER_AGENT = "SDRAccountBriefAgent/0.1 (+research bot)"
_TIMEOUT_SECONDS = 10
_MAX_CHARS = 8000


def _robots_allow(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
    except OSError:
        return True  # no reachable robots.txt: no restrictions declared
    return parser.can_fetch(_USER_AGENT, url)


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    return "\n".join(lines)


def _fetch(url: str) -> dict:
    fetched_at = datetime.now(UTC).isoformat()
    if not _robots_allow(url):
        return {"url": url, "fetched_at": fetched_at, "error": "disallowed by robots.txt"}
    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"url": url, "fetched_at": fetched_at, "error": str(exc)}
    return {"url": url, "fetched_at": fetched_at, "text": _extract_text(response.text)[:_MAX_CHARS]}


@tool
def fetch_page(url: str) -> str:
    """Fetch a specific page and return its visible text. Returns JSON with
    'url', 'fetched_at', and either 'text' (extracted page text, truncated to
    ~8000 characters) or 'error' if the page couldn't be fetched — including
    when robots.txt disallows it. On 'error', treat the underlying fact as
    unverified rather than trying another route to the same content.
    """
    return json.dumps(_fetch(url))


_CAREERS_PATHS = ["/careers", "/jobs", "/about/careers", "/company/careers"]


@tool
def fetch_careers_page(domain: str) -> str:
    """Fetch a company's open-roles/careers page by trying common URL paths
    on the given domain (e.g. 'acme.com' -> 'acme.com/careers'). Returns JSON
    with 'url', 'fetched_at', and either 'text' or 'error'.

    Job postings are the strongest available signal for what a company is
    building and investing in — weight this tool heavily when reasoning about
    likely use case. If no common path resolves, use web_search for
    '<domain> careers' and fetch the top result with fetch_page instead of
    giving up.
    """
    domain = domain.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    for path in _CAREERS_PATHS:
        result = _fetch(f"https://{domain}{path}")
        if "text" in result:
            return json.dumps(result)
    return json.dumps(
        {
            "url": None,
            "fetched_at": datetime.now(UTC).isoformat(),
            "error": "no common careers path resolved; try web_search",
        }
    )
