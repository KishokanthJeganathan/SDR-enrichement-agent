

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
        # RobotFileParser.read() fetches robots.txt itself via bare urllib,
        # with no User-Agent header — several sites (e.g. CNBC) 403 that
        # generic request as an anti-bot measure, and the stdlib treats a
        # 403 on robots.txt as "disallow everything". Fetch it the same way
        # we fetch the real page instead, so a missing UA never masquerades
        # as an actual robots.txt policy.
        headers = {"User-Agent": _USER_AGENT}
        response = requests.get(robots_url, headers=headers, timeout=_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return True  # no robots.txt: no restrictions declared
        response.raise_for_status()
        parser.parse(response.text.splitlines())
    except requests.RequestException:
        return True  # robots.txt unreachable for any reason: no restrictions declared
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


def fetch_text(url: str) -> str | None:
    """Fetch a URL and return its extracted text, or None on failure. Public,
    non-tool helper — used by the eval harness (Phase 3) to re-fetch a cited
    source and check whether it actually supports the claim, without
    reaching into this module's private _fetch().
    """
    result = _fetch(url)
    return result.get("text")


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
