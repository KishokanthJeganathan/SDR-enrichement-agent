from langchain_core.tools import tool

from .search import web_search


@tool
def lookup_funding(company_name: str) -> str:
    """Search for a company's funding history (rounds, amounts, investors).
    Returns the same JSON shape as web_search — candidate results to read via
    fetch_page, not confirmed facts. Public sources only (press releases,
    news coverage, the company's own site).
    """
    query = f'"{company_name}" funding round OR "Series A" OR "Series B" OR raised'
    return web_search.invoke({"query": query})
