"""DuckDuckGo-based web search tool.

Exposes a CrewAI tool function to perform text searches on DuckDuckGo and
return the top results in a simple formatted string.
"""

from crewai.tools import tool
from ddgs import DDGS

@tool("Search with DuckDuckGo")
def research_ddg_tool(query: str) -> str:
    """Run a DuckDuckGo search and return the top three results.

    Args:
        query: The text query to search for on DuckDuckGo.

    Returns:
        A string with the first three results, each including title, URL,
        and snippet, separated by blank lines.
    """
    with DDGS(verify=False) as ddgs:
        results = ddgs.text(query, region="it-it", safesearch="off", max_results=3)
        return "\n\n".join(
            f"{i+1}. {r['title']}\n{r.get('href') or r.get('url')}\n{r.get('body')}"
            for i, r in enumerate(results)
        )