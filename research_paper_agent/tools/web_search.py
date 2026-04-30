"""
tools/web_search.py
────────────────────
Tavily-powered web search used by Node 3 (LLM-as-Judge).

Why Tavily?
  - Designed specifically for LLM agents (returns clean, chunked results)
  - Searches academic sources, arXiv, Semantic Scholar, etc.
  - Better signal-to-noise than raw Google for research queries

Exports:
  search_recent_papers(query, max_results) -> str
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def search_recent_papers(query: str, max_results: int = 5) -> str:
    """
    Searches the web for recent papers, benchmarks, and industry research
    related to the given query.

    Used by Node 3 to give the LLM-judge grounding in current literature.

    Parameters
    ----------
    query : str
        The research topic or specific aspect to search.
        e.g. "catastrophic forgetting transformer fine-tuning 2024"
    max_results : int
        How many search results to include. Default 5.

    Returns
    -------
    str
        Formatted string of search results, ready to be injected into
        the judge prompt as context.

    Example return format:
        === Search Results for: "catastrophic forgetting ..." ===

        [1] Title: "LoRA: Low-Rank Adaptation..." (arxiv.org)
        Summary: This paper proposes...
        URL: https://arxiv.org/abs/...

        [2] Title: ...
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        logger.warning(
            "TAVILY_API_KEY not set. Skipping web search. "
            "Node 3 will judge without external context."
        )
        return (
            "⚠️  Web search unavailable (TAVILY_API_KEY not set). "
            "Judge is working from internal knowledge only."
        )

    try:
        # Lazy import — only load if key is set
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)

        # Search with academic focus
        response = client.search(
            query=query,
            search_depth="advanced",    # Deeper search
            max_results=max_results,
            include_answer=True,        # Get a synthesized answer too
            include_raw_content=False,  # Keep it clean
        )

        # Format the results into a readable string
        lines = [f'=== Search Results for: "{query}" ===\n']

        # Add the synthesized answer if available
        if response.get("answer"):
            lines.append(f"📌 Quick Answer: {response['answer']}\n")

        # Add individual results
        results = response.get("results", [])
        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "No summary available.")

            # Truncate content to avoid token explosion
            if len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"[{i}] {title}")
            lines.append(f"    Source: {url}")
            lines.append(f"    {content}\n")

        return "\n".join(lines)

    except ImportError:
        return (
            "⚠️  tavily-python not installed. "
            "Run: pip install tavily-python"
        )

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return (
            f"⚠️  Web search failed: {str(e)}. "
            "Judge will proceed with internal knowledge."
        )
